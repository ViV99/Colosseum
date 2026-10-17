# SP5 Plan — Part 1: Foundations (contract baselines, frame, protocol, security, config, weight cache)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** This part owns the following parts of `docs/superpowers/specs/2026-10-17-sp5-distributed-design.md` (Russian; read the spec first, it is the binding authority):
- **§6, first bullet, and §7 P0:** contract tests of today's behaviour before the main process moves into the hub — the result flow read from run-dir artefacts, lineup generation and owner rotation over several refreshes and worker counts, and the SP4 run-dir fixture made by unchanged code with its child-run resume test (T0.1). Retention and evictions reaching workers are already pinned by `tests/contract/test_sp3_snapshot_eviction.py` (T2.5 ports it); T0.1 adds nothing there (cross-task notes).
- **Block 4.9 (wire format):** the array frame with its hostile-input checks, adaptive lz4, chunk / weights / checkpoint frames, the learner's chunk checks (`ChunkValidator`), the criterion-5 measurement (T1.1); typed protobuf v1 for every session, blob, relay, data-port, bootstrap and admin message, the converters of the SP4 match record and lineups, `PROTOCOL_VERSION` (T1.2).
- **Blocks 2.1 and 3.1–3.4 at the library level (rulings R1, R15):** the hub identity per machine (CA = TLS server certificate, token secret, node book, ban list, rotation), the token format and its resolution (`--token` > `--token-file` > `COLOSSEUM_TOKEN`), node identities and certificates, gRPC credentials for TLS / `--insecure` / loopback, the R1 interceptor of the hub port and the mutual-TLS interceptor of node data ports, and every refusal of criterion 7 on a real loopback server (T1.3).
- **Block 9.1 (config) and the `transport` part of 4.10:** the global `distributed` section, `learner.max_policy_lag` / `recovery_interval_sec` / `placement.*`, `learner.device` values, the `transport.*` translation, `--set`, the pinned benchmark config (T1.4).
- **Blocks 4.6 and 7.8 at the library level:** the per-machine shared-memory weight cache — immutable segments, pointer files, the node lock, readers that re-read a pointer, cleanup of dead owners only, run- and CA-scoped names, the free-space check, the macOS directory — and the criterion-4 read-latency measurement (T1.5).

Not in this part: anything that runs a hub, a node or a session (part 02), blob streaming in parts (`grpc_util.send_parts` / `receive_parts`, T2.3), same-machine connections (T2.2), the registration handler and its checks (T2.3, T6.3), the node's data-port allow-list (T3.2), the in-place resume of the SP4 fixture (T4.3), the bootstrap bundle (T6.1).

**What the part delivers.** After T1.5 (all six tasks are group A, parallel worktrees; overview «Cross-part execution notes»):
- baseline tests that pin today's result accounting, owner rotation and SP4 run-dir compatibility; T2.5 and T4.3 run them on the hub;
- `colosseum.net.frame` / `colosseum.net.chunks`: one frame format for chunks, weights and checkpoints, decoded without pickle and checked before allocation; the learner's chunk checks of spec 4.9; a measured ≥ 10× faster parse than today's npz path;
- `colosseum.net.proto` (protocol v1, generated code committed) and `colosseum.net.messages`; `grpcio` / `protobuf` / `cryptography` are runtime dependencies;
- `colosseum.net.security` / `colosseum.net.grpc_util`: everything a hub and a node need to trust each other, with every criterion-7 refusal tested on loopback;
- `colosseum.core.config.DistributedConfig` and the new learner keys; SP2–SP4 configs and resolved configs still load (one WARNING for `transport`);
- `colosseum.node.weights`: the shared-memory weight cache, measured at p95 ≪ 100 ms for 50 MB.

**Read `00-overview.md` first** (global constraints, file map, the binding interface contract). Names used here are the contract's; additions and deviations are listed in `## Contract notes` at the end.

**Conventions used in every task.**
- Run every command from the repository root as `env -u PYTHONPATH .venv/bin/python ...` / `env -u PYTHONPATH .venv/bin/ruff ...` (owner's machine: ROS sets `PYTHONPATH`). In a parallel worktree use the main checkout's venv with `PYTHONPATH=<worktree>/src` (process README) — e.g. `env -u PYTHONPATH PYTHONPATH=$PWD/src ../Colosseum/.venv/bin/python -m pytest ...`.
- "Full fast suite" means `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (zero failures, zero warnings), followed by `env -u PYTHONPATH .venv/bin/ruff check .`.
- New test files start with `test_sp5_` and have unique basenames; support modules are imported by bare name and added to `known-first-party` in `pyproject.toml` by the task that creates them (`net_kit`: T1.3). Tests that use `game_harness` live in `tests/contract/`.
- Every file a test writes is under `tmp_path`; fixtures under `tests/fixtures/` are read-only — **a test never opens `tests/fixtures/sp4/run/matches.db` in place** (SQLite in WAL mode creates `-wal` / `-shm` files next to a journal it opens); it reads a `copy_sp4_run(tmp_path)` copy. From T1.3 on the autouse fixture `_isolate_colosseum_dirs` puts `XDG_DATA_HOME`, `XDG_CACHE_HOME` and `COLOSSEUM_SHM_DIR` under `tmp_path` for every test and the processes it starts.
- At most 2 loaded CPU processes per fast test (one training run = 1 worker + 1 learner); processes through `tests/cli_runner.py`; durations with `time.monotonic()` / `time.perf_counter()`; data ordered by counters, never by `ts`. Messages go through `logging` (never `warnings.warn`), so `-rw` stays empty.
- Commit messages use conventional prefixes and carry no attribution or co-author lines; push after every task (`git push`; group-A tasks push their worktree branch, the controller integrates by cherry-pick).

**Cross-task notes.**
- **T0.1 runs on the unchanged tree** (only plan commits since `6fa6d8a`); its generator runs SP4's `train`, `pin` and `tournament`. It is committed with its output, like `tests/fixtures/sp3/`. Its tests are baselines (green on SP4 code, green through SP5): T2.5 reruns `test_sp5_rotation_baseline.py` (the rotation through `OwnerRotation`, spec 4.2) and `test_sp5_result_flow_baseline.py` on the hub; T4.3 adds the in-place `--resume` of an SP4 fixture copy; T7.1 extends the result-flow test with `dropped_match_results_by_reason`. T2.1's rotation test imports `owner_sequence` from `test_sp5_rotation_baseline.py`'s expectation (copy the 3-line function into `hub_kit` or `game_helpers` if a cross-test import is awkward; never change its formula).
- **Retention and evictions reaching workers** (spec §6 lists them) are already pinned on today's code by `tests/contract/test_sp3_snapshot_eviction.py` (the coordinator hands each eviction out once, the launcher sends evictions to every worker and retries a full queue, a worker unloads a snapshot once no lineup uses it), `tests/unit/test_sp3_snapshot_retention.py` and `tests/unit/test_sp4_retention.py`. T0.1 adds no copy of them; T2.5 ports `test_sp3_snapshot_eviction.py` to the hub (evictions as `Announce.evictions`, unload at the node) with the same assertions, and T3.1 adds the evicted-snapshot hold.
- **File split inside group A** (overview): `pyproject.toml` — T1.2 edits `[project].dependencies`, the extras and the ruff excludes, T1.3 adds `cryptography` and `net_kit` (`known-first-party`); `core/types.py` — T1.2 the three dataclass fields, **and T1.1 only `validate_slot_structure`** (its numpy core `check_slot_arrays`, needed for criterion 5; disjoint hunks, deviation 3 in the contract notes); `core/config.py` — only T1.4; `core/validation.py` — only T1.1; `distributed.py` — only T1.4 (one line); `tests/conftest.py` — only T1.3; `tests/game_helpers.py` and `.gitignore` — only T0.1; `scripts/bench_throughput.py` — only T1.4.
- **Imports inside the new packages.** `colosseum.net.frame` imports numpy and lz4 only; `colosseum.net.chunks` imports `frame`, `colosseum.core.{config,errors,specs,types}`, `colosseum.envs.game`, `colosseum.networks.state`; `colosseum.net.messages` imports `colosseum.core.types` and the generated proto; `colosseum.net.security` imports grpc and cryptography only; `colosseum.net.grpc_util` imports `security`; `colosseum.node.weights` imports `colosseum.net.{chunks,frame}` and `colosseum.core.run_state`. None of them starts a thread or a process. T1.5 imports T1.1's `colosseum.net.chunks`: in its worktree it is developed against T1.1's branch (cherry-pick T1.1 first) — the controller integrates T1.1 before T1.5.
- **`colosseum.transport.serialization` stays until T7.2.** T1.1 only compares with it (one equality test, the measurement's "old" side); T7.2 deletes that test with the module and keeps `scripts/measure_frame_decode.py` working by dropping its "old" column (the T1.1 result is in its task report and in `docs/benchmarks.md` via T7.4).

**Verified while writing this part.** All library code and test code of T1.1–T1.5 below was run in a scratch copy of `src/` (`/tmp/.../scratchpad/part01/`, the repo's `.venv`: Python 3.12.15, numpy 2.5.4, torch 2.14.1, grpcio / grpcio-tools 1.84.0, protobuf 7.36.2, cryptography 50.0.2, lz4 4.4.5): the new test files pass (frame 61, chunks 48, messages 18, security 17, config 25, weight cache 9 tests), ruff (the repo's rules) is clean, and **with every change of this part applied** the repo's whole `tests/unit` + `tests/contract` fast set passes (1974 tests; the two `test_bench_throughput` failures seen there were the unchanged repo copy of the script and a path artefact of the scratch copy — the edited script passes its guards). `protoc` (grpcio-tools 1.84) generated the v1 code: gencode protobuf 7.35.1, `GRPC_GENERATED_VERSION` 1.84.0 — the dependency floors of T1.2. Measured on the owner's machine (not quiet, so only indicative; the tasks measure again): criterion 5 old (lz4 npz + `validate_chunk_payload`) / new (plain frame + every check) = 13.3–14.5 (1146–1222 µs vs 84–85 µs; old plain / new 11.6–12.0; up to a built `TrajectoryChunk` 9.1–9.6, see ruling P1-1); criterion 4 50 MB read while publishing twice a second: median 1.4 ms, p95 1.7 ms, max 36 ms, 6 269 reads, 24 versions, no torn read. T0.1: the SP4 fixture generator ran twice on the unchanged tree (≈ 11 s each; 5–6 snapshots, one pinned, ≈ 800–1 060 training, 12 pulse and 8 tournament matches, ≈ 700 KB); a child-run resume from a copy with SP4 code passed with the log lines the test checks; `test_sp5_rotation_baseline.py` (28 tests) and `test_sp5_result_flow_baseline.py` passed on the unchanged tree; the SP4 compat tests passed against the generated fixture. Security: the prototype facts of R1 / R15 (grpc-python cannot request an optional client certificate; the CA as the TLS leaf works for grpc, `ssl.get_server_certificate` and curl's pin; node certificates give mutual TLS on data ports) are reproduced by the T1.3 tests.

**Traceability (spec item → task / test).**

| Spec | Item | Task: test |
|---|---|---|
| §6 | result flow before the move (journal exactly once or counted) | T0.1: `test_sp5_result_flow_baseline.py::test_every_finished_match_is_journaled_once_or_counted` (+ the existing `test_sp4_result_flow.py`, `test_sp4_journal_runs.py`) |
| §6 | lineups and owner rotation, `test_owner_rotates_with_global_env_index_and_round` and neighbours | T0.1: `test_sp5_rotation_baseline.py` (27 cases + the launcher refresh) |
| §6 | retention and evictions | existing `test_sp3_snapshot_eviction.py`, `test_sp3_snapshot_retention.py`, `test_sp4_retention.py` (cross-task notes) |
| §6, crit. 6 | SP4 run dir: fixture, child resume | T0.1: `test_sp5_sp4_compat.py`, `test_sp5_sp4_resume.py` (in-place: T4.3) |
| 4.9 | frame layout, zero-copy views, all node types, lz4 flag | T1.1: `test_sp5_frame.py` round trips, `test_the_byte_layout_is_the_contracts`, `test_arrays_are_views_of_the_buffer_unless_copied` |
| 4.9 | hostile input: header ≤ 1 MiB, sizes ≤ cap before allocation, offsets in bounds / non-overlapping, dtype whitelist, known namedtuples only, no pickle, lz4 bomb, negative sizes | T1.1: `test_sp5_frame.py` hostile-input section (13 tests, 50+ cases) |
| 4.9 | chunk checks: slot structure, obs / gs / action / mask shapes and dtypes, finite `behavior_logp`, zero rewards on BOOT / PAD, `teacher_action` in the action space, `has_teacher` bool | T1.1: `test_sp5_chunks.py::test_the_validator_refuses_each_broken_rule`, `test_units_and_dict_actions_are_range_checked_per_component`, `test_chunks_the_worker_buffers_build_pass_after_a_round_trip` |
| 4.9 | adaptive lz4 (probe, ≥ 1.3, reprobe 10 min), weights never compressed | T1.1: `test_the_probe_*`, `test_weights_round_trip_with_their_meta_and_are_never_compressed` |
| 4.9 | weights frames with agent, generation, version, `teacher_active`, architecture | T1.1: `test_weights_round_trip_*` |
| 4.9, §6 | same content as the npz format | T1.1: `test_frame_and_npz_formats_decode_to_the_same_content` |
| crit. 5 | ≥ 10× faster than npz + `validate_chunk_payload` | T1.1: `scripts/measure_frame_decode.py` (rule fixed in the script) |
| 3.4 | no pickle, caps before allocation; `weights_only` trainer state | T1.1: frame tests; `test_checkpoints_round_trip_with_the_trainer_state_as_bytes` |
| 4.9, 4.1 | typed protobuf, session / blob / relay / data-port / bootstrap / admin messages, `int64` versions and counters, protocol version | T1.2: `test_sp5_messages.py` |
| 4.3 | match record schema = SP4's + seat generation + node | T1.2: `test_every_match_record_variant_round_trips_through_bytes` |
| 2.1 | hub identity per machine, token `cst1.<secret>.<fp>`, rotate, node identity dir, `--token-file` / `COLOSSEUM_TOKEN` | T1.3: `test_the_token_format_and_its_resolution`, `test_the_hub_identity_*`, `test_rotate_*`, `test_a_node_identity_*` |
| 2.2 (ban) | `--ban` by node id and machine id | T1.3: `test_a_ban_refuses_the_node_id_and_the_machine_id`, `test_node_calls_need_a_registered_not_banned_node_and_its_secret` |
| 3.2, R1, R15 | TLS with the pinned CA, fixed server names, node secret on the hub port, mutual TLS on data ports | T1.3: loopback tests |
| 3.3, crit. 7 | no token / wrong token / wrong fingerprint / another CA / no TLS / `--insecure` still needs the token / banned | T1.3: the seven refusal tests |
| 3.6 | `cryptography` dependency | T1.3 (pyproject) |
| 9.1 | `distributed.*` defaults, global only, `learner.*` keys with per-agent override, port set check, `--set`, `_make_config` | T1.4: `test_sp5_config.py`, `test_bench_throughput.py` guards |
| 4.10 | `transport.mode` / `grpc_port` ignored with a WARNING, `grpc_max_message_mb` → `distributed.max_message_mb`, both = ConfigError | T1.4: `test_the_transport_section_is_translated_with_one_warning`, `test_old_and_new_message_size_together_are_an_error` |
| 4.6 | immutable segments + pointer, mmap readers, retry on a vanished segment, dead owners only, names per CA / run / node, free-space check with the hint | T1.5: `test_sp5_weight_cache.py` |
| 7.8 | `/dev/shm` on Linux, temp dir on macOS, process liveness by pid + start time | T1.5: `test_the_cache_dir_and_the_prefix`, `test_the_node_lock_and_cleanup_of_dead_owners_only` |
| crit. 4 (2nd half) | 50 MB read p95 ≤ 100 ms at 2 publishes/s | T1.5: `scripts/measure_weight_cache.py` |

---

### Task T0.1: Contract tests of today's behaviour; SP4 run-dir fixture and its resume test

Spec §6 first bullet ("Контрактные тесты до переноса") and §7 P0; criterion 6 (an SP4 run dir continues). Before the main process becomes the hub, unchanged SP4 code produces a small SP4 run dir (journal with training, pulse and tournament matches, league and pulse state, a pinned snapshot, run identity and state), and four baseline test files pin what SP5 must keep: the run-level result accounting read from run-dir artefacts only, the owner rotation of today's launcher over several refreshes and worker counts (spec 4.2: "на одном узле ... даёт сегодняшнюю последовательность `trainable[(g + round) % n]`"), the SP4 run dir's shape and readability, and a child-run resume (`training.resume_from`) from it.

**Files:**
- Create: `tests/fixtures/sp4/sp4_ttt_tiny.yaml`, `tests/fixtures/sp4/make_sp4_run.py` (generator, run once now; committed with its output)
- Create (generated): `tests/fixtures/sp4/run/{config.resolved.yaml,run.json,run_state.json,metrics.jsonl,ratings.json,matches.db,league_state.json,pulse_state.json,checkpoints/main/ckpt_v*/...,tournaments/t0/...}`, `tests/fixtures/sp4/manifest.json`
- Modify: `.gitignore` (the repo ignores `checkpoints/` and `*.pt`), `tests/game_helpers.py` (`SP4_FIXTURES`, `SP4_TTT_TINY`, `SP4_RUN`, `sp4_manifest`, `copy_sp4_run`)
- Test: `tests/contract/test_sp5_rotation_baseline.py`, `tests/integration/test_sp5_result_flow_baseline.py`, `tests/unit/test_sp5_sp4_compat.py`, `tests/integration/test_sp5_sp4_resume.py`

**Interfaces:**
- Consumes (SP4, unchanged): `colosseum train` / `pin` / `tournament` / `report` (CLI), `Coordinator.generate_lineups` / `next_round` / `refresh_round`, `Launcher._refresh_worker_matches`, `MatchLogReader`, `load_checkpoint_dir`, `check_model_state`, `read_run_id`, `cli_runner.run_train`, `game_helpers.{make_coordinator, make_test_config, tree_digest}`.
- Produces (test support, `tests/game_helpers.py`): `SP4_FIXTURES: Path`, `SP4_TTT_TINY: Path`, `SP4_RUN: Path`, `sp4_manifest() -> dict` (`agent`, `anchor`, `layout`, `run_id`, `versions`, `final_version`, `pinned`, `with_trainer_state`, `env_steps`, `league_env_steps`, `journal: {by_source, schema_version, counts_rows}`, `tournament`, `files`), `copy_sp4_run(tmp_path) -> Path`; `test_sp5_rotation_baseline.owner_sequence(workers, envs, agents, rounds)` (the expectation T2.1 / T2.5 reuse).

- [ ] **Step 1: Write the fixture config and the generator**

Run `git status` first: only plan files may differ from `6fa6d8a` (the tree is SP4's code).

Create `tests/fixtures/sp4/sp4_ttt_tiny.yaml`:
```yaml
# SP4-form config of the SP4 run-dir fixture (tests/fixtures/sp4/run), SP5 task T0.1: tic-tac-toe with a tiny MLP, a
# trainable agent 'main' and a RandomBot anchor 'random'; the journal, live ratings and pulse are on, so the run dir
# holds matches.db (training, pulse and, after the generator's tournament, tournament matches), league_state.json and
# pulse_state.json. keep_last 3 + keep_every 2 at interval 20 leave snapshots with and without trainer_state.pt.
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
evaluation:
  enabled: true
  field: [random]
  every: 2
  blocks: 2
  num_envs: 2
ratings:
  refit_interval_sec: 2.0
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

Create `tests/fixtures/sp4/make_sp4_run.py`:
```python
"""Generator of the SP4 run-dir fixture (SP5 task T0.1). Run ONCE on unchanged SP4 code; the output is committed.

    env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python tests/fixtures/sp4/make_sp4_run.py

Trains ``sp4_ttt_tiny.yaml`` with ``colosseum train`` in a temporary directory, pins the second-newest snapshot
(``colosseum pin``), plays a small gauntlet on the run (``colosseum tournament --run``) and copies what an SP4 run
dir holds into ``tests/fixtures/sp4/run/``: ``config.resolved.yaml`` (``run.dir`` rewritten to ``runs``), ``run.json``,
``run_state.json``, ``metrics.jsonl``, ``ratings.json``, ``matches.db`` (its WAL checkpointed into the file first),
``league_state.json``, ``pulse_state.json``, ``checkpoints/`` and ``tournaments/`` (every ``logs/`` is left out: they
hold machine paths). It then writes ``manifest.json`` with what tests need to know about the run: training is
asynchronous, so the snapshot versions and the match counts differ between generations. It is a record of how the
fixture was made; SP5 tasks never re-run it (the code after T1.1 makes SP5 run dirs).
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
CONFIG = HERE / "sp4_ttt_tiny.yaml"
AGENT = "main"
ANCHOR = "random"
TOURNAMENT = "t0"
FILES = ("config.resolved.yaml", "run.json", "run_state.json", "metrics.jsonl", "ratings.json", "matches.db",
         "league_state.json", "pulse_state.json")


def _colosseum(*args: str) -> None:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "WANDB_MODE": "disabled"}
    subprocess.run([sys.executable, "-m", "colosseum", *args], cwd=REPO_ROOT, env=env, check=True)


def _journal_counts(db: Path) -> dict:
    conn = sqlite3.connect(db)
    try:
        sources = dict(conn.execute("SELECT e.name, COUNT(*) FROM matches m JOIN entities e ON e.id = m.source "
                                    "GROUP BY e.name").fetchall())
        schema = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
        counts = conn.execute("SELECT COUNT(*) FROM counts").fetchone()[0]
        runs = [r[0] for r in conn.execute("SELECT run_id FROM runs ORDER BY id")]
    finally:
        conn.close()
    return {"by_source": {k: int(v) for k, v in sorted(sources.items())}, "schema_version": int(schema),
            "counts_rows": int(counts), "run_ids": runs}


def main() -> None:
    dst = HERE / "run"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "sp4_run"
        _colosseum("train", "-c", str(CONFIG), "--set", f"run.dir={tmp}", "--set", "run.name=sp4_run")
        versions = sorted(int(p.name.removeprefix("ckpt_v")) for p in (src / "checkpoints" / AGENT).glob("ckpt_v*"))
        assert len(versions) >= 3, versions
        pinned = versions[-2]
        _colosseum("pin", str(src), f"{AGENT}@v{pinned}")
        _colosseum("tournament", "--run", str(src), "--snapshots", "last:2", "--mode", "gauntlet", "--blocks", "2",
                   "--seed", "0", "--num-workers", "1", "--num-envs", "2", "--name", TOURNAMENT)
        conn = sqlite3.connect(src / "matches.db")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()
        assert not (src / "matches.db-wal").exists() or (src / "matches.db-wal").stat().st_size == 0
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir()
        for name in FILES:
            shutil.copy2(src / name, dst / name)
        resolved = yaml.safe_load((src / "config.resolved.yaml").read_text())
        resolved["run"]["dir"] = "runs"
        (dst / "config.resolved.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))
        shutil.copytree(src / "checkpoints", dst / "checkpoints")
        shutil.copytree(src / "tournaments", dst / "tournaments", ignore=shutil.ignore_patterns("logs"))
    agent_dir = dst / "checkpoints" / AGENT
    metas = {p.name: json.loads((p / "meta.json").read_text()) for p in sorted(agent_dir.iterdir())}
    finals = [m["policy_version"] for m in metas.values() if m["final"]]
    assert len(finals) == 1, finals
    run_id = json.loads((dst / "run.json").read_text())["run_id"]
    journal = _journal_counts(dst / "matches.db")
    assert journal["run_ids"] == [run_id] and set(journal["by_source"]) == {"train", "pulse", "tournament"}, journal
    manifest = {
        "agent": AGENT,
        "anchor": ANCHOR,
        "layout": "2p",
        "run_id": run_id,
        "versions": sorted(m["policy_version"] for m in metas.values()),
        "final_version": finals[0],
        "pinned": sorted(m["policy_version"] for m in metas.values() if m.get("pinned")),
        "with_trainer_state": sorted(m["policy_version"] for name, m in metas.items()
                                     if (agent_dir / name / "trainer_state.pt").exists()),
        "env_steps": max(int(m["env_steps"]) for m in metas.values()),
        "league_env_steps": json.loads((dst / "league_state.json").read_text())["env_steps"],
        "journal": {k: journal[k] for k in ("by_source", "schema_version", "counts_rows")},
        "tournament": TOURNAMENT,
        "files": sorted(p.name for p in dst.iterdir()),
    }
    assert manifest["pinned"] == [pinned] and len(manifest["with_trainer_state"]) < len(manifest["versions"])
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {dst}: {manifest}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the run dir and let git see it**

Append to `.gitignore` (after the SP3 fixture lines):
```
# SP4 run-dir fixture (SP5 T0.1): committed SP4 snapshots and tournament snapshot links inside an SP4-shaped run dir
!/tests/fixtures/sp4/run/checkpoints/
!/tests/fixtures/sp4/run/checkpoints/**
!/tests/fixtures/sp4/run/tournaments/
!/tests/fixtures/sp4/run/tournaments/**
```
Run:
```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python tests/fixtures/sp4/make_sp4_run.py
git status --porcelain -uall tests/fixtures/sp4 .gitignore
du -sh tests/fixtures/sp4
```
Expected: about 11 s; `wrote .../tests/fixtures/sp4/run: {...}` with 4–6 versions (two without trainer state), one pinned (the second-newest), `env_steps` a little above 6000, journal `by_source` ≈ `{"pulse": 12, "tournament": 8, "train": 800–1100}`, `schema_version` 1; `git status` lists the config, the generator, `manifest.json` and every file under `run/` (including `checkpoints/main/ckpt_v*/model.pt` and `tournaments/t0/snapshots/.../model.pt`); about 700 KB. No `matches.db-wal` / `-shm` and no `logs/` may be listed. If an assertion of the generator fails (fewer than three snapshots), raise `training.total_timesteps` in `sp4_ttt_tiny.yaml` by 2000 and run it again.

- [ ] **Step 3: Add the fixture paths to `tests/game_helpers.py`**

Append after the SP3 fixture block (`copy_sp3_run` / `tree_digest`):
```python
# ---------------------------------------------------------------------------
# SP5 (T0.1): SP4 compatibility fixture (read-only input; copy before writing — and before opening its journal)
# ---------------------------------------------------------------------------


SP4_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sp4"
SP4_TTT_TINY = SP4_FIXTURES / "sp4_ttt_tiny.yaml"
SP4_RUN = SP4_FIXTURES / "run"


def sp4_manifest() -> dict:
    """What the SP4 run fixture holds (``tests/fixtures/sp4/manifest.json``, written by its generator)."""
    return json.loads((SP4_FIXTURES / "manifest.json").read_text())


def copy_sp4_run(tmp_path) -> Path:
    """A writable copy of the SP4 run fixture under ``tmp_path``. SQLite creates ``-wal`` / ``-shm`` files next to a
    journal it opens, so a test opens ``matches.db`` only in such a copy."""
    dst = Path(tmp_path) / "sp4_run"
    shutil.copytree(SP4_RUN, dst)
    return dst
```

- [ ] **Step 4: Write the baseline tests**

Create `tests/contract/test_sp5_rotation_baseline.py`:
```python
"""Owner rotation and lineups on today's code (SP5 T0.1, spec §6 "Контрактные тесты до переноса", block 4.2): the
sequence of owners the launcher produces over several refreshes for 1-3 workers, 1-3 envs per worker and 1-3
trainable agents; every lineup's owner collects; the launcher's refresh advances one round and uses global env
offsets. T2.1 runs ``owner_sequence``'s expectation against ``hub.state.OwnerRotation`` (one node, every cell expiring
at once); T2.5 ports the launcher test to the hub (same assertions)."""
from __future__ import annotations

import queue

import pytest

from colosseum.launcher import Launcher
from game_helpers import make_coordinator, make_test_config

ROUNDS = 5
CASES = [(workers, envs, agents) for workers in (1, 2, 3) for envs in (1, 2, 3) for agents in (1, 2, 3)]


def _config(workers: int, envs: int, agents: int):
    ids = ["a", "b", "c"][:agents]
    return make_test_config("turns", agents={aid: {} for aid in ids},
                            rollout={"num_workers": workers, "envs_per_worker": envs},
                            matchmaking={"opponents": {"latest": 1.0, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.0}})


def _owner(lineup) -> str:
    collecting = {s.agent_id for s in lineup.seats if s.collect and s.source in ("owner", "")}
    owners = {s.agent_id for s in lineup.seats if s.source == "owner"}
    assert owners and owners <= collecting, lineup
    return owners.pop()


def owner_sequence(workers: int, envs: int, agents: int, rounds: int = ROUNDS) -> list[list[list[str]]]:
    """``[round][worker][env]`` -> owner: what ``trainable[(g + round) % n]`` gives for global env ``g = w * E + e``."""
    ids = ["a", "b", "c"][:agents]
    return [[[ids[(w * envs + e + r) % agents] for e in range(envs)] for w in range(workers)] for r in range(rounds)]


@pytest.mark.parametrize("workers, envs, agents", CASES)
def test_the_owner_sequence_over_refreshes(tmp_path, workers, envs, agents):
    """The launcher's order: initial lineups at round 0 (``_start_children``), then per refresh ``next_round`` and one
    ``generate_lineups(E, w * E)`` per worker (``_refresh_worker_matches``)."""
    coord = make_coordinator(_config(workers, envs, agents), tmp_path / "ckpt")
    seen = []
    for r in range(ROUNDS):
        if r:
            coord.next_round()
        seen.append([[_owner(lu) for lu in coord.generate_lineups(envs, env_offset=w * envs)] for w in range(workers)])
    assert seen == owner_sequence(workers, envs, agents)
    flat = [o for rnd in seen for worker in rnd for o in worker]
    counts = {aid: flat.count(aid) for aid in set(flat)}
    if (workers * envs) % agents == 0:
        assert len(set(counts.values())) == 1                     # equal shares when the cells divide evenly


def test_the_launcher_refresh_advances_one_round_with_global_env_offsets(tmp_path):
    cfg = _config(3, 2, 3)
    coord = make_coordinator(cfg, tmp_path / "ckpt")
    launcher = Launcher.__new__(Launcher)                         # _refresh_worker_matches needs only _config
    launcher._config = cfg
    queues = [queue.Queue(maxsize=1) for _ in range(3)]
    sent = [{aid: set() for aid in ("a", "b", "c")} for _ in range(3)]
    expected = owner_sequence(3, 2, 3, rounds=3)
    for r in (1, 2):
        launcher._refresh_worker_matches(coord, ["a", "b", "c"], queues, sent)
        assert coord.refresh_round == r
        assert [[_owner(lu) for lu in q.get_nowait().lineups] for q in queues] == expected[r]
```

Create `tests/integration/test_sp5_result_flow_baseline.py`:
```python
"""The result flow of a training run, read only from run-dir artefacts (SP5 T0.1, spec §6 "Контрактные тесты до
переноса", block 4.3): every match a worker finished is in the journal exactly once or counted in
``dropped_match_results``; uids are unique, carry the run id and name the worker, env and episode; the journal saw no
duplicate. Written on today's code; T2.5 runs it on the hub + local node unchanged except ``_finished_episodes``
(the worker logs' "finished" line), and T4.3 / T7.1 extend it (node column, ``by_reason``)."""
from __future__ import annotations

import ast
import re
import sqlite3

from cli_runner import TTT_CONFIG, run_train
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader
from colosseum.core.run_dir import read_run_id

BUDGET = 8000
UID_RE = re.compile(r"^(?P<run>[0-9a-f]{32})/train/(?:n(?P<n>\d+)/)?w(?P<w>\d+)_e(?P<e>\d+)_(?:g(?P<g>\d+)_)?"
                    r"ep(?P<k>\d+)$")


def _finished_episodes(run) -> int:
    """Episodes every worker finished, from each worker log's last "finished" line."""
    total = 0
    for log in sorted((run.root / "logs").glob("worker-*.log")):
        lines = [line for line in log.read_text().splitlines() if ": finished. " in line]
        if lines:
            total += int(ast.literal_eval(lines[-1].split("finished. ", 1)[1])["episodes"])
    return total


def _journal_rows(path) -> int:
    conn = sqlite3.connect(path)
    try:
        detailed = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        counted = conn.execute("SELECT COALESCE(SUM(n), 0) FROM counts WHERE kind = 0").fetchone()[0]
    finally:
        conn.close()
    return detailed + counted                     # tic-tac-toe: one team pair per match


def test_every_finished_match_is_journaled_once_or_counted(tmp_path):
    run = run_train(TTT_CONFIG, tmp_path, name="flow", overrides={"training.total_timesteps": str(BUDGET),
                                                                   "match_log.detail_window_mb": "512"})
    assert run.returncode == 0, run.stderr[-3000:]
    finished = _finished_episodes(run)
    system = run.records("system")[-1]
    assert finished > 100
    assert _journal_rows(run.root / MATCH_LOG_FILE) + system["dropped_match_results"] == finished
    assert system["match_log"]["duplicates"] == 0
    run_id = read_run_id(run.root)
    with MatchLogReader(run.root / MATCH_LOG_FILE) as reader:
        rows = reader.matches(source="train", limit=None)
    uids = [r.uid for r in rows]
    assert len(uids) == len(set(uids)) == _journal_rows(run.root / MATCH_LOG_FILE)
    parsed = [UID_RE.match(uid) for uid in uids]
    assert all(m is not None and m["run"] == run_id for m in parsed), uids[:5]
    episodes = {(m["n"], m["w"], m["e"], m["g"], m["k"]) for m in parsed}
    assert len(episodes) == len(uids)                             # one row per (node, worker, env, epoch, episode)
```

Create `tests/unit/test_sp5_sp4_compat.py`:
```python
"""SP4 compatibility baseline (SP5 T0.1, spec §6 and criterion 6): the SP4 run fixture is an SP4 run dir — journal of
schema version 1 with training, pulse and tournament matches, league and pulse state, a pinned snapshot, run identity
and state — whose snapshots load read-only, whose journal reads without a change, and that `colosseum report` renders.
Created on SP4 code; it stays green through SP5 (T4.3 adds the in-place resume of a copy)."""
from __future__ import annotations

import json

from click.testing import CliRunner

from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader
from colosseum.cli import main
from colosseum.coordinator.checkpoint_manager import check_model_state, load_checkpoint_dir
from colosseum.core.config import NetworkConfig, load_config
from colosseum.core.registry import build_network, env_spec
from colosseum.core.run_dir import read_run_id
from game_helpers import SP4_RUN, SP4_TTT_TINY, copy_sp4_run, sp4_manifest, tree_digest

SP4_META_KEYS = {"agent_id", "checkpoint_id", "config_hash", "env_steps", "final", "networks", "pinned",
                 "policy_version", "role_signature", "roles", "run_id", "timestamp"}


def test_the_run_fixture_is_an_sp4_run_dir():
    manifest = sp4_manifest()
    assert sorted(p.name for p in SP4_RUN.iterdir()) == manifest["files"]
    assert not any(p.name == "logs" for p in SP4_RUN.rglob("*"))           # machine paths stay out
    assert read_run_id(SP4_RUN) == manifest["run_id"]
    agent_dir = SP4_RUN / "checkpoints" / manifest["agent"]
    versions = sorted(int(p.name.removeprefix("ckpt_v")) for p in agent_dir.iterdir())
    assert versions == manifest["versions"] and manifest["final_version"] == max(versions)
    for version in versions:
        meta = json.loads((agent_dir / f"ckpt_v{version}" / "meta.json").read_text())
        assert set(meta) == SP4_META_KEYS and meta["run_id"] == manifest["run_id"]
        assert meta["pinned"] is (version in manifest["pinned"]) and meta["final"] is (version == max(versions))
    assert json.loads((SP4_RUN / "league_state.json").read_text())["version"] == 1
    assert (SP4_RUN / "pulse_state.json").is_file()
    assert json.loads((SP4_RUN / "run_state.json").read_text())["status"] == "finished"
    assert {"bt", "layouts", "env_steps"} <= set(json.loads((SP4_RUN / "ratings.json").read_text()))
    assert (SP4_RUN / "tournaments" / manifest["tournament"] / "report.json").is_file()
    resolved = load_config(SP4_RUN / "config.resolved.yaml")
    assert resolved.get_trainable_agent_ids() == [manifest["agent"]] and resolved.evaluation.enabled


def test_the_sp4_journal_reads_on_a_copy(tmp_path):
    """SQLite in WAL mode creates -wal / -shm files next to a journal it opens: journals are read on copies only."""
    manifest = sp4_manifest()
    copy = copy_sp4_run(tmp_path)
    with MatchLogReader(copy / MATCH_LOG_FILE) as reader:
        counts = {s: len(reader.matches(source=s, limit=None)) for s in ("train", "pulse", "tournament")}
        assert [r.run_id for r in reader.runs()] == [manifest["run_id"]]
    assert counts == manifest["journal"]["by_source"] and manifest["journal"]["schema_version"] == 1
    assert not list(SP4_RUN.glob(f"{MATCH_LOG_FILE}-*"))


def test_every_sp4_snapshot_loads_read_only_with_its_architecture():
    manifest = sp4_manifest()
    before = tree_digest(SP4_RUN)
    spec = env_spec(load_config(SP4_TTT_TINY))
    for version in manifest["versions"]:
        ckpt = load_checkpoint_dir(SP4_RUN / "checkpoints" / manifest["agent"] / f"ckpt_v{version}")
        model = build_network(NetworkConfig.model_validate(ckpt["meta"]["networks"]), spec.roles["player"])
        check_model_state(model, ckpt["model_state"], f"ckpt_v{version}")
    assert tree_digest(SP4_RUN) == before


def test_the_report_of_a_copy_renders(tmp_path):
    copy = copy_sp4_run(tmp_path)
    out = tmp_path / "report.html"
    result = CliRunner().invoke(main, ["report", str(copy), "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert out.stat().st_size > 10_000
```

Create `tests/integration/test_sp5_sp4_resume.py`:
```python
"""SP4 compatibility baseline (SP5 T0.1, criterion 6): `colosseum train` with ``training.resume_from`` = a copy of the
SP4 run fixture starts a child run that continues the counters, carries the pool over (the pinned snapshot too),
restores the league state and imports the parent journal; the source is only read. T4.3 adds the in-place
``--resume`` of a copy (journal migration, same run id)."""
from __future__ import annotations

from cli_runner import run_train
from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLogReader
from colosseum.core.run_dir import read_run_id
from game_helpers import SP4_TTT_TINY, copy_sp4_run, sp4_manifest, tree_digest


def _digest(root) -> dict[str, str]:
    """``tree_digest`` without SQLite's -wal / -shm side files (opening a journal may create them)."""
    return {k: v for k, v in tree_digest(root).items() if not k.endswith(("-wal", "-shm"))}


def test_training_resumes_from_the_sp4_run_dir_as_a_child_run(tmp_path):
    manifest = sp4_manifest()
    source = copy_sp4_run(tmp_path)
    before = _digest(source)
    budget = manifest["env_steps"] + 2000
    run = run_train(SP4_TTT_TINY, tmp_path, name="child", tiny=False,
                    overrides={"training.resume_from": str(source), "training.total_timesteps": str(budget),
                               "evaluation.enabled": "false"})
    assert run.returncode == 0, run.stderr[-3000:]
    log = run.log("main")
    assert f"Resume [{manifest['agent']}]:" in log and f"(policy_version {manifest['final_version']})" in log
    assert f"env-step counter continues from {manifest['env_steps']}" in log
    assert "league state (ratings, PFSP statistics) restored from" in log
    assert "snapshot pool carried over" in log
    copied = manifest["journal"]["by_source"]["pulse"] + manifest["journal"]["by_source"]["tournament"]
    assert f"{copied} tournament / pulse matches copied" in log
    assert read_run_id(run.root) != manifest["run_id"]
    with MatchLogReader(run.root / MATCH_LOG_FILE) as reader:
        assert [r.run_id for r in reader.runs()][1:] == [manifest["run_id"]]
        assert len(reader.matches(source="tournament", limit=None)) == manifest["journal"]["by_source"]["tournament"]
    stored = sorted(int(p.name.removeprefix("ckpt_v")) for p in (run.root / "checkpoints" / "main").glob("ckpt_v*"))
    assert set(manifest["pinned"]) <= set(stored) and max(stored) > manifest["final_version"]
    assert _digest(source) == before
```

- [ ] **Step 5: Run the baseline tests (they pass on SP4 code)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/contract/test_sp5_rotation_baseline.py tests/integration/test_sp5_result_flow_baseline.py tests/unit/test_sp5_sp4_compat.py tests/integration/test_sp5_sp4_resume.py -q -rw`
Expected: 34 passed (27 rotation cases + the launcher refresh, 1 result flow, 4 compat, 1 resume), about 25 s (two training runs). This is a baseline, not a failing-first test: if anything fails, the fixture or the test is wrong — fix those, never the code. Then `git status --porcelain tests/fixtures` must be empty (no `-wal` / `-shm` written into the fixture).

- [ ] **Step 6: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings; ruff clean.

- [ ] **Step 7: Commit and push**

```bash
git add .gitignore tests/fixtures/sp4 tests/game_helpers.py tests/contract/test_sp5_rotation_baseline.py \
        tests/integration/test_sp5_result_flow_baseline.py tests/unit/test_sp5_sp4_compat.py \
        tests/integration/test_sp5_sp4_resume.py
git commit -m "test: SP4 run-dir fixture and SP5 baselines (result flow, owner rotation, SP4 compatibility, child resume)"
git push
```

---

### Task T1.1: Frame codec, chunk / weights / checkpoint frames, `ChunkValidator`; criterion-5 script

Spec block 4.9 (frame, hostile input, chunk checks, adaptive lz4), 3.4 (no pickle, caps before allocation), §6 "Формат кадра", criterion 5. New pure modules `colosseum.net.frame` and `colosseum.net.chunks` (complete code below). `validate_slot_structure` gets a numpy core `check_slot_arrays` (same rules, same messages) so the learner checks a decoded payload before it builds torch tensors (criterion 5; ruling P1-1). `core.validation._check_state_payload` checks a model's state on the frame codec instead of the SP1 npz codec. The SP1 codec (`colosseum.transport.serialization`) stays until T7.2; one test compares the two by content.

**Files:**
- Create: `src/colosseum/net/__init__.py`, `src/colosseum/net/frame.py`, `src/colosseum/net/chunks.py`, `scripts/measure_frame_decode.py`
- Modify: `src/colosseum/core/types.py` (`validate_slot_structure` → `check_slot_arrays`; nothing else of the file), `src/colosseum/core/validation.py` (`_check_state_payload`)
- Test: `tests/unit/test_sp5_frame.py`, `tests/unit/test_sp5_chunks.py`

**Interfaces:**
- Consumes: `TrajectoryChunk.to_payload` / `from_payload`, `ObsSpec` / `ActionSpec` (`allocate`, `allocate_actions`, `full_mask`, `num_deciders`, `groups`, `_get`), `split_unit_actions`, `check_agent_id`, `RoleSpec`, `state_to_numpy`; test support `game_helpers.{chunk_v2_payload, learner_role}`, `colosseum.worker.buffers.{BufferSpec, RolloutBuffer}`.
- Produces (contract `colosseum.net.frame` / `colosseum.net.chunks`, plus additions marked * — contract notes 1–3): `FRAME_MAGIC`, `FRAME_VERSION`, `FRAME_ALIGN`, `MAX_HEADER_BYTES`, `MAX_LEAVES`, `MAX_DEPTH`, *`MAX_NDIM`, *`FLAG_LZ4`, *`FRAME_KINDS`, *`DTYPES`, `DECOMPRESSED_SIZE_FACTOR`, `LZ4_*`; `FrameError(reason, message)`; `payload_byte_cap`; `encode_frame`; `decode_frame(buf, *, cap, copy=False, *kind=None)`; `peek_meta`; *`frame_sizes`; *`lookup_class`; `Lz4Probe` (+ *`enabled`); `ChunkMeta`; `encode_chunk`; `decode_chunk(buf, *, cap, *validator=None)`; `ChunkError(reason, *detail)`; *`chunk_meta`; *`check_payload_fields`; `ChunkValidator(role_spec, *, agent_id, teacher, *chunk_length=None, *state_template=None)` with *`check(payload, agent_id, policy_version=None)` and `validate(chunk)`; `WeightsMeta`, `encode_weights`, `decode_weights`, `encode_checkpoint`, `decode_checkpoint`, *`WEIGHTS_KINDS`, *`CHECKPOINT_KINDS`; *`colosseum.core.types.check_slot_arrays(kind, terminal, reset_after, has_teacher, *, agent_id, policy_version)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_sp5_frame.py`:
```python
"""The array frame (SP5 T1.1, spec block 4.9 and §6 "Формат кадра"): round trips of every payload node type, the byte
layout, zero-copy views, lz4, hostile input refused before allocation (ported from test_sp2_serialization and
extended), peek_meta, frame_sizes and the adaptive lz4 probe."""
from __future__ import annotations

import json
import struct
import sys
import types
from collections import namedtuple

import lz4.frame
import numpy as np
import pytest
import torch

from colosseum.net.frame import (
    DTYPES,
    FRAME_ALIGN,
    FRAME_MAGIC,
    MAX_DEPTH,
    MAX_HEADER_BYTES,
    MAX_LEAVES,
    FrameError,
    Lz4Probe,
    decode_frame,
    encode_frame,
    frame_sizes,
    payload_byte_cap,
    peek_meta,
)

HC = namedtuple("HC", ["h", "c"])
CAP = payload_byte_cap(64)


def _header(buf: bytes) -> tuple[dict, int]:
    (hlen,) = struct.unpack_from("<Q", buf, 8)
    return json.loads(buf[16:16 + hlen]), hlen


def _forge(header: dict, region: bytes = b"", flags: int = 0, version: int = 1) -> bytes:
    """A frame with a hand-written header (and data region)."""
    text = json.dumps(header).encode()
    head = struct.pack("<4sHHQ", FRAME_MAGIC, version, flags, len(text)) + text
    pad = (-len(head)) % FRAME_ALIGN
    return head + bytes(pad) + region


def _base(**changes) -> dict:
    header = {"v": 1, "kind": "tree", "meta": {}, "tree": {"t": "nd", "i": 0},
              "leaves": [{"dtype": "<f4", "shape": [4], "offset": 0, "nbytes": 16}], "data_bytes": 16, "lz4": None}
    header.update(changes)
    return header


@pytest.mark.parametrize("compress", [False, True])
def test_every_node_type_round_trips_with_its_dtype(compress):
    tree = {
        "a": np.arange(6, dtype=np.int64).reshape(2, 3),
        3: [np.float32(1.5), None, True, "s", 7, 2.5],
        "t": (np.zeros(2, dtype=bool), 7),
        "n": {"x": np.ones((1, 1, 4), np.float32), "u8": np.arange(5, dtype=np.uint8)},
        "hc": HC(np.full((1, 2), 3.0, np.float16), [np.zeros((0, 3), np.int32)]),
        1.5: np.array(4.0),                  # a 0-d leaf
        None: np.zeros(3, np.float64),
        False: np.array([-1], np.int8),
    }
    back, meta = decode_frame(encode_frame(tree, meta={"x": 1, "s": "y"}, compress=compress), cap=CAP)
    assert meta == {"x": 1, "s": "y"}
    assert list(back) == list(tree)
    assert np.array_equal(back["a"], tree["a"]) and back["a"].dtype == np.int64
    assert back[3] == [1.5, None, True, "s", 7, 2.5]
    assert isinstance(back["t"], tuple) and back["t"][1] == 7 and back["t"][0].dtype == np.bool_
    assert back["n"]["x"].shape == (1, 1, 4) and back["n"]["u8"].tolist() == [0, 1, 2, 3, 4]
    assert type(back["hc"]) is HC and back["hc"].h.dtype == np.float16 and back["hc"].c[0].shape == (0, 3)
    assert back[1.5].shape == () and float(back[1.5]) == 4.0
    assert back[None].dtype == np.float64 and back[False].tolist() == [-1]


def test_the_byte_layout_is_the_contracts():
    buf = encode_frame({"w": np.arange(3, dtype=np.float32), "b": np.ones(5, np.int8)}, kind="weights",
                       meta={"agent_id": "a"})
    magic, version, flags, hlen = struct.unpack_from("<4sHHQ", buf, 0)
    assert (magic, version, flags) == (b"CSTF", 1, 0)
    header, _ = _header(buf)
    assert set(header) == {"v", "kind", "meta", "tree", "leaves", "data_bytes", "lz4"}
    assert header["kind"] == "weights" and header["lz4"] is None
    assert [leaf["offset"] for leaf in header["leaves"]] == [0, 64]           # 64-aligned
    assert [leaf["dtype"] for leaf in header["leaves"]] == ["<f4", "|i1"]
    start = (16 + hlen + FRAME_ALIGN - 1) // FRAME_ALIGN * FRAME_ALIGN
    assert len(buf) == start + header["data_bytes"] == start + 64 + 5
    assert np.frombuffer(buf, np.float32, 3, start).tolist() == [0.0, 1.0, 2.0]


def test_arrays_are_views_of_the_buffer_unless_copied():
    buf = bytearray(encode_frame({"x": np.arange(16, dtype=np.float32)}))
    view, _ = decode_frame(buf, cap=CAP)
    assert view["x"].base is not None and view["x"].flags.writeable       # a bytearray: writable views
    buf[-4:] = np.float32(99.0).tobytes()
    assert view["x"][-1] == 99.0                                           # shares the buffer
    copied, _ = decode_frame(buf, cap=CAP, copy=True)
    buf[-4:] = np.float32(5.0).tobytes()
    assert copied["x"][-1] == 99.0
    read_only, _ = decode_frame(bytes(buf), cap=CAP)
    assert not read_only["x"].flags.writeable


def test_big_endian_arrays_are_written_little_endian():
    back, _ = decode_frame(encode_frame(np.arange(3, dtype=">i4")), cap=CAP)
    assert back.dtype == np.dtype("<i4") and back.tolist() == [0, 1, 2]


@pytest.mark.parametrize("value, message", [
    (torch.zeros(1), "Tensor"),
    (np.array([object()], dtype=object), "dtype"),
    (np.array(["a"]), "dtype"),
    (np.zeros(1, np.complex64), "dtype"),
    (b"bytes", "bytes"),
    ({(1, 2): 3}, "keys"),
    ({"x": float("nan")}, "non-finite"),
])
def test_encoding_refuses_what_cannot_be_framed(value, message):
    with pytest.raises(TypeError, match=message):
        encode_frame(value)


def test_a_local_namedtuple_cannot_be_framed():
    Local = namedtuple("Local", ["h"])
    with pytest.raises(TypeError, match=r"namedtuple .*Local.* module level"):
        encode_frame({"s": Local(np.zeros(1))})


def test_peek_meta_reads_only_the_header_and_frame_sizes_report_the_region():
    plain = encode_frame({"x": np.zeros(1000, np.float32)}, kind="chunk", meta={"agent_id": "a"})
    packed = encode_frame({"x": np.zeros(1000, np.float32)}, kind="chunk", meta={"agent_id": "a"}, compress=True)
    assert peek_meta(plain) == peek_meta(packed) == ("chunk", {"agent_id": "a"})
    _, hlen = _header(plain)
    assert peek_meta(plain[:16 + hlen]) == ("chunk", {"agent_id": "a"})    # the body is not read
    assert frame_sizes(plain) == (4000, 4000)
    raw, stored = frame_sizes(packed)
    assert raw == 4000 and stored < 200


# ---------------------------------------------------------------------------
# Hostile input (spec 4.9; SP2's serialization tests ported and extended)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("buf, reason", [
    (b"", "truncated"),
    (b"XXXX" + bytes(12), "magic"),
    (struct.pack("<4sHHQ", FRAME_MAGIC, 2, 0, 0), "version"),
    (struct.pack("<4sHHQ", FRAME_MAGIC, 1, 6, 0), "flags"),
    (struct.pack("<4sHHQ", FRAME_MAGIC, 1, 0, MAX_HEADER_BYTES + 1), "header_too_large"),
    (struct.pack("<4sHHQ", FRAME_MAGIC, 1, 0, 100) + b"{}", "truncated"),
    (struct.pack("<4sHHQ", FRAME_MAGIC, 1, 0, 2) + b"\xff\xfe", "header_json"),
    (struct.pack("<4sHHQ", FRAME_MAGIC, 1, 0, 3) + b"[1]", "header"),
])
def test_malformed_prefixes_and_headers_are_refused(buf, reason):
    with pytest.raises(FrameError) as info:
        decode_frame(buf, cap=CAP)
    assert info.value.reason == reason and isinstance(info.value, ValueError)


def test_a_huge_header_length_is_refused_before_reading_it():
    buf = struct.pack("<4sHHQ", FRAME_MAGIC, 1, 0, 2**63)
    with pytest.raises(FrameError) as info:
        decode_frame(buf, cap=CAP)
    assert info.value.reason == "header_too_large"


@pytest.mark.parametrize("changes, reason", [
    ({"data_bytes": 10**12, "leaves": [{"dtype": "<f4", "shape": [250 * 10**9], "offset": 0, "nbytes": 10**12}]},
     "too_large"),
    ({"leaves": [{"dtype": "<f4", "shape": [-4], "offset": 0, "nbytes": -16}]}, "shape"),
    ({"leaves": [{"dtype": "<f4", "shape": [True], "offset": 0, "nbytes": 4}]}, "shape"),
    ({"leaves": [{"dtype": "<f4", "shape": [4], "offset": 0, "nbytes": 12}]}, "nbytes"),
    ({"leaves": [{"dtype": "<f4", "shape": [4], "offset": 8, "nbytes": 16}]}, "offset"),
    ({"leaves": [{"dtype": "<f4", "shape": [4], "offset": 64, "nbytes": 16}]}, "offset"),
    ({"leaves": [{"dtype": "|O", "shape": [4], "offset": 0, "nbytes": 32}]}, "dtype"),
    ({"leaves": [{"dtype": ">f4", "shape": [4], "offset": 0, "nbytes": 16}]}, "dtype"),
    ({"leaves": [{"dtype": "<U1", "shape": [4], "offset": 0, "nbytes": 16}]}, "dtype"),
    ({"leaves": [{"dtype": "<f4", "shape": [4], "offset": 0, "nbytes": 16, "x": 1}]}, "header"),
    ({"leaves": [{"dtype": "<f4", "shape": [1] * 33, "offset": 0, "nbytes": 4}]}, "shape"),
    ({"tree": {"t": "nd", "i": 1}}, "leaf_ref"),
    ({"tree": {"t": "list", "v": [{"t": "nd", "i": 0}, {"t": "nd", "i": 0}]}}, "leaf_ref"),
    ({"tree": {"t": "py", "v": None}}, "leaf_ref"),                       # leaf 0 never referenced
    ({"tree": {"t": "py", "v": [1]}}, "node"),
    ({"tree": {"t": "what"}}, "node"),
    ({"tree": [1]}, "node"),
    ({"tree": {"t": "dict", "k": [[1]], "v": [{"t": "nd", "i": 0}]}}, "dict_key"),
    ({"tree": {"t": "dict", "k": ["a", "a"], "v": [{"t": "nd", "i": 0}, {"t": "py", "v": 1}]}}, "dict_key"),
    ({"tree": {"t": "dict", "k": ["a"], "v": []}}, "node"),
    ({"kind": "evil"}, "kind"),
    ({"meta": [1]}, "header"),
    ({"v": 2}, "version"),
    ({"lz4": {"raw_bytes": 15}}, "header"),
])
def test_malformed_descriptors_and_trees_are_refused(changes, reason):
    with pytest.raises(FrameError) as info:
        decode_frame(_forge(_base(**changes), bytes(16)), cap=CAP)
    assert info.value.reason == reason


def test_overlapping_leaves_are_refused():
    header = _base(tree={"t": "list", "v": [{"t": "nd", "i": 0}, {"t": "nd", "i": 1}]},
                   leaves=[{"dtype": "<f4", "shape": [32], "offset": 0, "nbytes": 128},
                           {"dtype": "<f4", "shape": [4], "offset": 64, "nbytes": 16}], data_bytes=128)
    with pytest.raises(FrameError) as info:
        decode_frame(_forge(header, bytes(128)), cap=CAP)
    assert info.value.reason == "overlap"


def test_a_negative_dimension_cannot_cancel_a_huge_one():
    header = _base(tree={"t": "list", "v": [{"t": "nd", "i": 0}, {"t": "nd", "i": 1}]},
                   leaves=[{"dtype": "<f4", "shape": [-(10**10)], "offset": 0, "nbytes": 0},
                           {"dtype": "<f4", "shape": [10**10], "offset": 0, "nbytes": 0}], data_bytes=0)
    with pytest.raises(FrameError) as info:
        decode_frame(_forge(header), cap=2**20)
    assert info.value.reason == "shape"


def test_declared_sizes_over_the_cap_are_refused_before_allocation():
    buf = encode_frame({"x": np.zeros(2**18, np.float32)})              # 1 MiB of data
    assert decode_frame(buf, cap=2**20)[0]["x"].shape == (2**18,)
    with pytest.raises(FrameError) as info:
        decode_frame(buf, cap=2**20 - 1)
    assert info.value.reason == "too_large"


def test_a_truncated_or_padded_data_region_is_refused():
    buf = encode_frame({"x": np.arange(100, dtype=np.int64)})
    for bad in (buf[:-1], buf + b"\0"):
        with pytest.raises(FrameError) as info:
            decode_frame(bad, cap=CAP)
        assert info.value.reason == "size"


def test_an_lz4_bomb_is_refused_at_the_cap():
    """4 MiB of zeros compress to a few KB: the declared raw size is checked against the cap first, and a frame that
    lies about it decompresses only up to the declared size."""
    buf = encode_frame({"x": np.zeros(2**20, np.float32)}, compress=True)
    assert len(buf) < 2**16
    with pytest.raises(FrameError) as info:
        decode_frame(buf, cap=2**20)
    assert info.value.reason == "too_large"
    header, _ = _header(buf)
    region = lz4.frame.compress(bytes(2**22))
    lying = _base(tree={"t": "nd", "i": 0}, leaves=[{"dtype": "<f4", "shape": [4], "offset": 0, "nbytes": 16}],
                  data_bytes=16, lz4={"raw_bytes": 16})
    with pytest.raises(FrameError) as info:
        decode_frame(_forge(lying, region, flags=1), cap=CAP)
    assert info.value.reason == "lz4"


@pytest.mark.parametrize("damage", ["truncate", "garbage", "flag_without_header"])
def test_corrupt_lz4_regions_are_refused(damage):
    buf = encode_frame({"x": np.arange(1000)}, compress=True)
    if damage == "truncate":
        bad = buf[: len(buf) - 10]
    elif damage == "garbage":
        bad = buf[:-20] + bytes(20)
    else:
        bad = bytearray(encode_frame({"x": np.arange(1000)}))
        bad[6] = 1                                                          # flag set, header says no lz4
        bad = bytes(bad)
    with pytest.raises(FrameError) as info:
        decode_frame(bad, cap=CAP)
    assert info.value.reason in ("lz4", "flags")


def test_non_finite_numbers_in_the_header_are_refused():
    text = json.dumps(_base(meta={"x": 1})).replace('"x": 1', '"x": NaN').encode()
    head = struct.pack("<4sHHQ", FRAME_MAGIC, 1, 0, len(text)) + text
    with pytest.raises(FrameError) as info:
        decode_frame(head + bytes((-len(head)) % FRAME_ALIGN) + bytes(16), cap=CAP)
    assert info.value.reason == "header_json"


def test_too_many_leaves_and_too_deep_trees_are_refused(monkeypatch):
    import colosseum.net.frame as frame

    assert MAX_LEAVES == 100_000
    monkeypatch.setattr(frame, "MAX_LEAVES", 10)                            # 100 001 leaves exceed the header cap
    leaves = [{"dtype": "<f4", "shape": [0], "offset": 0, "nbytes": 0}] * 11
    with pytest.raises(FrameError) as info:
        decode_frame(_forge(_base(leaves=leaves, data_bytes=0)), cap=CAP)
    assert info.value.reason == "too_many_leaves"
    node: dict = {"t": "nd", "i": 0}
    for _ in range(MAX_DEPTH + 1):
        node = {"t": "list", "v": [node]}
    with pytest.raises(FrameError) as info:
        decode_frame(_forge(_base(tree=node), bytes(16)), cap=CAP)
    assert info.value.reason == "depth"
    deep = "[" * 100_000 + "]" * 100_000                                    # json's own recursion limit
    head = struct.pack("<4sHHQ", FRAME_MAGIC, 1, 0, len(deep)) + deep.encode()
    with pytest.raises(FrameError) as info:
        decode_frame(head, cap=CAP)
    assert info.value.reason == "header_json"


def test_unknown_namedtuples_are_refused_and_never_imported(monkeypatch):
    calls = []
    lazy = types.ModuleType("lazy_mod_sp5")
    lazy.__getattr__ = lambda name: calls.append(name) or HC              # PEP 562 lazy import hook
    monkeypatch.setitem(sys.modules, "lazy_mod_sp5", lazy)
    for cls, fields, reason in [("no_such_module_sp5:HC", ["h", "c"], "namedtuple"),
                                ("lazy_mod_sp5:HC", ["h", "c"], "namedtuple"),
                                (f"{__name__}:HC", ["h", "x"], "namedtuple"),
                                ("builtins:int", [], "namedtuple")]:
        tree = {"t": "nt", "cls": cls, "f": fields, "v": [{"t": "nd", "i": 0}, {"t": "py", "v": 1}][:len(fields)]}
        header = _base(tree=tree) if fields else _base(tree=tree, leaves=[], data_bytes=0)
        with pytest.raises(FrameError) as info:
            decode_frame(_forge(header, bytes(16) if fields else b""), cap=CAP)
        assert info.value.reason == reason
    assert calls == [] and "no_such_module_sp5" not in sys.modules


def test_the_dtype_whitelist_is_numeric_and_bool_only():
    assert DTYPES == {"|b1", "|i1", "|u1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8", "<f2", "<f4", "<f8"}


# ---------------------------------------------------------------------------
# Adaptive lz4 (spec 4.9)
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_the_probe_keeps_lz4_on_for_compressible_frames_and_reprobes_on_its_clock():
    clock = _Clock()
    probe = Lz4Probe(min_ratio=1.3, probe_frames=4, reprobe_sec=600.0, clock=clock)
    for _ in range(4):
        assert probe.compress()
        probe.observe(1000, 500)                    # ratio 2.0
    assert probe.enabled and probe.compress()
    probe.observe(1000, 900)                        # outside a probe: ignored
    clock.t = 599.0
    assert probe.compress() and probe.enabled
    clock.t = 600.0
    for _ in range(4):                              # a new probe: the stream became incompressible
        assert probe.compress()
        probe.observe(1000, 990)
    assert not probe.enabled and not probe.compress()
    probe.observe(1000, None)                       # sent plain: nothing to learn
    assert not probe.compress()


def test_the_probe_switches_lz4_off_below_the_ratio():
    probe = Lz4Probe(min_ratio=1.3, probe_frames=2, clock=_Clock())
    probe.observe(1000, 800)                        # 1.25
    probe.observe(1000, 800)
    assert not probe.enabled and not probe.compress()
    with pytest.raises(ValueError):
        Lz4Probe(probe_frames=0)
```

Create `tests/unit/test_sp5_chunks.py`:
```python
"""Chunk, weights and checkpoint frames and the learner's chunk checks (SP5 T1.1, spec block 4.9): chunk v2 round
trips with every optional field, the same content as the SP1 npz format, decode_chunk's field checks, every
ChunkValidator rule, buffer-built chunks of several role kinds passing, weights and checkpoint frames."""
from __future__ import annotations

import io
from collections import namedtuple

import gymnasium
import numpy as np
import pytest
import torch

from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.types import TrajectoryChunk
from colosseum.envs.game import RoleSpec
from colosseum.envs.spaces import Units
from colosseum.net.chunks import (
    ChunkError,
    ChunkMeta,
    ChunkValidator,
    WeightsMeta,
    decode_checkpoint,
    decode_chunk,
    decode_weights,
    encode_checkpoint,
    encode_chunk,
    encode_weights,
)
from colosseum.net.frame import FrameError, encode_frame, payload_byte_cap
from colosseum.networks.state import state_to_numpy
from colosseum.worker.buffers import BufferSpec, RolloutBuffer
from game_helpers import chunk_v2_payload, learner_role

CAP = payload_byte_cap(64)
LSTM = namedtuple("LSTM", ["h", "c"])
ROLE = learner_role(3)                                    # Box(4) obs, Discrete(3) actions (masks: bool[3])


def _chunk(**changes) -> TrajectoryChunk:
    payload = chunk_v2_payload(pattern="AATP")
    payload.update(changes)
    return TrajectoryChunk.from_payload(payload)


def _validator(**kwargs) -> ChunkValidator:
    return ChunkValidator(ROLE, agent_id="a", teacher=False, **kwargs)


def _round_trip(chunk: TrajectoryChunk, **kwargs) -> tuple[TrajectoryChunk, ChunkMeta]:
    return decode_chunk(encode_chunk(chunk, generation=2, node=5, worker=1, **kwargs), cap=CAP)


def _equal(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, tuple | list):
        return type(a) is type(b) and len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _chunks_equal(a: TrajectoryChunk, b: TrajectoryChunk) -> bool:
    fields = ("agent_id", "policy_version", "initial_state", "obs", "global_state", "actions", "action_masks", "kind",
              "reward", "terminal", "reset_after", "behavior_logp", "behavior_unit_logp", "teacher_action",
              "has_teacher")
    return all(_equal(getattr(a, f), getattr(b, f)) for f in fields)


@pytest.mark.parametrize("compress", [False, True])
def test_a_chunk_round_trips_with_every_optional_field(compress):
    payload = chunk_v2_payload(version=7, pattern="AATP")
    payload["obs"] = {"grid": np.arange(4 * 6, dtype=np.uint8).reshape(4, 2, 3), "vec": np.ones((4, 3), np.float32)}
    payload["global_state"] = np.zeros((4, 5), np.float32)
    payload["behavior_unit_logp"] = np.full((4, 2), -0.5, np.float32)
    payload["teacher_action"] = np.array([1, 2, 0, 0], np.int64)
    payload["has_teacher"] = np.array([True, True, False, False])
    state = LSTM(np.ones((1, 1, 8), np.float32), np.zeros((1, 1, 8), np.float32))
    payload["initial_state"] = {"core": state, "extra": [np.ones((1, 2), np.float32), (np.zeros(1, np.int64),)]}
    chunk = TrajectoryChunk.from_payload(payload)
    back, meta = _round_trip(chunk, compress=compress)
    assert meta == ChunkMeta(agent_id="a", policy_version=7, generation=2, node=5, worker=1)
    assert _chunks_equal(back, chunk)
    assert type(back.initial_state["core"]) is LSTM and torch.equal(back.initial_state["core"].h,
                                                                     torch.from_numpy(state.h))
    assert back.obs["grid"].dtype == torch.uint8 and back.kind.dtype == torch.int8


def test_frame_and_npz_formats_decode_to_the_same_content():
    """The frame replaces the SP1 npz format (removed in T7.2): the same payload decodes to the same chunk."""
    from colosseum.transport.serialization import deserialize_chunk_payload, serialize_chunk_payload

    payload = chunk_v2_payload(version=3, pattern="ATPP")
    payload["obs"] = {"a": np.ones((4, 2), np.float32), "b": {"c": np.arange(4, dtype=np.int16)}}
    payload["initial_state"] = state_to_numpy((torch.ones(1, 3), [torch.zeros(1, 2, dtype=torch.int64)]))
    chunk = TrajectoryChunk.from_payload(payload)
    via_npz = TrajectoryChunk.from_payload(deserialize_chunk_payload(*serialize_chunk_payload(chunk.to_payload())))
    via_frame, _ = _round_trip(chunk)
    assert _chunks_equal(via_frame, via_npz)


def test_the_meta_names_the_agent_and_never_needs_decoding_the_body():
    from colosseum.net.frame import peek_meta

    buf = encode_chunk(_chunk(), generation=0, node=-1, worker=3)
    assert peek_meta(buf) == ("chunk", {"agent_id": "a", "policy_version": 0, "generation": 0, "node": -1,
                                        "worker": 3})


@pytest.mark.parametrize("meta, detail", [
    ({"agent_id": "a b", "policy_version": 0, "generation": 0, "node": 0, "worker": 0}, "agent_id"),
    ({"agent_id": "a", "policy_version": -1, "generation": 0, "node": 0, "worker": 0}, "policy_version"),
    ({"agent_id": "a", "policy_version": True, "generation": 0, "node": 0, "worker": 0}, "policy_version"),
    ({"agent_id": "a", "policy_version": 0, "generation": 0, "node": -2, "worker": 0}, "node"),
    ({"agent_id": "a", "policy_version": 0, "generation": 0, "node": 0}, "meta keys"),
])
def test_decode_chunk_refuses_malformed_meta(meta, detail):
    payload = chunk_v2_payload()
    payload.pop("agent_id"), payload.pop("policy_version")
    payload.update(teacher_action=None, has_teacher=None)
    with pytest.raises(ChunkError) as info:
        decode_chunk(encode_frame(payload, kind="chunk", meta=meta), cap=CAP)
    assert info.value.reason == "invalid" and detail in info.value.detail


def _framed(**changes) -> bytes:
    payload = chunk_v2_payload()
    payload.pop("agent_id"), payload.pop("policy_version")
    payload.update(teacher_action=None, has_teacher=None)
    payload.update(changes)
    return encode_frame(payload, kind="chunk", meta={"agent_id": "a", "policy_version": 0, "generation": 0,
                                                     "node": 0, "worker": 0})


@pytest.mark.parametrize("changes, detail", [
    ({"kind": None}, "'kind'"),
    ({"reward": np.zeros(3, np.float32)}, "'reward'"),
    ({"obs": {"grid": np.zeros((3, 2), np.uint8)}}, "obs/grid"),
    ({"obs": [np.zeros(4)]}, "'obs'"),
    ({"obs": {1: np.zeros(4)}}, "'obs'"),
    ({"behavior_unit_logp": np.zeros(3, np.float32)}, "behavior_unit_logp"),
    ({"has_teacher": np.zeros(4, bool)}, "come together"),
    ({"initial_state": "x"}, "initial_state"),
    ({"kind": np.zeros(1, np.int8), "reward": np.zeros(1, np.float32), "terminal": np.zeros(1, bool),
      "reset_after": np.zeros(1, bool), "behavior_logp": np.zeros(1, np.float32), "obs": np.zeros((1, 4)),
      "actions": np.zeros(1, np.int64), "action_masks": np.ones((1, 3), bool)}, "at least 2 slots"),
])
def test_decode_chunk_refuses_malformed_fields(changes, detail):
    with pytest.raises(ChunkError) as info:
        decode_chunk(_framed(**changes), cap=CAP)
    assert info.value.reason == "invalid" and detail in info.value.detail


def test_decode_chunk_refuses_other_frames_and_missing_fields():
    with pytest.raises(ChunkError, match="frame:kind"):
        decode_chunk(encode_weights({"w": np.zeros(1, np.float32)}, WeightsMeta("a", "latest", 1, 0)), cap=CAP)
    with pytest.raises(ChunkError, match="frame:magic"):
        decode_chunk(b"nonsense-bytes-nonsense", cap=CAP)
    payload = chunk_v2_payload()
    payload.pop("agent_id"), payload.pop("policy_version")
    with pytest.raises(ChunkError, match="payload fields"):                 # no teacher fields at all
        decode_chunk(encode_frame(payload, kind="chunk", meta={"agent_id": "a", "policy_version": 0,
                                                               "generation": 0, "node": 0, "worker": 0}), cap=CAP)


# ---------------------------------------------------------------------------
# ChunkValidator (spec 4.9 "Проверка чанка на лёрнере")
# ---------------------------------------------------------------------------


def test_a_good_chunk_passes():
    _validator(chunk_length=4).validate(_chunk())


@pytest.mark.parametrize("chunk, detail", [
    (_chunk(agent_id="b"), "a chunk of agent 'b'"),
    (_chunk(obs=np.zeros((4, 5), np.float32)), "obs/<root>"),
    (_chunk(obs=np.zeros((4, 4), np.float64)), "obs/<root>"),
    (_chunk(obs={"x": np.zeros((4, 4), np.float32)}), "obs: expected an array"),
    (_chunk(global_state=np.zeros((4, 2), np.float32)), "global_state present"),
    (_chunk(action_masks=np.ones((4, 3), np.int8)), "action_masks/<root>"),
    (_chunk(action_masks=None), "action_masks missing"),
    (_chunk(actions=np.array([0, 3, 0, 0], np.int64)), "outside the action space"),
    (_chunk(actions=np.array([0, -1, 0, 0], np.int64)), "outside the action space"),
    (_chunk(actions=np.zeros(4, np.int32)), "actions/<root>"),
    (_chunk(behavior_logp=np.array([np.nan, 0, 0, 0], np.float32)), "non-finite behavior_logp"),
    (_chunk(behavior_logp=np.array([0, -np.inf, 0, 0], np.float32)), "non-finite behavior_logp"),
    (_chunk(reward=np.array([0, 0, 0, 1], np.float32)), "non-zero reward on a BOOT or PAD"),
    (_chunk(reward=np.array([np.inf, 0, 0, 0], np.float32)), "non-finite reward"),
    (_chunk(reward=np.zeros(4, np.float64)), "reward"),
    (_chunk(behavior_unit_logp=np.zeros((4, 2), np.float32)), "behavior_unit_logp present"),
    (_chunk(teacher_action=np.zeros(4, np.int64), has_teacher=np.zeros(4, bool)), "DAgger labels on a chunk"),
    (_chunk(kind=np.array([0, 0, 0, 0], np.int8), terminal=np.zeros(4, bool), reset_after=np.zeros(4, bool)),
     "an ACT in the last slot"),
])
def test_the_validator_refuses_each_broken_rule(chunk, detail):
    with pytest.raises(ChunkError) as info:
        _validator().validate(chunk)
    assert info.value.reason == "invalid" and detail in str(info.value)


def test_a_non_finite_logp_off_act_slots_and_the_chunk_length():
    _validator().validate(_chunk(behavior_logp=np.array([0, 0, 0, np.nan], np.float32)))   # the PAD slot
    with pytest.raises(ChunkError, match="expected rollout.chunk_length 8"):
        _validator(chunk_length=8).validate(_chunk())


def test_the_initial_state_must_match_the_models_template():
    template = state_to_numpy(LSTM(torch.zeros(1, 1, 8), torch.zeros(1, 1, 8)))
    good = _chunk(initial_state=state_to_numpy(LSTM(torch.ones(1, 1, 8), torch.ones(1, 1, 8))))
    _validator(state_template=template).validate(good)
    for bad in (None, (np.zeros((1, 1, 8), np.float32), np.zeros((1, 1, 8), np.float32)),
                LSTM(np.zeros((1, 1, 4), np.float32), np.zeros((1, 1, 8), np.float32)),
                LSTM(np.zeros((1, 1, 8), np.float64), np.zeros((1, 1, 8), np.float32))):
        with pytest.raises(ChunkError, match="initial_state"):
            _validator(state_template=template).validate(_chunk(initial_state=bad))


DICT_ACT = gymnasium.spaces.Dict({"move": gymnasium.spaces.Discrete(4),
                                  "aim": gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32),
                                  "pick": gymnasium.spaces.MultiDiscrete([2, 3])})
UNITS_ACT = Units(3, gymnasium.spaces.Dict({"accel": gymnasium.spaces.Box(-1.0, 1.0, (2,), np.float32),
                                            "push": gymnasium.spaces.Discrete(2)}))
OBS = gymnasium.spaces.Dict({"ships": gymnasium.spaces.Box(-1, 1, (3, 9), np.float32),
                             "mask": gymnasium.spaces.MultiBinary(5), "turn": gymnasium.spaces.Discrete(7)})


def _buffer_chunk(role: RoleSpec, *, teacher: bool = False, slots: int = 4) -> TrajectoryChunk:
    """A chunk the worker's own buffer builds: ACT, ACT, terminal ACT, PAD."""
    spec = BufferSpec(ObsSpec.from_space(role.observation_space), ActionSpec.from_space(role.action_space),
                      None if role.global_state_space is None else ObsSpec.from_space(role.global_state_space),
                      teacher=teacher)
    buf = RolloutBuffer(slots, spec)
    buf.begin(None, 1)
    obs = spec.obs.allocate(())
    action = spec.action.allocate_actions(())
    mask = spec.action.full_mask(())
    gs = None if spec.global_state is None else spec.global_state.allocate(())
    unit_logp = np.full(spec.action.num_deciders, -0.25, np.float32) if spec.action.num_deciders > 1 else None
    for _ in range(slots - 1):
        buf.write_act(obs, gs, mask, action, -0.5, unit_logp, 0.0, teacher_action=action if teacher else None)
    buf.mark_terminal()
    buf.write_pad()
    return buf.build_chunk("a")


@pytest.mark.parametrize("role, teacher", [
    (RoleSpec(OBS, DICT_ACT), True),
    (RoleSpec(OBS, UNITS_ACT, gymnasium.spaces.Box(-1, 1, (6,), np.float32)), False),
    (RoleSpec(gymnasium.spaces.Discrete(5), gymnasium.spaces.Box(-1, 1, (3,), np.float32)), False),
])
def test_chunks_the_worker_buffers_build_pass_after_a_round_trip(role, teacher):
    chunk = _buffer_chunk(role, teacher=teacher)
    back, _ = _round_trip(chunk)
    ChunkValidator(role, agent_id="a", teacher=teacher, chunk_length=4).validate(back)
    with pytest.raises(ChunkError):
        ChunkValidator(role, agent_id="a", teacher=not teacher).validate(back)


def test_units_and_dict_actions_are_range_checked_per_component():
    units_role = RoleSpec(OBS, UNITS_ACT)
    chunk = _buffer_chunk(units_role)
    chunk.actions["push"][0, 1] = 2                                          # Discrete(2) component
    with pytest.raises(ChunkError, match="component 'push'"):
        ChunkValidator(units_role, agent_id="a", teacher=False).validate(chunk)
    chunk = _buffer_chunk(units_role)
    chunk.actions["accel"][1, 0, 0] = float("nan")
    with pytest.raises(ChunkError, match="component 'accel'"):
        ChunkValidator(units_role, agent_id="a", teacher=False).validate(chunk)
    dict_role = RoleSpec(OBS, DICT_ACT)
    chunk = _buffer_chunk(dict_role, teacher=True)
    chunk.teacher_action["pick"][0, 1] = 3                                   # MultiDiscrete([2, 3])
    with pytest.raises(ChunkError, match="teacher_action/pick"):
        ChunkValidator(dict_role, agent_id="a", teacher=True).validate(chunk)
    chunk = _buffer_chunk(dict_role, teacher=True)
    chunk.has_teacher = chunk.has_teacher.to(torch.int8)
    with pytest.raises(ChunkError, match="has_teacher"):
        ChunkValidator(dict_role, agent_id="a", teacher=True).validate(chunk)


# ---------------------------------------------------------------------------
# Weights and checkpoints
# ---------------------------------------------------------------------------


def test_weights_round_trip_with_their_meta_and_are_never_compressed():
    sd = {"w": np.random.default_rng(0).standard_normal((3, 3)).astype(np.float32), "n": np.arange(2, dtype=np.int64)}
    meta = WeightsMeta("main", "fixed", 0, 40, teacher_active=True, checkpoint_id="ckpt_v40",
                       architecture={"encoder_class": "x.Y", "kwargs": {"h": 8}}, roles=("player",))
    buf = encode_weights(sd, meta)
    from colosseum.net.frame import frame_sizes

    assert frame_sizes(buf)[0] == frame_sizes(buf)[1]                         # plain
    back, back_meta = decode_weights(buf, cap=CAP)
    assert back_meta == meta and set(back) == {"w", "n"} and np.array_equal(back["w"], sd["w"])
    with pytest.raises(FrameError, match="kind"):
        decode_weights(encode_frame({"w": np.zeros(1)}), cap=CAP)
    with pytest.raises(FrameError, match="weights_meta"):
        decode_weights(encode_frame({"w": np.zeros(1)}, kind="weights", meta={"agent_id": "a"}), cap=CAP)
    with pytest.raises(FrameError, match="weights"):
        decode_weights(encode_frame([np.zeros(1)], kind="weights", meta={}), cap=CAP)
    with pytest.raises(ValueError, match="kind"):
        encode_weights(sd, WeightsMeta("a", "nonsense", 0, 0))                 # type: ignore[arg-type]


@pytest.mark.parametrize("with_trainer", [True, False])
def test_checkpoints_round_trip_with_the_trainer_state_as_bytes(with_trainer):
    buf = io.BytesIO()
    torch.save({"step": torch.tensor(3), "opt": {"lr": 0.1}}, buf)
    payload = {"agent_id": "main", "policy_version": 120, "model_state": {"w": np.ones((2, 2), np.float32)},
               "trainer_state_bytes": buf.getvalue() if with_trainer else None, "final": True}
    back, meta = decode_checkpoint(encode_checkpoint(payload, kind="final", generation=3), cap=CAP)
    assert meta == {"agent_id": "main", "policy_version": 120, "final": True, "kind": "final", "generation": 3}
    assert back["trainer_state_bytes"] == payload["trainer_state_bytes"] and back["final"] is True
    assert np.array_equal(back["model_state"]["w"], payload["model_state"]["w"])
    if with_trainer:
        state = torch.load(io.BytesIO(back["trainer_state_bytes"]), weights_only=True)
        assert int(state["step"]) == 3
    with pytest.raises(FrameError, match="checkpoint"):
        decode_checkpoint(encode_frame({"model_state": {}}, kind="checkpoint", meta={}), cap=CAP)


def test_validate_refuses_a_model_state_that_cannot_be_framed():
    """core.validation checks a model's state on the frame codec (T1.1): nested containers and importable
    namedtuples pass, a namedtuple defined in a function does not."""
    from colosseum.core.errors import ConfigError
    from colosseum.core.validation import _check_state_payload

    _check_state_payload({"core": LSTM(torch.zeros(2, 1, 4), torch.ones(2, 1, 4)), "n": [torch.zeros(2, 3)]}, "m")
    Local = namedtuple("Local", ["h"])
    with pytest.raises(ConfigError, match="initial_state cannot be sent between processes.*module level"):
        _check_state_payload(Local(torch.zeros(2, 4)), "m")
```

- [ ] **Step 2: Run them (RED)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_frame.py tests/unit/test_sp5_chunks.py -q`
Expected: both modules fail at collection with `ModuleNotFoundError: No module named 'colosseum.net'`.

- [ ] **Step 3: Create the package and the frame codec**

Create `src/colosseum/net/__init__.py`:
```python
"""The SP5 wire layer (no processes): array frames (``frame``), chunk / weights / checkpoint frames and the learner's
chunk checks (``chunks``), protobuf v1 and its converters (``proto``, ``messages``), identities, tokens, certificates
and node authentication (``security``), gRPC helpers (``grpc_util``) and same-machine connections (``local``).

Nothing in this package pickles data that crosses a machine, and nothing here starts a process or a thread.
"""
```

Create `src/colosseum/net/frame.py`:
```python
"""The array frame (SP5 spec block 4.9): trees of numpy arrays + JSON metadata in one buffer, decoded without pickle.

Every chunk, weights set, snapshot, frozen / teacher / init weights, checkpoint and recovery point crosses a process
or a machine boundary as one frame::

    offset 0   4 B  magic b"CSTF"
           4   2 B  uint16 LE format version (1)
           6   2 B  uint16 LE flags (bit 0: the data region is one lz4 frame)
           8   8 B  uint64 LE header length H (<= MAX_HEADER_BYTES)
          16   H B  UTF-8 JSON header
      align64(16 + H)  data region: raw little-endian buffers, each at a 64-aligned offset (or one lz4 frame of it)

Header: ``{"v": 1, "kind", "meta", "tree", "leaves", "data_bytes", "lz4"}``. ``tree`` mirrors the payload:
``{"t": "nd", "i": k}`` (leaf k), ``{"t": "dict", "k": [keys], "v": [nodes]}`` (keys str / int / float / bool /
null), ``{"t": "list" | "tuple", "v": [nodes]}``, ``{"t": "nt", "cls": "module:Qualname", "f": [fields], "v":
[nodes]}`` (a namedtuple) and ``{"t": "py", "v": scalar}`` (None / bool / int / float / str). ``leaves[k]`` =
``{"dtype", "shape", "offset", "nbytes"}`` with offsets relative to the data region; ``data_bytes`` is the raw size of
the data region and ``lz4`` is null or ``{"raw_bytes": data_bytes}``.

Decoding treats the bytes as hostile and raises :class:`FrameError` (a ``ValueError`` with a short ``reason``)
before it allocates anything large: the header length, the declared sizes against the caller's ``cap``, each leaf's
``nbytes == prod(shape) * itemsize``, 64-aligned non-overlapping offsets inside the data region, a dtype whitelist
(bool and the little-endian integer / float types; never ``object``), the leaf count and the nesting depth, every
leaf referenced exactly once, no NaN / Infinity in the header, and a namedtuple only as a class that is already
loaded with the same fields (looked up in ``sys.modules`` through ``__dict__``; never imported). lz4 decompression is
bounded by ``data_bytes``. Arrays are zero-copy ``np.frombuffer`` views of the buffer (read-only for ``bytes``)
unless ``copy=True`` or the view would be misaligned.

lz4 is a flag of the frame; :class:`Lz4Probe` decides it per (sender, agent): compress the first probe frames,
keep compressing when the ratio is at least ``LZ4_MIN_RATIO``, probe again every ``LZ4_REPROBE_SEC``.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

import lz4.frame
import numpy as np

FRAME_MAGIC = b"CSTF"
FRAME_VERSION = 1
FRAME_ALIGN = 64
FLAG_LZ4 = 1
MAX_HEADER_BYTES = 1 << 20
MAX_LEAVES = 100_000
MAX_DEPTH = 64
MAX_NDIM = 32
DECOMPRESSED_SIZE_FACTOR = 8             # payload_byte_cap(mb) = 8 x mb MiB (SP1 rule)
LZ4_MIN_RATIO = 1.3
LZ4_PROBE_FRAMES = 4
LZ4_REPROBE_SEC = 600.0
FRAME_KINDS = ("tree", "chunk", "weights", "checkpoint")

_PREFIX = struct.Struct("<4sHHQ")         # magic, version, flags, header length: 16 bytes
_HEADER_KEYS = {"v", "kind", "meta", "tree", "leaves", "data_bytes", "lz4"}
_LEAF_KEYS = {"dtype", "shape", "offset", "nbytes"}
_KEY_TYPES = (str, int, float, bool, type(None))
# Little-endian (or byte-order-free) numeric dtypes and bool; never object, str, bytes, void, datetime or complex.
DTYPES = frozenset({"|b1", "|i1", "|u1", "<i2", "<u2", "<i4", "<u4", "<i8", "<u8", "<f2", "<f4", "<f8"})
_DTYPE_OBJECTS = {name: np.dtype(name) for name in DTYPES}


class FrameError(ValueError):
    """A malformed or oversized frame. ``reason`` is a short machine-readable tag (counters, tests)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


def payload_byte_cap(max_message_mb: int) -> int:
    """Largest decoded payload (bytes) accepted for a message limit of ``max_message_mb`` MiB."""
    return int(max_message_mb) * 2**20 * DECOMPRESSED_SIZE_FACTOR


def _align(n: int) -> int:
    return (n + FRAME_ALIGN - 1) // FRAME_ALIGN * FRAME_ALIGN


def lookup_class(module: str, qualname: str) -> Any:
    """Class ``module.qualname`` from ``sys.modules``, or None. Walks ``__dict__`` only: never imports and never
    triggers a module ``__getattr__`` (PEP 562 lazy imports) or a descriptor."""
    obj: Any = sys.modules.get(module)
    if obj is None or "<locals>" in qualname:
        return None
    for part in qualname.split("."):
        namespace = getattr(obj, "__dict__", None)
        if not isinstance(namespace, Mapping):
            return None
        obj = namespace.get(part)
        if obj is None:
            return None
    return obj


def _is_namedtuple(x: Any) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_fields")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _encode_node(obj: Any, leaves: list[np.ndarray], depth: int) -> dict:
    if depth > MAX_DEPTH:
        raise TypeError(f"payload nested deeper than {MAX_DEPTH} levels")
    if isinstance(obj, np.ndarray):
        dtype = obj.dtype
        if dtype.byteorder == ">":
            obj = obj.astype(dtype.newbyteorder("<"))
            dtype = obj.dtype
        if dtype.str not in DTYPES:
            raise TypeError(f"arrays of dtype {dtype} cannot be framed (bool and numeric little-endian only)")
        if obj.ndim > MAX_NDIM:
            raise TypeError(f"arrays with more than {MAX_NDIM} dimensions cannot be framed")
        leaves.append(obj)
        return {"t": "nd", "i": len(leaves) - 1}
    if isinstance(obj, dict):
        bad = [k for k in obj if not isinstance(k, _KEY_TYPES)]
        if bad:
            raise TypeError(f"payload dict keys must be str/int/float/bool/None, got {type(bad[0]).__name__}")
        return {"t": "dict", "k": list(obj), "v": [_encode_node(v, leaves, depth + 1) for v in obj.values()]}
    if _is_namedtuple(obj):
        cls = type(obj)
        if lookup_class(cls.__module__, cls.__qualname__) is not cls:
            raise TypeError(f"cannot frame namedtuple {cls.__module__}.{cls.__qualname__}: the receiver rebuilds it "
                            f"by module path, so define it at module level of an importable module")
        return {"t": "nt", "cls": f"{cls.__module__}:{cls.__qualname__}", "f": list(cls._fields),
                "v": [_encode_node(v, leaves, depth + 1) for v in obj]}
    if isinstance(obj, tuple):
        return {"t": "tuple", "v": [_encode_node(v, leaves, depth + 1) for v in obj]}
    if isinstance(obj, list):
        return {"t": "list", "v": [_encode_node(v, leaves, depth + 1) for v in obj]}
    if isinstance(obj, np.generic):
        return {"t": "py", "v": obj.item()}
    if obj is None or isinstance(obj, bool | int | float | str):
        return {"t": "py", "v": obj}
    raise TypeError(f"cannot frame {type(obj).__name__} (numpy arrays, primitives, dict / list / tuple / namedtuple "
                    f"only; convert tensors with to_payload() / state_dict_to_numpy())")


def encode_frame(tree: Any, *, kind: str = "tree", meta: Mapping[str, Any] | None = None,
                 compress: bool = False) -> bytes:
    """One frame of ``tree`` (numpy arrays + primitives in dict / list / tuple / namedtuple) with JSON ``meta``.

    Raises TypeError for anything that cannot be framed (a tensor, an object array, a local namedtuple, bytes,
    NaN in ``meta`` or in a scalar)."""
    if kind not in FRAME_KINDS:
        raise ValueError(f"frame kind must be one of {FRAME_KINDS}, got {kind!r}")
    leaves: list[np.ndarray] = []
    node = _encode_node(tree, leaves, 0)
    descs = []
    parts: list[Any] = []                         # the data region: padding and raw leaf bytes
    offset = 0
    for leaf in leaves:
        arr = leaf if leaf.flags.c_contiguous else np.array(leaf, order="C")   # keeps 0-d leaves 0-d
        start = _align(offset)
        if start > offset:
            parts.append(bytes(start - offset))
        descs.append({"dtype": arr.dtype.str, "shape": list(arr.shape), "offset": start, "nbytes": int(arr.nbytes)})
        if arr.nbytes:
            parts.append(arr.reshape(-1).view(np.uint8).data)
        offset = start + int(arr.nbytes)
    data_bytes = offset
    header = {"v": FRAME_VERSION, "kind": kind, "meta": dict(meta or {}), "tree": node, "leaves": descs,
              "data_bytes": data_bytes, "lz4": {"raw_bytes": data_bytes} if compress else None}
    try:
        text = json.dumps(header, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except ValueError as e:                       # NaN / Infinity
        raise TypeError(f"frame header holds a non-finite number: {e}") from None
    if len(text) > MAX_HEADER_BYTES:
        raise TypeError(f"frame header of {len(text)} bytes exceeds {MAX_HEADER_BYTES} bytes (too many leaves?)")
    if compress:
        parts = [lz4.frame.compress(b"".join(parts))]
    head = _PREFIX.pack(FRAME_MAGIC, FRAME_VERSION, FLAG_LZ4 if compress else 0, len(text)) + text
    return b"".join([head, bytes(_align(len(head)) - len(head)), *parts])


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _reject_constant(name: str) -> Any:
    raise FrameError("header_json", f"non-finite number {name} in the header")


def _read_header(mv: memoryview) -> tuple[dict, int, int]:
    """``(header, flags, data start)`` of a frame; checks the prefix and the header's top-level shape."""
    if len(mv) < _PREFIX.size:
        raise FrameError("truncated", f"{len(mv)} bytes, shorter than the {_PREFIX.size}-byte prefix")
    magic, version, flags, hlen = _PREFIX.unpack_from(mv, 0)
    if magic != FRAME_MAGIC:
        raise FrameError("magic", f"not a Colosseum frame (magic {bytes(magic)!r})")
    if version != FRAME_VERSION:
        raise FrameError("version", f"frame format version {version}; this code reads version {FRAME_VERSION}")
    if flags & ~FLAG_LZ4:
        raise FrameError("flags", f"unknown flags {flags:#06x}")
    if hlen > MAX_HEADER_BYTES:
        raise FrameError("header_too_large", f"header of {hlen} bytes exceeds {MAX_HEADER_BYTES} bytes")
    if _PREFIX.size + hlen > len(mv):
        raise FrameError("truncated", f"header of {hlen} bytes runs past the frame's {len(mv)} bytes")
    try:
        header = json.loads(str(mv[_PREFIX.size:_PREFIX.size + hlen], "utf-8"), parse_constant=_reject_constant)
    except FrameError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError) as e:
        raise FrameError("header_json", f"{type(e).__name__}: {e}") from None
    if not isinstance(header, dict) or set(header) != _HEADER_KEYS:
        keys = sorted(header) if isinstance(header, dict) else type(header).__name__
        raise FrameError("header", f"header keys {keys}, expected {sorted(_HEADER_KEYS)}")
    if header["v"] != FRAME_VERSION or type(header["v"]) is not int:
        raise FrameError("version", f"header version {header['v']!r}")
    if header["kind"] not in FRAME_KINDS:
        raise FrameError("kind", f"unknown frame kind {header['kind']!r}")
    if not isinstance(header["meta"], dict):
        raise FrameError("header", "meta must be an object")
    return header, flags, _align(_PREFIX.size + hlen)


def _check_int(value: Any, what: str) -> int:
    if type(value) is not int or value < 0:
        raise FrameError("header", f"{what} must be a non-negative integer, got {value!r}")
    return value


def _check_leaves(header: dict, cap: int) -> list[tuple[np.dtype, tuple[int, ...], int, int]]:
    data_bytes = _check_int(header["data_bytes"], "data_bytes")
    if data_bytes > cap:
        raise FrameError("too_large", f"data region of {data_bytes} bytes exceeds the cap of {cap} bytes")
    lz4_info = header["lz4"]
    if lz4_info is not None and (not isinstance(lz4_info, dict) or set(lz4_info) != {"raw_bytes"}
                                 or lz4_info["raw_bytes"] != data_bytes or type(lz4_info["raw_bytes"]) is not int):
        raise FrameError("header", f"malformed lz4 entry {lz4_info!r}")
    raw = header["leaves"]
    if not isinstance(raw, list):
        raise FrameError("header", "leaves must be a list")
    if len(raw) > MAX_LEAVES:
        raise FrameError("too_many_leaves", f"{len(raw)} leaves, more than {MAX_LEAVES}")
    out = []
    for k, desc in enumerate(raw):
        if type(desc) is not dict or desc.keys() != _LEAF_KEYS:
            raise FrameError("header", f"leaf {k}: malformed descriptor {desc!r}"[:200])
        name, shape, offset, nbytes = desc["dtype"], desc["shape"], desc["offset"], desc["nbytes"]
        dtype = _DTYPE_OBJECTS.get(name) if type(name) is str else None
        if dtype is None:
            raise FrameError("dtype", f"leaf {k}: dtype {name!r} is not allowed")
        if type(shape) is not list or len(shape) > MAX_NDIM \
                or not all(type(d) is int and d >= 0 for d in shape):
            raise FrameError("shape", f"leaf {k}: shape must be a list of at most {MAX_NDIM} non-negative "
                                      f"integers, got {shape!r}"[:200])
        if type(offset) is not int or type(nbytes) is not int or offset < 0 or nbytes < 0:
            raise FrameError("header", f"leaf {k}: offset and nbytes must be non-negative integers")
        if nbytes != math.prod(shape) * dtype.itemsize:
            raise FrameError("nbytes", f"leaf {k}: nbytes {nbytes} != prod({shape}) x {dtype.itemsize}")
        if offset % FRAME_ALIGN:
            raise FrameError("offset", f"leaf {k}: offset {offset} is not {FRAME_ALIGN}-aligned")
        if offset + nbytes > data_bytes:
            raise FrameError("offset", f"leaf {k}: bytes [{offset}, {offset + nbytes}) run past the data region of "
                                       f"{data_bytes} bytes")
        out.append((dtype, tuple(shape), offset, nbytes))
    spans = sorted((o, o + n) for _, _, o, n in out if n)
    for (_, prev_end), (start, _) in zip(spans, spans[1:]):
        if start < prev_end:
            raise FrameError("overlap", f"leaf bytes overlap at offset {start}")
    return out


def _decompress(data: memoryview, raw_bytes: int) -> bytes:
    decompressor = lz4.frame.LZ4FrameDecompressor()
    try:
        out = decompressor.decompress(data, max_length=raw_bytes)
    except RuntimeError as e:
        raise FrameError("lz4", f"corrupt lz4 data: {e}") from None
    if not decompressor.eof:
        if decompressor.needs_input:
            raise FrameError("lz4", "truncated lz4 data")
        raise FrameError("lz4", f"lz4 data decompresses to more than the declared {raw_bytes} bytes")
    if decompressor.unused_data or len(out) != raw_bytes:
        raise FrameError("lz4", f"lz4 data decompresses to {len(out)} bytes (+{len(decompressor.unused_data)} "
                                f"trailing), declared {raw_bytes}")
    return out


class _Decoder:
    def __init__(self, arrays: list[np.ndarray]) -> None:
        self.arrays = arrays
        self.used = [False] * len(arrays)

    def node(self, obj: Any, depth: int) -> Any:
        if depth > MAX_DEPTH:
            raise FrameError("depth", f"tree nested deeper than {MAX_DEPTH} levels")
        tag = obj.get("t") if type(obj) is dict else None
        if tag == "nd" and len(obj) == 2:
            i = obj.get("i")
            if type(i) is not int or not 0 <= i < len(self.arrays) or self.used[i]:
                raise FrameError("leaf_ref", f"bad or repeated leaf reference {i!r}")
            self.used[i] = True
            return self.arrays[i]
        if tag == "dict" and len(obj) == 3:
            keys, values = obj.get("k"), obj.get("v")
            if type(keys) is list and type(values) is list and len(keys) == len(values):
                if not all(isinstance(k, _KEY_TYPES) for k in keys):
                    raise FrameError("dict_key", "dict keys must be str/int/float/bool/null")
                out = {k: self.node(v, depth + 1) for k, v in zip(keys, values)}
                if len(out) != len(keys):
                    raise FrameError("dict_key", "repeated dict key")
                return out
        elif tag == "py" and len(obj) == 2 and "v" in obj:
            if not isinstance(obj["v"], _KEY_TYPES):
                raise FrameError("node", f"scalar node holds a {type(obj['v']).__name__}")
            return obj["v"]
        elif tag in ("list", "tuple") and len(obj) == 2 and type(obj.get("v")) is list:
            items = [self.node(v, depth + 1) for v in obj["v"]]
            return items if tag == "list" else tuple(items)
        elif tag == "nt" and len(obj) == 4 and type(obj.get("cls")) is str and type(obj.get("f")) is list \
                and type(obj.get("v")) is list:
            module, _, qualname = obj["cls"].partition(":")
            cls = lookup_class(module, qualname)
            if cls is None:
                raise FrameError("namedtuple", f"namedtuple {module}.{qualname} is not loaded in this process "
                                               f"(frames never import modules; import {module!r} first)")
            if not (isinstance(cls, type) and issubclass(cls, tuple) and list(getattr(cls, "_fields", ())) == obj["f"]
                    and len(obj["v"]) == len(obj["f"])):
                raise FrameError("namedtuple", f"namedtuple {module}.{qualname} does not match the local class")
            return cls(*(self.node(v, depth + 1) for v in obj["v"]))
        raise FrameError("node", f"malformed tree node {obj!r}"[:200])


def decode_frame(buf: bytes | bytearray | memoryview, *, cap: int, copy: bool = False,
                 kind: str | None = None) -> tuple[Any, dict]:
    """``(tree, meta)`` of a frame. Arrays are views of ``buf`` (read-only for ``bytes``), or with ``copy`` writable
    views of one private copy of the data region (torch then shares them without another copy); raises
    :class:`FrameError` for anything malformed or larger than ``cap`` bytes, before allocating it, and (with ``kind``)
    for a frame of another kind."""
    mv = memoryview(buf)
    if mv.format != "B" or mv.ndim != 1:
        mv = mv.cast("B")
    header, flags, start = _read_header(mv)
    if kind is not None and header["kind"] != kind:
        raise FrameError("kind", f"expected a {kind} frame, got {header['kind']!r}")
    leaves = _check_leaves(header, int(cap))
    data_bytes = header["data_bytes"]
    if bool(flags & FLAG_LZ4) != (header["lz4"] is not None):
        raise FrameError("flags", "the lz4 flag and the header disagree")
    if start > len(mv):
        raise FrameError("truncated", "the frame ends inside the header padding")
    region: Any = mv[start:]
    if header["lz4"] is not None:
        region = _decompress(region, data_bytes)
    elif len(region) != data_bytes:
        raise FrameError("size", f"data region of {len(region)} bytes, declared {data_bytes}")
    if copy:
        region = bytearray(region)                 # one private copy: every array a writable view of it
    arrays = []
    for dtype, shape, offset, nbytes in leaves:
        if nbytes == 0:
            arr = np.zeros(shape, dtype)
        else:
            arr = np.ndarray(shape, dtype, buffer=region, offset=offset)
            if not arr.flags.aligned:
                arr = arr.copy()
        arrays.append(arr)
    decoder = _Decoder(arrays)
    tree = decoder.node(header["tree"], 0)
    if not all(decoder.used):
        raise FrameError("leaf_ref", f"leaf {decoder.used.index(False)} is never referenced")
    return tree, header["meta"]


def peek_meta(buf: bytes | bytearray | memoryview) -> tuple[str, dict]:
    """``(kind, meta)`` from the header only (routing): no arrays, no size cap beyond the header's."""
    mv = memoryview(buf)
    if mv.format != "B" or mv.ndim != 1:
        mv = mv.cast("B")
    header, _, _ = _read_header(mv)
    return header["kind"], header["meta"]


def frame_sizes(buf: bytes | bytearray | memoryview) -> tuple[int, int]:
    """``(raw data bytes, stored data bytes)`` of a well-formed frame (for :meth:`Lz4Probe.observe`)."""
    mv = memoryview(buf)
    if mv.format != "B" or mv.ndim != 1:
        mv = mv.cast("B")
    header, _, start = _read_header(mv)
    return _check_int(header["data_bytes"], "data_bytes"), max(0, len(mv) - start)


class Lz4Probe:
    """Adaptive lz4 for one (sender, agent) (spec 4.9).

    The first ``probe_frames`` frames are sent compressed and measured; compression stays on iff their total raw
    size is at least ``min_ratio`` times their total compressed size. Every ``reprobe_sec`` (sender's monotonic
    clock) the probe repeats, so a stream whose content changes is re-decided. Weights never use it (float32 does not
    compress: spec 4.9)."""

    def __init__(self, *, min_ratio: float = LZ4_MIN_RATIO, probe_frames: int = LZ4_PROBE_FRAMES,
                 reprobe_sec: float = LZ4_REPROBE_SEC, clock: Callable[[], float] = time.monotonic) -> None:
        if probe_frames < 1:
            raise ValueError(f"probe_frames must be >= 1, got {probe_frames}")
        self._min_ratio = float(min_ratio)
        self._probe_frames = int(probe_frames)
        self._reprobe_sec = float(reprobe_sec)
        self._clock = clock
        self._left = self._probe_frames          # probe frames still to measure
        self._raw = 0
        self._stored = 0
        self._enabled = False
        self._next_probe: float | None = None

    @property
    def enabled(self) -> bool:
        """The decision of the last finished probe (False before the first one ends)."""
        return self._enabled

    def compress(self) -> bool:
        """Whether the next frame should be compressed."""
        if self._left == 0 and self._next_probe is not None and self._clock() >= self._next_probe:
            self._left, self._raw, self._stored = self._probe_frames, 0, 0
        return True if self._left > 0 else self._enabled

    def observe(self, raw_bytes: int, compressed_bytes: int | None) -> None:
        """Sizes of a sent frame's data region: ``compressed_bytes`` None = it was sent uncompressed."""
        if compressed_bytes is None or self._left == 0:
            return
        self._raw += int(raw_bytes)
        self._stored += int(compressed_bytes)
        self._left -= 1
        if self._left == 0:
            self._enabled = self._raw >= self._min_ratio * self._stored
            self._next_probe = self._clock() + self._reprobe_sec
```

- [ ] **Step 4: Give `validate_slot_structure` a numpy core**

In `src/colosseum/core/types.py` replace the function `validate_slot_structure` (from its `def` to the end of its body, just before `@dataclass class SeatAssignment`) by the following two functions. The rules, their order (`_SLOT_RULES`), the tie rule and every message stay as they are; only the work moves into `check_slot_arrays` (cheaper than `np.isin` / `np.concatenate`; the existing tests `tests/unit/test_chunk_v2_types.py`, `tests/contract/test_chunk_v2_rules.py`, `tests/unit/test_sp3_dagger_chunks.py` pin the messages):
```python
def validate_slot_structure(chunk: TrajectoryChunk) -> None:
    """Raise ``ValueError`` (naming the agent) if ``chunk``'s slots break the chunk v2 rules.

    The rules the worker's ``RolloutBuffer`` guarantees and the learner's V-trace relies on:
    an ACT is never the last slot; an open (non-terminal) ACT is followed by an ACT or a
    BOOT; a BOOT follows an open ACT; a PAD follows a slot that ends an episode (a terminal
    ACT or a BOOT with ``reset_after``, possibly after other PADs); a BOOT without
    ``reset_after`` takes only the last slot. The flags must match the slot kind (APPO
    resets the recurrent state from ``reset_after``): a terminal ACT and a PAD have
    ``reset_after``, an open ACT has not, and only ACTs are ``terminal``. The earliest broken
    slot is reported. A cheap numpy check over ``kind`` / ``terminal`` / ``reset_after``
    (:func:`check_slot_arrays`); the slot letters for the message are spelled only on failure. A teacher label
    (``has_teacher``) only on ACT slots.
    """
    check_slot_arrays(
        chunk.kind.cpu().numpy(), chunk.terminal.cpu().numpy(), chunk.reset_after.cpu().numpy(),
        None if chunk.has_teacher is None else chunk.has_teacher.cpu().numpy(),
        agent_id=chunk.agent_id, policy_version=chunk.policy_version,
    )


def check_slot_arrays(kind: np.ndarray, terminal: np.ndarray, reset_after: np.ndarray,
                      has_teacher: np.ndarray | None, *, agent_id: str, policy_version: int) -> None:
    """The rules of :func:`validate_slot_structure` on numpy slot arrays (the SP5 learner checks a decoded payload
    before it builds the chunk); the same ValueError messages."""
    terminal = terminal.astype(bool, copy=False)
    reset_after = reset_after.astype(bool, copy=False)
    where = f"agent {agent_id!r}: malformed chunk v2 (policy_version {policy_version}"
    if not (kind.ndim == terminal.ndim == reset_after.ndim == 1
            and kind.shape == terminal.shape == reset_after.shape):
        raise ValueError(
            f"{where}): kind, terminal and reset_after must be 1-D arrays of one length, got shapes "
            f"{kind.shape}, {terminal.shape}, {reset_after.shape}"
        )
    if kind.size == 0:
        raise ValueError(f"{where}): the chunk has no slots")
    if has_teacher is not None:
        has_teacher = has_teacher.astype(bool, copy=False)
        if has_teacher.shape != kind.shape:
            raise ValueError(f"{where}): has_teacher must have the shape {kind.shape} of kind, got "
                             f"{has_teacher.shape}")

    def fail(slot: int, problem: str) -> ValueError:
        return ValueError(f"{where}, slots {_spell_slots(kind, terminal, reset_after)!r}): slot {slot}: {problem}")

    act, boot, pad = kind == SLOT_ACT, kind == SLOT_BOOT, kind == SLOT_PAD
    unknown = ~(act | boot | pad)
    if unknown.any():
        first = int(np.argmax(unknown))
        raise fail(first, f"unknown slot kind {kind[first]}")
    open_act = act & ~terminal
    ends_episode = (act & terminal) | (boot & reset_after)
    prev_open = np.empty_like(open_act)
    prev_open[0], prev_open[1:] = False, open_act[:-1]
    prev_pad = np.empty_like(pad)
    prev_pad[0], prev_pad[1:] = False, pad[:-1]
    prev_ends = np.empty_like(ends_episode)
    prev_ends[0], prev_ends[1:] = False, ends_episode[:-1]
    next_pad = np.empty_like(pad)
    next_pad[-1], next_pad[:-1] = False, pad[1:]
    last = np.zeros(kind.size, dtype=bool)
    last[-1] = True
    broken = (
        act & terminal & ~reset_after,          # a terminal ACT without reset_after
        open_act & reset_after,                 # an open ACT with reset_after
        pad & ~reset_after,                     # a PAD without reset_after
        ~act & terminal,                        # a terminal flag on a non-ACT slot
        act & last,                             # an ACT in the last slot
        open_act & next_pad,                    # an open ACT followed by a PAD
        boot & ~prev_open,                      # a BOOT that does not follow an open ACT
        pad & ~prev_pad & ~prev_ends,           # a PAD (first of a run) not after an episode end
        boot & ~reset_after & ~last,            # a BOOT without reset_after before the last slot
        has_teacher & ~act if has_teacher is not None else None,   # a label off ACT
    )
    any_broken = broken[0] | broken[1] | broken[2] | broken[3] | broken[4] | broken[5] | broken[6] | broken[7] \
        | broken[8]
    if broken[9] is not None:
        any_broken = any_broken | broken[9]
    if not any_broken.any():
        return
    # Earliest broken slot; ties go to the first rule in _SLOT_RULES.
    found = [(int(np.argmax(mask)), r) for r, mask in enumerate(broken) if mask is not None and mask.any()]
    slot, rule = min(found)
    raise fail(slot, _SLOT_RULES[rule])
```

- [ ] **Step 5: Create the chunk, weights and checkpoint frames**

Create `src/colosseum/net/chunks.py`:
```python
"""Chunk, weights and checkpoint frames, and the learner's chunk checks (SP5 spec block 4.9).

- **Chunks:** ``encode_chunk`` frames ``TrajectoryChunk.to_payload()`` (kind ``chunk``; agent, policy version,
  learner generation, sender node and worker in the header meta, so a node routes a frame by ``peek_meta`` without
  decoding it). ``decode_chunk`` checks the frame (``colosseum.net.frame``) and the payload's fields and builds the
  chunk; :class:`ChunkValidator` then checks it against the agent's role on the learner: the slot rules
  (``validate_slot_structure``), every leaf of ``obs`` / ``global_state`` / ``actions`` / ``action_masks`` /
  ``teacher_action`` with shape ``[S, ...]`` and the dtype the worker's buffers use, the chunk length, finite
  ``behavior_logp`` (and ``behavior_unit_logp``) on ACT slots, finite rewards and zero rewards on BOOT / PAD, discrete
  actions (and teacher actions) inside the action space and finite box values, the recorded action legal under its
  mask, ``teacher_action`` / ``has_teacher`` both or neither (only for an agent with a scripted teacher), the agent
  id, and (with a template) the structure of ``initial_state``. Every failure is a :class:`ChunkError` with reason
  ``invalid`` and a ``detail``; the learner drops the chunk and counts it (spec 4.7), it never stops.
- **Weights:** ``encode_weights`` / ``decode_weights``: a numpy ``state_dict`` with :class:`WeightsMeta` (agent,
  kind, learner generation, policy version, ``teacher_active``, optional checkpoint id, architecture and roles).
  Never compressed (float32 does not compress).
- **Checkpoints:** ``encode_checkpoint`` / ``decode_checkpoint``: ``learner.make_checkpoint_payload()`` with its
  ``torch.save`` trainer-state bytes as one ``uint8`` leaf; the hub stores them as bytes and never unpickles them
  (a learner reads its own trainer state with ``torch.load(..., weights_only=True)``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from colosseum.core.config import check_agent_id
from colosseum.core.errors import ConfigError
from colosseum.core.specs import ActionSpec, ObsSpec, split_unit_actions
from colosseum.core.types import SLOT_ACT, TrajectoryChunk, check_slot_arrays
from colosseum.envs.game import RoleSpec
from colosseum.net.frame import FrameError, decode_frame, encode_frame, peek_meta
from colosseum.networks.state import State

WEIGHTS_KINDS = ("latest", "snapshot", "fixed", "teacher", "init", "recovery")
CHECKPOINT_KINDS = ("periodic", "final", "recovery")
_FLAT = ("kind", "reward", "terminal", "reset_after", "behavior_logp")
_FLAT_DTYPES = {"kind": np.int8, "reward": np.float32, "terminal": np.bool_, "reset_after": np.bool_,
                "behavior_logp": np.float32}
_PAYLOAD_KEYS = {"initial_state", "obs", "global_state", "actions", "action_masks", *_FLAT, "behavior_unit_logp",
                 "teacher_action", "has_teacher"}


class ChunkError(ValueError):
    """A chunk the learner drops (spec 4.7): ``reason`` is the ``dropped_chunks`` key, ``detail`` says what."""

    def __init__(self, reason: Literal["invalid", "stale", "stale_generation"], detail: str) -> None:
        super().__init__(f"{reason} chunk: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ChunkMeta:
    agent_id: str
    policy_version: int
    generation: int          # learner generation of the weights with policy_version (0 = unknown / SP4 path)
    node: int                # sender node number (-1 in local tests)
    worker: int


def encode_chunk(chunk: TrajectoryChunk, *, generation: int, node: int, worker: int,
                 compress: bool = False) -> bytes:
    """One ``chunk`` frame of ``chunk`` (its numpy payload; agent and version travel in the header meta)."""
    payload = chunk.to_payload()
    meta = {"agent_id": payload.pop("agent_id"), "policy_version": payload.pop("policy_version"),
            "generation": int(generation), "node": int(node), "worker": int(worker)}
    return encode_frame(payload, kind="chunk", meta=meta, compress=compress)


def _int_field(meta: Mapping[str, Any], key: str, low: int) -> int:
    value = meta.get(key)
    if type(value) is not int or value < low:
        raise ChunkError("invalid", f"chunk meta {key!r} must be an integer >= {low}, got {value!r}")
    return value


def chunk_meta(meta: Mapping[str, Any]) -> ChunkMeta:
    """The :class:`ChunkMeta` of a ``chunk`` frame's header meta (``peek_meta`` result); ChunkError if malformed."""
    if set(meta) != {"agent_id", "policy_version", "generation", "node", "worker"}:
        raise ChunkError("invalid", f"chunk meta keys {sorted(meta)}")
    agent_id = meta["agent_id"]
    try:
        check_agent_id(agent_id if isinstance(agent_id, str) else "")
    except ConfigError:
        raise ChunkError("invalid", f"chunk meta agent_id {agent_id!r}") from None
    return ChunkMeta(agent_id, _int_field(meta, "policy_version", 0), _int_field(meta, "generation", 0),
                     _int_field(meta, "node", -1), _int_field(meta, "worker", 0))


def _check_tree(tree: Any, name: str, num_slots: int) -> None:
    """A numpy array or a dict tree of them (str keys) with the slot dim first."""
    if isinstance(tree, np.ndarray):
        if tree.ndim == 0 or tree.shape[0] != num_slots:
            raise ChunkError("invalid", f"field {name!r} has a leaf of shape {tree.shape}, expected S={num_slots} "
                                        f"first")
        return
    if isinstance(tree, dict) and tree and all(isinstance(k, str) for k in tree):
        for key, value in tree.items():
            _check_tree(value, f"{name}/{key}", num_slots)
        return
    raise ChunkError("invalid", f"field {name!r} must be a numpy array or a dict tree of them, "
                                f"got {type(tree).__name__}")


def _check_state_nodes(state: Any, depth: int = 0) -> None:
    if state is None or isinstance(state, np.ndarray):
        return
    if depth < 64 and isinstance(state, tuple | list):
        for item in state:
            _check_state_nodes(item, depth + 1)
        return
    if depth < 64 and isinstance(state, dict):
        for item in state.values():
            _check_state_nodes(item, depth + 1)
        return
    raise ChunkError("invalid", f"initial_state holds a {type(state).__name__}")


def check_payload_fields(payload: Any) -> int:
    """Field checks of a chunk payload without ``agent_id`` / ``policy_version`` (presence, types, one slot dim);
    returns ``S``. ChunkError("invalid") otherwise."""
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
        keys = sorted(payload, key=str) if isinstance(payload, dict) else type(payload).__name__
        raise ChunkError("invalid", f"payload fields {keys}")
    for name in _FLAT:
        value = payload[name]
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise ChunkError("invalid", f"field {name!r} must be a 1-D array [S]")
    num_slots = payload["kind"].shape[0]
    if num_slots < 2:
        raise ChunkError("invalid", f"a chunk needs at least 2 slots, got {num_slots}")
    for name in _FLAT:
        if payload[name].shape[0] != num_slots:
            raise ChunkError("invalid", f"field {name!r} has shape {payload[name].shape}, expected [S={num_slots}]")
    for name in ("obs", "actions"):
        _check_tree(payload[name], name, num_slots)
    for name in ("global_state", "action_masks", "teacher_action"):
        if payload[name] is not None:
            _check_tree(payload[name], name, num_slots)
    for name in ("behavior_unit_logp", "has_teacher"):
        value = payload[name]
        if value is not None and (not isinstance(value, np.ndarray) or value.ndim == 0 or value.shape[0] != num_slots):
            raise ChunkError("invalid", f"field {name!r} must be None or an array with S={num_slots} first")
    if (payload["teacher_action"] is None) != (payload["has_teacher"] is None):
        raise ChunkError("invalid", "fields 'teacher_action' and 'has_teacher' come together")
    _check_state_nodes(payload["initial_state"])
    return num_slots


def decode_chunk(buf: bytes | bytearray | memoryview, *, cap: int,
                 validator: ChunkValidator | None = None) -> tuple[TrajectoryChunk, ChunkMeta]:
    """The chunk of a ``chunk`` frame and its meta (CPU tensors). Checks the frame, the meta and the payload's fields
    (presence, types, one slot dim ``S``); with ``validator`` also every role check of spec 4.9 (the learner's
    intake path: frame -> decode -> check -> batch). Every problem is a ChunkError("invalid"); ``detail`` starts
    with ``frame:`` for frame errors."""
    try:
        payload, meta = decode_frame(buf, cap=cap, kind="chunk", copy=True)
    except FrameError as e:
        raise ChunkError("invalid", f"frame:{e.reason}: {e}") from None
    info = chunk_meta(meta)
    check_payload_fields(payload)
    if validator is not None:
        validator.check(payload, info.agent_id, info.policy_version)
    payload["agent_id"], payload["policy_version"] = info.agent_id, info.policy_version
    try:
        chunk = TrajectoryChunk.from_payload(payload)
    except (KeyError, TypeError, ValueError) as e:
        raise ChunkError("invalid", f"malformed payload: {type(e).__name__}: {e}") from None
    return chunk, info


# ---------------------------------------------------------------------------
# The learner's checks
# ---------------------------------------------------------------------------


def _template(tree: Any, path: str = "") -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """``{path: (shape without the leading dim, dtype)}`` of a numpy template allocated with leading ``(1,)`` (a
    bare array has the path ``""``)."""
    if isinstance(tree, np.ndarray):
        return {path: (tuple(tree.shape[1:]), tree.dtype)}
    out: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
    for key, value in tree.items():
        out.update(_template(value, f"{path}/{key}" if path else key))
    return out


def _leaves(tree: Any, path: str = "") -> dict[str, np.ndarray]:
    if isinstance(tree, np.ndarray):
        return {path: tree}
    out: dict[str, np.ndarray] = {}
    for key, value in tree.items():
        out.update(_leaves(value, f"{path}/{key}" if path else key))
    return out


def _state_template(state: Any) -> Any:
    """Structure of a numpy state (batch 1): ('nd', shape, dtype) / ('seq', type, items) / ('dict', keys, items) /
    ('none',)."""
    if state is None:
        return ("none",)
    if isinstance(state, np.ndarray):
        return ("nd", tuple(state.shape), state.dtype)
    if isinstance(state, tuple | list):
        return ("seq", type(state), tuple(_state_template(s) for s in state))
    return ("dict", tuple(state), tuple(_state_template(v) for v in state.values()))


class ChunkValidator:
    """Spec 4.9 checks of a chunk on its agent's learner (see the module docstring), on the numpy payload.

    ``role_spec``: the agent's spaces (``agent_role_spec``); ``teacher``: the agent has a scripted kickstart teacher
    (its chunks carry DAgger labels); ``chunk_length``: the expected ``S`` (``rollout.chunk_length``; None = any);
    ``state_template``: the model's ``initial_state(1)`` as numpy (``state_to_numpy``), whose structure, shapes and
    dtypes every chunk's ``initial_state`` must have (None = not checked)."""

    def __init__(self, role_spec: RoleSpec, *, agent_id: str, teacher: bool, chunk_length: int | None = None,
                 state_template: State | None = None) -> None:
        self._agent_id = agent_id
        self._teacher = bool(teacher)
        self._chunk_length = chunk_length
        self._obs = _template(ObsSpec.from_space(role_spec.observation_space).allocate((1,)))
        gs = role_spec.global_state_space
        self._gs = None if gs is None else _template(ObsSpec.from_space(gs).allocate((1,)))
        self._action_spec = ActionSpec.from_space(role_spec.action_space)
        self._actions = _template(self._action_spec.allocate_actions((1,)))
        self._masks = _template(self._action_spec.full_mask((1,))) if self._action_spec.has_masks else None
        self._deciders = self._action_spec.num_deciders
        self._state = None if state_template is None else _state_template(state_template)
        self._flat = {name: np.dtype(dtype) for name, dtype in _FLAT_DTYPES.items()}

    def _fail(self, detail: str) -> ChunkError:
        return ChunkError("invalid", f"agent {self._agent_id!r}: {detail}")

    def _check_leaves(self, tree: Any, template: dict, name: str, num_slots: int) -> None:
        if isinstance(tree, np.ndarray) != ("" in template):
            raise self._fail(f"{name}: expected {'an array' if '' in template else 'a dict tree'}")
        leaves = _leaves(tree)
        if leaves.keys() != template.keys():
            raise self._fail(f"{name}: leaves {sorted(leaves)}, expected {sorted(template)}")
        for path, leaf in leaves.items():
            shape, dtype = template[path]
            if leaf.shape != (num_slots, *shape) or leaf.dtype != dtype:
                raise self._fail(f"{name}/{path or '<root>'}: {leaf.dtype} {leaf.shape}, expected {dtype} "
                                 f"{(num_slots, *shape)}")

    def _check_action_values(self, actions: Any, name: str) -> None:
        """Discrete values inside the space, box values finite (every slot: BOOT / PAD carry zeros)."""
        for group in self._action_spec.groups:
            value = actions if not self._action_spec.is_dict else ActionSpec._get(actions, group.path)
            where = f"{name}/{'/'.join(group.path) or '<root>'}"
            if group.kind == "discrete":
                bad = (value < 0) | (value >= group.nvec[0])
            elif group.kind == "multi_discrete":
                bad = (value < 0) | (value >= np.asarray(group.nvec))
            elif group.kind == "box":
                bad = ~np.isfinite(value)
            else:
                parts = split_unit_actions(group.units, value)
                for comp in group.units.components:
                    part = parts[comp.name]
                    bad = (part < 0) | (part >= comp.size) if comp.kind == "discrete" else ~np.isfinite(part)
                    if bad.any():
                        raise self._fail(f"{where} component {comp.name!r}: a value outside the action space")
                continue
            if bad.any():
                raise self._fail(f"{where}: a value outside the action space")

    def check(self, payload: Mapping[str, Any], agent_id: str, policy_version: int | None = None) -> None:
        """Every check (the slot rules too) on a numpy payload that passed ``check_payload_fields``."""
        if agent_id != self._agent_id:
            raise self._fail(f"a chunk of agent {agent_id!r}")
        num_slots = payload["kind"].shape[0]
        if self._chunk_length is not None and num_slots != self._chunk_length:
            raise self._fail(f"{num_slots} slots, expected rollout.chunk_length {self._chunk_length}")
        for name, dtype in self._flat.items():
            if payload[name].dtype != dtype:
                raise self._fail(f"{name}: {payload[name].dtype}, expected {dtype}")
        self._check_leaves(payload["obs"], self._obs, "obs", num_slots)
        if (payload["global_state"] is None) != (self._gs is None):
            raise self._fail("global_state present for a role without one" if self._gs is None
                             else "global_state missing")
        if self._gs is not None:
            self._check_leaves(payload["global_state"], self._gs, "global_state", num_slots)
        self._check_leaves(payload["actions"], self._actions, "actions", num_slots)
        if (payload["action_masks"] is None) != (self._masks is None):
            raise self._fail("action_masks present for an action space without masks" if self._masks is None
                             else "action_masks missing")
        if self._masks is not None:
            self._check_leaves(payload["action_masks"], self._masks, "action_masks", num_slots)
        unit = payload["behavior_unit_logp"]
        if (unit is None) != (self._deciders <= 1):
            raise self._fail(f"behavior_unit_logp {'missing' if unit is None else 'present'} for "
                             f"{self._deciders} decider(s)")
        if unit is not None and (unit.shape != (num_slots, self._deciders) or unit.dtype != np.float32):
            raise self._fail(f"behavior_unit_logp: {unit.dtype} {unit.shape}, expected float32 "
                             f"({num_slots}, {self._deciders})")
        if (payload["teacher_action"] is not None) != self._teacher:
            raise self._fail("DAgger labels on a chunk of an agent without a scripted teacher" if not self._teacher
                             else "DAgger labels missing")
        act = payload["kind"] == SLOT_ACT
        if not np.isfinite(payload["behavior_logp"][act]).all():
            raise self._fail("a non-finite behavior_logp on an ACT slot")
        if unit is not None and not np.isfinite(unit[act]).all():
            raise self._fail("a non-finite behavior_unit_logp on an ACT slot")
        reward = payload["reward"]
        if not np.isfinite(reward).all():
            raise self._fail("a non-finite reward")
        if (reward[~act] != 0).any():
            raise self._fail("a non-zero reward on a BOOT or PAD slot")
        self._check_action_values(payload["actions"], "actions")
        if self._teacher:
            self._check_leaves(payload["teacher_action"], self._actions, "teacher_action", num_slots)
            has = payload["has_teacher"]
            if has.dtype != np.bool_ or has.shape != (num_slots,):
                raise self._fail(f"has_teacher: {has.dtype} {has.shape}, expected bool ({num_slots},)")
            self._check_action_values(payload["teacher_action"], "teacher_action")
        if self._state is not None and _state_template(payload["initial_state"]) != self._state:
            raise self._fail("initial_state does not match the model's state structure, shapes and dtypes")
        try:
            check_slot_arrays(payload["kind"], payload["terminal"], payload["reset_after"], payload["has_teacher"],
                              agent_id=agent_id, policy_version=-1 if policy_version is None else policy_version)
        except ValueError as e:
            raise ChunkError("invalid", str(e)) from None

    def validate(self, chunk: TrajectoryChunk) -> None:
        """Every check on a chunk that is already built (e.g. from an SP4 payload on a queue)."""
        payload = chunk.to_payload()
        agent_id = payload.pop("agent_id")
        version = payload.pop("policy_version")
        check_payload_fields(payload)
        self.check(payload, agent_id, version)


# ---------------------------------------------------------------------------
# Weights and checkpoints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightsMeta:
    agent_id: str
    kind: Literal["latest", "snapshot", "fixed", "teacher", "init", "recovery"]
    generation: int
    policy_version: int
    teacher_active: bool = False
    checkpoint_id: str | None = None
    architecture: dict | None = None     # meta.json "networks" (or a frozen entry's) for fixed / teacher / init
    roles: tuple[str, ...] | None = None


def encode_weights(state_dict: Mapping[str, np.ndarray], meta: WeightsMeta) -> bytes:
    """One ``weights`` frame (never compressed). ``state_dict`` is numpy (``state_dict_to_numpy``)."""
    bad = [k for k, v in state_dict.items() if not isinstance(k, str) or not isinstance(v, np.ndarray)]
    if bad:
        raise TypeError(f"weights must be a dict of str -> numpy array, got {bad[0]!r}")
    if meta.kind not in WEIGHTS_KINDS:
        raise ValueError(f"weights kind must be one of {WEIGHTS_KINDS}, got {meta.kind!r}")
    data = asdict(meta)
    data["roles"] = None if meta.roles is None else list(meta.roles)
    return encode_frame(dict(state_dict), kind="weights", meta=data)


def _weights_meta(meta: Mapping[str, Any]) -> WeightsMeta:
    fields = {"agent_id", "kind", "generation", "policy_version", "teacher_active", "checkpoint_id", "architecture",
              "roles"}
    ok = (set(meta) == fields and isinstance(meta["agent_id"], str) and meta["kind"] in WEIGHTS_KINDS
          and type(meta["generation"]) is int and meta["generation"] >= 0
          and type(meta["policy_version"]) is int and meta["policy_version"] >= 0
          and isinstance(meta["teacher_active"], bool)
          and (meta["checkpoint_id"] is None or isinstance(meta["checkpoint_id"], str))
          and (meta["architecture"] is None or isinstance(meta["architecture"], dict))
          and (meta["roles"] is None or (isinstance(meta["roles"], list)
                                         and all(isinstance(r, str) for r in meta["roles"]))))
    if not ok:
        raise FrameError("weights_meta", f"malformed weights meta {dict(meta)!r}"[:300])
    roles = None if meta["roles"] is None else tuple(meta["roles"])
    return WeightsMeta(meta["agent_id"], meta["kind"], meta["generation"], meta["policy_version"],
                       meta["teacher_active"], meta["checkpoint_id"], meta["architecture"], roles)


def decode_weights(buf: bytes | bytearray | memoryview, *, cap: int) -> tuple[dict[str, np.ndarray], WeightsMeta]:
    """``(state_dict, meta)`` of a ``weights`` frame (views of ``buf``; FrameError if malformed)."""
    kind, _ = peek_meta(buf)
    if kind != "weights":
        raise FrameError("kind", f"expected a weights frame, got {kind!r}")
    tree, meta = decode_frame(buf, cap=cap)
    if not isinstance(tree, dict) or not all(isinstance(k, str) and isinstance(v, np.ndarray) for k, v in tree.items()):
        raise FrameError("weights", "a weights frame holds a dict of str -> array")
    return tree, _weights_meta(meta)


def encode_checkpoint(payload: Mapping[str, Any], *, kind: Literal["periodic", "final", "recovery"],
                      generation: int) -> bytes:
    """One ``checkpoint`` frame of ``learner.make_checkpoint_payload()`` (trainer-state bytes as a uint8 leaf)."""
    if kind not in CHECKPOINT_KINDS:
        raise ValueError(f"checkpoint kind must be one of {CHECKPOINT_KINDS}, got {kind!r}")
    trainer = payload.get("trainer_state_bytes")
    tree = {"model_state": dict(payload["model_state"]),
            "trainer_state": None if trainer is None else np.frombuffer(trainer, dtype=np.uint8)}
    meta = {"agent_id": str(payload["agent_id"]), "policy_version": int(payload["policy_version"]),
            "final": bool(payload.get("final", False)), "kind": kind, "generation": int(generation)}
    return encode_frame(tree, kind="checkpoint", meta=meta)


def decode_checkpoint(buf: bytes | bytearray | memoryview, *, cap: int) -> tuple[dict, dict]:
    """``(payload as make_checkpoint_payload, meta)`` of a ``checkpoint`` frame; FrameError if malformed."""
    kind, _ = peek_meta(buf)
    if kind != "checkpoint":
        raise FrameError("kind", f"expected a checkpoint frame, got {kind!r}")
    tree, meta = decode_frame(buf, cap=cap)
    ok = (isinstance(tree, dict) and set(tree) == {"model_state", "trainer_state"}
          and isinstance(tree["model_state"], dict)
          and all(isinstance(k, str) and isinstance(v, np.ndarray) for k, v in tree["model_state"].items())
          and (tree["trainer_state"] is None or (isinstance(tree["trainer_state"], np.ndarray)
                                                 and tree["trainer_state"].dtype == np.uint8
                                                 and tree["trainer_state"].ndim == 1))
          and set(meta) == {"agent_id", "policy_version", "final", "kind", "generation"}
          and isinstance(meta["agent_id"], str) and type(meta["policy_version"]) is int
          and meta["policy_version"] >= 0 and isinstance(meta["final"], bool) and meta["kind"] in CHECKPOINT_KINDS
          and type(meta["generation"]) is int and meta["generation"] >= 0)
    if not ok:
        raise FrameError("checkpoint", "malformed checkpoint frame")
    trainer = tree["trainer_state"]
    payload = {"agent_id": meta["agent_id"], "policy_version": meta["policy_version"],
               "model_state": tree["model_state"],
               "trainer_state_bytes": None if trainer is None else trainer.tobytes(), "final": meta["final"]}
    return payload, dict(meta)
```

Design notes for the implementer (they are the reasons behind the code; keep them in mind when adapting):
- `decode_chunk` decodes with `copy=True`: one private copy of the data region (`bytearray`), every array a writable view of it, so `TrajectoryChunk.from_payload` shares the memory with torch (`numpy_to_tensor` copies read-only arrays, which would cost one allocation per leaf).
- The validator works on the numpy payload, before `from_payload` (criterion 5): the role templates are built once in `__init__` from `ObsSpec.allocate((1,))` / `ActionSpec.allocate_actions((1,))` / `full_mask((1,))` — exactly what the worker's `RolloutBuffer` allocates, so a buffer-built chunk always passes.
- Mask legality of the recorded actions is **not** checked here (ruling P1-2): the worker's legality gate guarantees it, and `ActionSpec.first_illegal_action` stays the one home of the mask rules.

- [ ] **Step 6: Check a model's state on the frame codec**

In `src/colosseum/core/validation.py` replace `_check_state_payload` by:
```python
def _check_state_payload(state: Any, where: str) -> None:
    """The model state must survive the inter-process round trip: chunks carry ``initial_state`` to the learner in
    an array frame (SP5 spec 4.9; ``colosseum.net.frame``), which takes numpy leaves in dict / list / tuple and
    importable namedtuples only."""
    from colosseum.net.frame import decode_frame, encode_frame, payload_byte_cap
    from colosseum.networks.state import slice_batch, state_from_numpy, state_to_numpy

    if state is None:
        return
    try:
        tree, _ = decode_frame(encode_frame(state_to_numpy(slice_batch(state, 0))), cap=payload_byte_cap(64))
        state_from_numpy(tree)
    except (TypeError, ValueError) as e:
        raise ConfigError(f"{where}: initial_state cannot be sent between processes: {e}") from e
```

- [ ] **Step 7: Run the focused tests (GREEN)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_frame.py tests/unit/test_sp5_chunks.py tests/unit/test_chunk_v2_types.py tests/contract/test_chunk_v2_rules.py tests/unit/test_sp3_dagger_chunks.py tests/unit/test_sp2_validate.py -q -rw`
Expected: all pass (61 + 48 new), zero warnings.

- [ ] **Step 8: Write the criterion-5 measurement script and run it on a quiet machine**

Create `scripts/measure_frame_decode.py` (the rule is in its docstring and fixed now, before measuring):
```python
"""SP5 criterion 5 (spec §3): parsing a 256-slot chunk of space_miners' size (~260 KB) together with the hostile-input
checks and the chunk checks is at least 10x faster on the new frame than on today's npz + JSON path with
``validate_chunk_payload``.

    env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/measure_frame_decode.py [--repeats 1000] [--json OUT]

Both sides start from the bytes a learner receives and end with a checked numpy payload:
- old (today's wire, ``GRPCTransport.send_chunk``): ``deserialize_chunk_payload`` of the lz4 npz + JSON blob +
  ``validate_chunk_payload`` (field checks, building the chunk once, slot rules); also reported without lz4;
- new (a worker's frame to a learner on its machine, never compressed): ``decode_frame`` (hostile-input checks) +
  ``chunk_meta`` + ``check_payload_fields`` + ``ChunkValidator.check`` (role checks, slot rules, finiteness, zero
  rewards, action ranges); also reported for an lz4 frame (another machine).
For information it also reports both sides up to a built ``TrajectoryChunk`` (old + ``from_payload``, new
``decode_chunk(..., validator=...)``): building torch tensors is the same work on both sides and not parsing.

The role is space_miners' (``examples/space_miners/game.py``; spaces copied here so Box2D is not needed): Dict
observations of 924 B per slot, ``Units(3, Dict(accel=Box(2), push=Discrete(2)))`` actions with masks, 3 deciders.
The chunk is built by the worker's own ``RolloutBuffer`` (255 ACTs and a BOOT) with random content.

Rule (fixed before measuring, T1.1): PASS iff median(old, lz4) / median(new, plain) >= 10. Medians of ``--repeats``
timed runs after 100 warm-up ones, ``time.perf_counter`` per run (a duration on one machine, not a wall-clock
comparison); alone on a quiet machine, two runs, both must pass.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import gymnasium
import numpy as np

from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.types import TrajectoryChunk
from colosseum.envs.game import RoleSpec
from colosseum.envs.spaces import Units
from colosseum.net.chunks import ChunkValidator, check_payload_fields, chunk_meta, decode_chunk, encode_chunk
from colosseum.net.frame import decode_frame, payload_byte_cap
from colosseum.transport.serialization import (
    deserialize_chunk_payload,
    serialize_chunk_payload,
    validate_chunk_payload,
)
from colosseum.worker.buffers import BufferSpec, RolloutBuffer

SLOTS = 256
RATIO_MIN = 10.0
WARMUP = 100
MAX_SHIPS, SHIP_FEATURES, ASTEROID_FEATURES, MAX_ASTEROIDS = 3, 9, 7, 24   # examples/space_miners/game.py


def space_miners_role() -> RoleSpec:
    box = gymnasium.spaces.Box
    obs = gymnasium.spaces.Dict([
        ("ships", box(-1.0, 1.0, (MAX_SHIPS, SHIP_FEATURES), np.float32)),
        ("enemy_ships", box(-1.0, 1.0, (MAX_SHIPS, SHIP_FEATURES), np.float32)),
        ("asteroids", box(-1.0, 1.0, (MAX_ASTEROIDS, ASTEROID_FEATURES), np.float32)),
        ("asteroid_mask", gymnasium.spaces.MultiBinary(MAX_ASTEROIDS)),
        ("global", box(-1.0, 1.0, (3,), np.float32)),
    ])
    per_ship = gymnasium.spaces.Dict([("accel", box(-1.0, 1.0, (2,), np.float32)),
                                      ("push", gymnasium.spaces.Discrete(2))])
    return RoleSpec(obs, Units(MAX_SHIPS, per_ship))


def build_chunk(role: RoleSpec, seed: int = 0):
    rng = np.random.default_rng(seed)
    spec = BufferSpec(ObsSpec.from_space(role.observation_space), ActionSpec.from_space(role.action_space), None)
    buf = RolloutBuffer(SLOTS, spec)
    buf.begin(None, 7)
    mask = spec.action.full_mask(())
    for _ in range(SLOTS - 1):
        obs = {k: (rng.random(v.shape) * 2 - 1).astype(v.dtype) if v.dtype == np.float32
               else rng.integers(0, 2, v.shape).astype(v.dtype) for k, v in spec.obs.allocate(()).items()}
        action = {"accel": rng.uniform(-1, 1, (MAX_SHIPS, 2)).astype(np.float32),
                  "push": rng.integers(0, 2, MAX_SHIPS).astype(np.int64)}
        buf.write_act(obs, None, mask, action, -1.5, np.full(MAX_SHIPS, -0.5, np.float32), 0.01)
    buf.write_boot(obs, None, reset_after=False)
    return buf.build_chunk("main")


def timed(fn, repeats: int) -> float:
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--json", type=str, default=None, help="write the numbers to this file")
    args = parser.parse_args()
    role = space_miners_role()
    chunk = build_chunk(role)
    validator = ChunkValidator(role, agent_id="main", teacher=False, chunk_length=SLOTS)
    cap = payload_byte_cap(64)
    payload = chunk.to_payload()
    blobs = {"old_lz4": serialize_chunk_payload(payload, compress=True)[0],
             "old_plain": serialize_chunk_payload(payload, compress=False)[0],
             "new_plain": encode_chunk(chunk, generation=1, node=1, worker=0),
             "new_lz4": encode_chunk(chunk, generation=1, node=1, worker=0, compress=True)}

    def old(data: bytes, compressed: bool, build: bool = False) -> None:
        decoded = deserialize_chunk_payload(data, compressed)
        validate_chunk_payload(decoded)
        if build:
            TrajectoryChunk.from_payload(decoded)

    def new(data: bytes) -> None:
        decoded, meta = decode_frame(data, cap=cap, kind="chunk")
        info = chunk_meta(meta)
        check_payload_fields(decoded)
        validator.check(decoded, info.agent_id, info.policy_version)

    t = {"old_lz4": timed(lambda: old(blobs["old_lz4"], True), args.repeats),
         "old_plain": timed(lambda: old(blobs["old_plain"], False), args.repeats),
         "new_plain": timed(lambda: new(blobs["new_plain"]), args.repeats),
         "new_lz4": timed(lambda: new(blobs["new_lz4"]), args.repeats),
         "old_lz4_chunk": timed(lambda: old(blobs["old_lz4"], True, build=True), args.repeats),
         "new_plain_chunk": timed(lambda: decode_chunk(blobs["new_plain"], cap=cap, validator=validator),
                                  args.repeats)}
    ratio = t["old_lz4"] / t["new_plain"]
    verdict = "PASS" if ratio >= RATIO_MIN else "FAIL"
    for key, value in t.items():
        size = len(blobs[key.removesuffix("_chunk")])
        print(f"{key:16s} {size / 1024:7.1f} KB  median {value * 1e6:8.1f} us")
    print(f"criterion 5: old_lz4 / new_plain = {ratio:.1f} (>= {RATIO_MIN}: {verdict}); for information: "
          f"old_plain / new_plain = {t['old_plain'] / t['new_plain']:.1f}, old_lz4 / new_lz4 = "
          f"{t['old_lz4'] / t['new_lz4']:.1f}, up to a built chunk = {t['old_lz4_chunk'] / t['new_plain_chunk']:.1f}")
    if args.json:
        Path(args.json).write_text(json.dumps({"slots": SLOTS, "repeats": args.repeats, "seconds": t,
                                               "bytes": {k: len(v) for k, v in blobs.items()}, "ratio": ratio,
                                               "ratio_min": RATIO_MIN, "verdict": verdict}, indent=2) + "\n")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
```
Run alone on a quiet machine, twice:
```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/measure_frame_decode.py --json /tmp/frame-1.json
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/measure_frame_decode.py --json /tmp/frame-2.json
```
Expected: `criterion 5: old_lz4 / new_plain = 13–15 (>= 10.0: PASS)` both times, exit 0 (scratch: 13.3 and 14.5). Put both outputs (all six lines) into the task report; T7.4 copies them into `docs/benchmarks.md`. If a run fails, do not tune the rule: report the numbers (and a `cProfile` of `decode_chunk(..., validator=...)`), the controller decides.

- [ ] **Step 9: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings; ruff clean.

- [ ] **Step 10: Commit and push**

```bash
git add src/colosseum/net/__init__.py src/colosseum/net/frame.py src/colosseum/net/chunks.py src/colosseum/core/types.py \
        src/colosseum/core/validation.py scripts/measure_frame_decode.py tests/unit/test_sp5_frame.py tests/unit/test_sp5_chunks.py
git commit -m "feat: SP5 array frame, chunk / weights / checkpoint frames and the learner's chunk checks"
git push
```

---

### Task T1.2: Protocol v1, generated code and converters; record fields; runtime dependencies

Spec blocks 4.1 (session messages), 4.9 ("Управляющие сообщения ... типизированный protobuf; схема MatchResult повторяет контракт записи матча SP4 плюс поколение места и номер узла. Версия протокола — в Hello и в Bootstrap; policy_version и счётчики — int64") and 3.6 (`grpcio` already there). The `.proto` holds every message of the overview's outline with fixed field numbers; generated code is committed; `colosseum.net.messages` converts the SP4 match record, lineups and version announcements. `Lineup.epoch`, `SeatResult.generation` and `MatchResult.node` are added (defaults None; used from T2.2 / T2.5). `grpcio` / `protobuf` become runtime dependencies (the hub and every `train` need them), `grpcio-tools` moves to `dev`.

**Files:**
- Create: `src/colosseum/net/proto/__init__.py` (empty), `src/colosseum/net/proto/colosseum_v1.proto`, generated `src/colosseum/net/proto/colosseum_v1_pb2.py` and `colosseum_v1_pb2_grpc.py`, `scripts/gen_proto.sh`, `src/colosseum/net/messages.py`
- Modify: `src/colosseum/core/types.py` (three fields; nothing else), `pyproject.toml` (dependencies, extras, ruff excludes), `scripts/setup-dev.sh` (extras)
- Test: `tests/unit/test_sp5_messages.py`

**Interfaces:**
- Consumes: `Lineup`, `SeatAssignment`, `SeatResult`, `TeamResult`, `MatchResult` (SP4).
- Produces: `colosseum.net.proto.colosseum_v1_pb2` (`pb`) / `_pb2_grpc` (`HubStub`, `BootstrapStub`, `NodeDataStub`, `add_*Servicer_to_server`); `colosseum.net.messages.{PROTOCOL_VERSION, protocol_compatible, VersionInfo, result_to_proto, result_from_proto, results_to_batch, results_from_batch, seat_result_to_proto, seat_result_from_proto, lineup_to_proto, lineup_from_proto, version_to_proto, version_from_proto}` (contract notes 4–5); `Lineup.epoch`, `SeatResult.generation`, `MatchResult.node`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_sp5_messages.py`:
```python
"""Protocol v1 (SP5 T1.2, spec blocks 4.1 and 4.9): the generated code matches the .proto, every SP4 match-record and
lineup variant round-trips through protobuf bytes (None stays unset, int64 versions and counters), version
announcements round-trip, the services and messages of the contract exist, and the protocol version rule."""
from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from colosseum.core.types import Lineup, MatchResult, SeatAssignment, SeatResult, TeamResult
from colosseum.net import messages
from colosseum.net.messages import (
    PROTOCOL_VERSION,
    VersionInfo,
    lineup_from_proto,
    lineup_to_proto,
    protocol_compatible,
    result_from_proto,
    result_to_proto,
    results_from_batch,
    results_to_batch,
    version_from_proto,
    version_to_proto,
)
from colosseum.net.proto import colosseum_v1_pb2 as pb
from colosseum.net.proto import colosseum_v1_pb2_grpc as pb_grpc

PROTO = Path(messages.__file__).resolve().parent / "proto" / "colosseum_v1.proto"
BIG = 2**62 + 5                       # an int64 that int32 cannot hold


def _seat(seat: int, **kw) -> SeatResult:
    return SeatResult(seat, "player", seat, kw.pop("agent", "main"), kw.pop("network", "latest"),
                      kw.pop("reward", 1.0 - 2 * seat), **kw)


RESULTS = [
    # SP3 shape: only the positional fields
    MatchResult("m0", "2p", "wdl", [_seat(0), _seat(1)], [TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)], 9),
    # SP4 training record: uid, versions, env_steps stamped, one eliminated seat, a snapshot opponent
    MatchResult("w0_e3_ep7", "2p", "wdl",
                [_seat(0, version_start=40, version_end=41, source="owner"),
                 _seat(1, network="ckpt_v20", version_start=20, version_end=20, eliminated_step=5,
                       source="snapshots")],
                [TeamResult(0, 1.0, 1.0), TeamResult(1, 2.0, -1.0)], 12, uid="RUN/train/w0_e3_ep7", seed=123456,
                env_steps=BIG),
    # SP5 training record: node, generations, epoch in the match id
    MatchResult("w1_e0_g4096_ep2", "2p", "wdl",
                [_seat(0, version_start=BIG, version_end=BIG + 1, generation=3), _seat(1, network="fixed",
                                                                                       agent="random")],
                [TeamResult(0, 1.5, 0.0), TeamResult(1, 1.5, 0.0)], 30, uid="RUN/train/n2/w1_e0_g4096_ep2",
                seed=0, env_steps=0, node=2),
    # tournament block, FFA rank outcome with ties, a draw-like score
    MatchResult("t_A_0_7_r1", "ffa4", "rank", [_seat(i, network="fixed", agent=f"b{i}", reward=0.25) for i in range(4)],
                [TeamResult(i, [1.0, 2.5, 2.5, 4.0][i], float(i)) for i in range(4)], 50, uid="T1/A/0/7/r1",
                source="tournament", seed=2**31, block="A/0/7"),
    # pulse: a snapshot's env_steps, a solo score layout with one team
    MatchResult("p0", "solo", "score", [_seat(0, network="ckpt_v100", version_start=100, version_end=100)],
                [TeamResult(0, 1.0, 17.5)], 100, uid="RUN/pulse/x", source="pulse", env_steps=4000, block="P/3"),
]


def test_the_new_record_fields_default_to_none_and_come_last():
    assert Lineup("2p", [SeatAssignment("a")]).epoch is None
    assert SeatResult(0, "player", 0, "a", "latest", 1.0).generation is None
    assert MatchResult("m", "2p", "wdl", [], [], 1).node is None
    assert [f.name for f in dataclasses.fields(Lineup)][-1] == "epoch"
    assert [f.name for f in dataclasses.fields(SeatResult)][-1] == "generation"
    assert [f.name for f in dataclasses.fields(MatchResult)][-1] == "node"


@pytest.mark.parametrize("result", RESULTS, ids=lambda r: r.match_id)
def test_every_match_record_variant_round_trips_through_bytes(result):
    wire = result_to_proto(result).SerializeToString()
    back = result_from_proto(pb.MatchResultProto.FromString(wire))
    assert back == result
    assert dataclasses.asdict(back) == dataclasses.asdict(result)


def test_unset_optional_fields_stay_none_and_zero_stays_zero():
    proto = result_to_proto(RESULTS[0])
    for field in ("seed", "env_steps", "block", "node"):
        assert not proto.HasField(field)
    for field in ("eliminated_step", "version_start", "version_end", "generation"):
        assert not proto.seats[0].HasField(field)
    zero = result_to_proto(RESULTS[2])
    assert zero.HasField("seed") and zero.seed == 0 and zero.HasField("env_steps") and zero.env_steps == 0


def test_a_batch_of_256_results_round_trips_in_order():
    results = [dataclasses.replace(RESULTS[1], match_id=f"w0_e0_ep{i}", uid=f"RUN/train/w0_e0_ep{i}")
               for i in range(256)]
    wire = results_to_batch(7, results).SerializeToString()
    batch = pb.ResultBatch.FromString(wire)
    assert batch.seq == 7 and results_from_batch(batch) == results


LINEUPS = [
    Lineup("2p", [SeatAssignment("main"), SeatAssignment("main", "ckpt_v20", False, "snapshots")]),
    Lineup("2p", [SeatAssignment("a", source="owner"), SeatAssignment("random", "fixed", False, "anchors")],
           seed=99, block="A/1/2", uid="T/A/1/2/r0"),
    Lineup("2p", [SeatAssignment("a", source="owner"), SeatAssignment("b", source="rivals")], epoch=BIG),
    Lineup("2p", [SeatAssignment("a")], seed=0, epoch=0),
]


@pytest.mark.parametrize("lineup", LINEUPS)
def test_every_lineup_variant_round_trips_through_bytes(lineup):
    back = lineup_from_proto(pb.LineupProto.FromString(lineup_to_proto(lineup).SerializeToString()))
    assert back == lineup


def test_a_grant_carries_a_lineup_with_its_epoch_and_ttl():
    grant = pb.Grant(worker=1, env=3, epoch=1025, ttl_sec=30.0, lineup=lineup_to_proto(LINEUPS[2]))
    back = pb.HubMessage.FromString(pb.HubMessage(grants=pb.LineupGrants(grants=[grant])).SerializeToString())
    assert back.WhichOneof("m") == "grants"
    assert lineup_from_proto(back.grants.grants[0].lineup).epoch == BIG and back.grants.grants[0].ttl_sec == 30.0


def test_version_announcements_round_trip():
    info = VersionInfo("main", 3, BIG, 50 * 2**20, "ab" * 16, 2, ("10.0.0.5:47102", "[::1]:47102"), False, True)
    back = version_from_proto(pb.VersionInfo.FromString(version_to_proto(info).SerializeToString()))
    assert back == info and isinstance(back.addresses, tuple)


def test_values_a_field_cannot_hold_are_refused():
    with pytest.raises(ValueError):
        result_to_proto(dataclasses.replace(RESULTS[0], seed=2**64))


def test_the_protocol_version_rule():
    assert PROTOCOL_VERSION == (1, 0)
    assert protocol_compatible(1, 0) and protocol_compatible(1, 7)
    assert not protocol_compatible(2, 0) and not protocol_compatible(0, 9)


def test_the_contract_services_and_messages_exist():
    services = {s.name: [m.name for m in s.methods] for s in pb.DESCRIPTOR.services_by_name.values()}
    assert services == {
        "Bootstrap": ["Info", "Bundle"],
        "Hub": ["Register", "Session", "Upload", "Download", "RelaySend", "RelayInbox", "Admin", "AdminStream"],
        "NodeData": ["Chunks", "GetLatest"],
    }
    assert pb.DESCRIPTOR.package == "colosseum.v1"
    oneofs = {name: [f.name for f in pb.DESCRIPTOR.message_types_by_name[name].oneofs_by_name[o].fields]
              for name, o in (("NodeMessage", "m"), ("HubMessage", "m"))}
    assert oneofs == {
        "NodeMessage": ["hello", "heartbeat", "lineup_request", "results", "metrics", "version", "event",
                        "lease_request", "block_result"],
        "HubMessage": ["welcome", "grants", "ack", "command", "announce", "lease"],
    }
    int64_fields = [("VersionInfo", "version"), ("Hello", "env_steps"), ("Heartbeat", "env_steps"),
                    ("SeatResultProto", "version_start"), ("MatchResultProto", "env_steps"), ("Grant", "epoch"),
                    ("ResultBatch", "seq"), ("LearnerAssignment", "version_floor"), ("BlobPart", "total_size")]
    for message, field in int64_fields:
        assert pb.DESCRIPTOR.message_types_by_name[message].fields_by_name[field].type == 3, (message, field)  # INT64
    assert hasattr(pb_grpc, "HubStub") and hasattr(pb_grpc, "add_NodeDataServicer_to_server")


def test_the_committed_generated_code_matches_the_proto(tmp_path):
    """scripts/gen_proto.sh was run after the last .proto edit: compiling the .proto gives the committed descriptor."""
    pytest.importorskip("grpc_tools", reason="grpcio-tools is in the dev extra")
    from google.protobuf import descriptor_pb2

    src = PROTO.parents[3]                                                  # .../src
    out = tmp_path / "set.pb"
    subprocess.run([sys.executable, "-m", "grpc_tools.protoc", f"-I{src}", f"--descriptor_set_out={out}",
                    str(PROTO.relative_to(src))], check=True, cwd=src)
    compiled = descriptor_pb2.FileDescriptorSet.FromString(out.read_bytes()).file[0]
    committed = descriptor_pb2.FileDescriptorProto.FromString(pb.DESCRIPTOR.serialized_pb)

    def strip_json_names(messages) -> None:                               # protoc adds them to descriptor sets only
        for message in messages:
            for field in message.field:
                field.ClearField("json_name")
            strip_json_names(message.nested_type)

    strip_json_names(compiled.message_type)
    assert compiled == committed
```

- [ ] **Step 2: Run it (RED)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_messages.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.net'` (in the T1.2 worktree `colosseum.net` does not exist yet; with T1.1 integrated first: `No module named 'colosseum.net.messages'`).

- [ ] **Step 3: Add the three record fields**

In `src/colosseum/core/types.py` (append each field after the last existing field of its dataclass, so SP3/SP4 positional construction keeps working; add one docstring sentence each):
```python
@dataclass
class Lineup:
    ...                                  # layout, seats, seed, block, uid unchanged
    epoch: int | None = None             # SP5: the cell's epoch (hub training lineups, spec 4.2); None elsewhere


@dataclass
class SeatResult:
    ...                                  # SP4 fields unchanged
    generation: int | None = None        # SP5: learner generation of the seat's weights at the episode end (latest seats)


@dataclass
class MatchResult:
    ...                                  # SP4 fields unchanged
    node: int | None = None              # SP5: node number of the worker that played it (None: SP4 rows, local runs)
```
Docstring additions: `Lineup`: "``epoch`` (SP5): the cell epoch of a hub-granted training lineup; with an epoch the runner's match id is ``w<W>_e<E>_g<epoch>_ep<K>`` and the episode seed comes from the cell seed (T2.2)."; `SeatResult`: "``generation`` (SP5): the learner generation of the seat's weights at the episode end (``latest`` seats; the hub fences stale generations, spec 7.2)."; `MatchResult`: "``node`` (SP5): the node number of the worker that played it."

- [ ] **Step 4: Write the .proto and the generator script, generate**

Create `src/colosseum/net/proto/__init__.py` (empty) and `src/colosseum/net/proto/colosseum_v1.proto`:
```proto
// Colosseum wire protocol v1 (SP5 spec block 4.1, 4.9; plan 00-overview "colosseum.net.proto").
//
// Generated code is committed: scripts/gen_proto.sh (grpcio-tools, -I src) writes colosseum_v1_pb2.py and
// colosseum_v1_pb2_grpc.py next to this file. Versions, counters and sizes are int64; field numbers are never reused
// (a removed field is `reserved`); a change that old peers cannot read raises PROTOCOL_VERSION's major
// (colosseum.net.messages), which Bootstrap/Info and Hello compare.
//
// Large data (weights, snapshots, bundles, checkpoints) never travels on the Session stream: Upload / Download /
// GetLatest move it in BlobPart / UploadPart pieces of colosseum.net.grpc_util.PART_BYTES. Array payloads inside
// `bytes` fields are colosseum.net.frame frames, never pickle.
//
// /colosseum.v1.Bootstrap/Script is not declared here: it is a generic handler with identity serializers that returns
// the installer script's raw bytes (ruling R2), so `curl ... | tail -c +6` strips exactly the 5-byte gRPC prefix.

syntax = "proto3";

package colosseum.v1;

message Empty {}

// ---------------------------------------------------------------------------------------------------------------
// Bootstrap (no auth: Info; token secret: Bundle)
// ---------------------------------------------------------------------------------------------------------------

service Bootstrap {
  rpc Info(Empty) returns (BootstrapInfo);
  rpc Bundle(BundleRequest) returns (stream BlobPart);
}

message BootstrapInfo {
  uint32 protocol_major = 1;
  uint32 protocol_minor = 2;
  string code_hash = 3;
  string requirements_hash = 4;
  repeated string requirements = 5;
  string python = 6;              // "3.12"
  string torch_version = 7;       // "2.14.1"
  string torch_build = 8;         // "cu130" | "cpu" | ""
  int64 bundle_size = 9;
}

message BundleRequest { string code_hash = 1; }

// One piece of a blob. The first part carries total_size and meta_json (and not_modified for GetLatest); the last
// part carries the digest (blake2b-128 hex of the whole blob).
message BlobPart {
  bytes data = 1;
  int64 offset = 2;
  int64 total_size = 3;
  bool last = 4;
  string digest = 5;
  bool not_modified = 6;
  string meta_json = 7;
}

// ---------------------------------------------------------------------------------------------------------------
// Hub (token secret: Register, Admin*; node id + node secret: everything else)
// ---------------------------------------------------------------------------------------------------------------

service Hub {
  rpc Register(RegisterRequest) returns (RegisterReply);
  rpc Session(stream NodeMessage) returns (stream HubMessage);
  rpc Upload(stream UploadPart) returns (UploadReply);
  rpc Download(BlobRequest) returns (stream BlobPart);
  rpc RelaySend(stream ChunkFrame) returns (ChunkSummary);
  rpc RelayInbox(stream RelayCredit) returns (stream RelayItem);
  rpc Admin(AdminRequest) returns (AdminReply);
  rpc AdminStream(AdminRequest) returns (stream AdminReply);
}

message RegisterRequest {
  string public_key_pem = 1;
  string machine_id = 2;
  string name = 3;
  repeated string tags = 4;
  string node_id = 5;             // "" = a new node
}

message RegisterReply {
  string node_id = 1;
  int32 number = 2;
  string name = 3;
  string certificate_pem = 4;
  string node_secret = 5;
  string ca_pem = 6;
}

message NodeMessage {
  oneof m {
    Hello hello = 1;
    Heartbeat heartbeat = 2;
    LineupRequest lineup_request = 3;
    ResultBatch results = 4;
    MetricsBatch metrics = 5;
    VersionAnnounce version = 6;
    Event event = 7;
    BlockLeaseRequest lease_request = 8;
    BlockResult block_result = 9;
  }
}

message HubMessage {
  oneof m {
    Welcome welcome = 1;
    LineupGrants grants = 2;
    Ack ack = 3;
    Command command = 4;
    Announce announce = 5;
    BlockLease lease = 6;
  }
}

message Hello {
  string node_id = 1;
  string session_id = 2;          // new on every start of the node process
  uint32 protocol_major = 3;
  uint32 protocol_minor = 4;
  string code_hash = 5;
  Hardware hardware = 6;
  Slots slots = 7;
  repeated string data_addresses = 8;
  string run_id = 9;              // "" = no run yet
  repeated LiveCell live_cells = 10;
  int64 env_steps = 11;           // cumulative for this session
  int64 last_acked_seq = 12;
  repeated RunningLearner learners = 13;
  string name = 14;
  repeated string tags = 15;
}

message Hardware {
  int32 logical_cores = 1;
  int32 physical_cores = 2;
  int64 memory_bytes = 3;
  repeated Gpu gpus = 4;
  string os = 5;
  string platform = 6;
  string python = 7;
  string torch_version = 8;
  string torch_build = 9;
  int64 shm_free_bytes = 10;
  int64 cache_free_bytes = 11;
}

message Gpu {
  int32 index = 1;
  string name = 2;
  int64 memory_bytes = 3;
  string kind = 4;                // cuda | mps
}

message Slots {
  int32 workers = 1;
  int32 evaluators = 2;
  int32 cpu_learners = 3;
  repeated int32 gpus = 4;
  bool no_learners = 5;
  int32 envs_per_worker = 6;
}

message LiveCell {
  int32 worker = 1;
  int32 env = 2;
  int64 epoch = 3;
  int32 incarnation = 4;
}

message RunningLearner {
  string agent_id = 1;
  int64 generation = 2;
  int64 version = 3;
}

message Welcome {
  int32 node_number = 1;
  string run_id = 2;
  string config_json = 3;
  repeated AgentState agents = 4;
  repeated VersionInfo versions = 5;
  repeated Eviction evictions = 6;
  repeated LearnerAssignment learners = 7;
  int32 evaluators = 8;
  int64 env_steps = 9;
  repeated FixedBlob fixed = 10;
  bool collect_allowed = 11;
}

message AgentState {
  string agent_id = 1;
  string kind = 2;                // trainable | scripted | frozen
  string status = 3;              // active | pending | failed | frozen | removed
  int64 start_env_steps = 4;
}

message FixedBlob {
  string agent_id = 1;
  string blob_id = 2;
  string kind = 3;                // fixed | teacher | init
}

message Heartbeat {
  int64 env_steps = 1;            // cumulative for this session
  int64 dropped_results = 2;      // cumulative: node result buffer overflow
  repeated ProcState procs = 3;
  double load = 4;
  repeated string cached_keys = 5;
}

message ProcState {
  string name = 1;
  string state = 2;
  int32 restarts = 3;
}

message LineupRequest { repeated CellRequest cells = 1; }

message CellRequest {
  int32 worker = 1;
  int32 env = 2;
  int64 epoch = 3;                // 0 = none yet
  int32 incarnation = 4;
}

message LineupGrants { repeated Grant grants = 1; }

message Grant {
  int32 worker = 1;
  int32 env = 2;
  int64 epoch = 3;
  double ttl_sec = 4;             // 0 = the whole run
  LineupProto lineup = 5;
}

message LineupProto {
  string layout = 1;
  repeated SeatProto seats = 2;
  optional int64 seed = 3;
  optional string block = 4;
  optional string uid = 5;
  optional int64 epoch = 6;
}

message SeatProto {
  string agent_id = 1;
  string network_id = 2;
  bool collect = 3;
  string source = 4;
}

message ResultBatch {
  int64 seq = 1;                  // per session, from 1
  repeated MatchResultProto results = 2;
}

// The SP4 match record (SP4 spec block 1) plus the node and each seat's learner generation.
message MatchResultProto {
  string match_id = 1;
  string layout = 2;
  string outcome_kind = 3;
  repeated SeatResultProto seats = 4;
  repeated TeamResultProto teams = 5;
  int64 episode_length = 6;
  string uid = 7;
  string source = 8;
  optional int64 seed = 9;
  optional int64 env_steps = 10;
  optional string block = 11;
  optional int32 node = 12;
}

message SeatResultProto {
  int32 seat = 1;
  string role = 2;
  int32 team = 3;
  string agent_id = 4;
  string network_id = 5;
  double reward = 6;
  optional int64 eliminated_step = 7;
  string source = 8;
  optional int64 version_start = 9;
  optional int64 version_end = 10;
  optional int64 generation = 11;
}

message TeamResultProto {
  int32 team = 1;
  double rank = 2;
  double score = 3;
}

message MetricsBatch {
  repeated TrainMetrics train = 1;
  repeated WorkerStats workers = 2;
  repeated IntakeStats intake = 3;
}

message TrainMetrics {
  string agent_id = 1;
  int64 generation = 2;
  int64 train_step = 3;
  map<string, double> values = 4;
}

message WorkerStats {
  int32 worker = 1;
  int32 incarnation = 2;
  map<string, int64> counters = 3;
}

message IntakeStats {
  string agent_id = 1;
  int64 generation = 2;
  int32 depth = 3;
  map<string, int64> dropped = 4;
}

message Ack { repeated int64 seqs = 1; }

message VersionAnnounce { VersionInfo info = 1; }

message VersionInfo {
  string agent_id = 1;
  int64 generation = 2;
  int64 version = 3;
  int64 size = 4;
  string digest = 5;
  int32 node = 6;
  repeated string addresses = 7;
  bool relay_only = 8;
  bool teacher_active = 9;
}

message Event {
  string kind = 1;                // proc_died | child_error | slot_disabled | learner_failed | warning
  string proc = 2;
  int32 exit_code = 3;
  string error_kind = 4;
  string message = 5;
  string context = 6;
  bytes log_tail = 7;
  string log_path = 8;
}

message Command {
  string kind = 1;                // start_learner | stop_learner | set_evaluators | release | stop_run | update_code
  LearnerAssignment learner = 2;
  string reason = 3;
  string code_hash = 4;
  int32 evaluators = 5;
}

message LearnerAssignment {
  string agent_id = 1;
  int64 generation = 2;
  int64 version_floor = 3;
  string device = 4;
  string weights_blob = 5;
  string trainer_state_blob = 6;
  int64 env_steps = 7;
}

message Announce {
  repeated VersionInfo versions = 1;
  repeated Eviction evictions = 2;
  repeated AgentState agents = 3;
  int64 env_steps = 4;
  bool collect_allowed = 5;
}

message Eviction {
  string agent_id = 1;
  string checkpoint_id = 2;
}

message BlobRequest {
  string kind = 1;                // snapshot | fixed | teacher | init | recovery | trainer_state | latest
  string agent_id = 2;
  string ident = 3;
  int64 since_version = 4;
}

message UploadPart {
  oneof p {
    UploadHeader header = 1;
    bytes data = 2;
  }
}

message UploadHeader {
  string kind = 1;                // checkpoint | final | recovery | relay_latest
  string agent_id = 2;
  int64 generation = 3;
  int64 version = 4;
  int64 total_size = 5;
  string digest = 6;
  string request_id = 7;
}

message UploadReply {
  bool accepted = 1;
  string reason = 2;
  string checkpoint_id = 3;
}

message ChunkFrame {
  string agent_id = 1;
  int32 target_node = 2;
  int64 generation = 3;
  bytes frame = 4;                // a colosseum.net.frame "chunk" frame, forwarded undecoded
}

message ChunkSummary {
  int64 received = 1;
  int64 dropped = 2;
}

message RelayCredit {
  int32 frames = 1;
  string node_id = 2;
}

message RelayItem {
  oneof i {
    ChunkFrame chunk = 1;
    LatestRequest pull = 2;
  }
}

message LatestRequest {
  string agent_id = 1;
  int64 since_version = 2;
  string request_id = 3;
}

message BlockLeaseRequest { int32 free_slots = 1; }

message BlockLease {
  string lease_id = 1;
  double lease_sec = 2;
  repeated BlockSpec blocks = 3;
}

message BlockSpec {
  string block_id = 1;
  string source = 2;              // pulse | tournament
  repeated LineupProto lineups = 3;
  repeated PlayerRefProto players = 4;
  bool deterministic = 5;
}

message PlayerRefProto {
  string name = 1;
  string agent_id = 2;
  string network_id = 3;
  repeated string roles = 4;
  string kind = 5;                // neural | scripted
  string blob_id = 6;
  string networks_json = 7;
  string bot_class = 8;
  string bot_kwargs_json = 9;
}

message BlockResult {
  string lease_id = 1;
  string block_id = 2;
  repeated MatchResultProto results = 3;
  bool failed = 4;
  string error = 5;
}

message AdminRequest {
  string command = 1;
  string args_json = 2;
}

message AdminReply {
  int32 code = 1;
  string text = 2;
  string data_json = 3;
  bool done = 4;
}

// ---------------------------------------------------------------------------------------------------------------
// NodeData: a node's data port (mutual TLS with hub-issued node certificates)
// ---------------------------------------------------------------------------------------------------------------

service NodeData {
  rpc Chunks(stream ChunkFrame) returns (ChunkSummary);
  rpc GetLatest(LatestRequest) returns (stream BlobPart);   // first part: not_modified, or a weights frame's meta
}
```

Create `scripts/gen_proto.sh` (executable, `chmod +x`):
```bash
#!/usr/bin/env bash
# Regenerate src/colosseum/net/proto/colosseum_v1_pb2.py and colosseum_v1_pb2_grpc.py from colosseum_v1.proto
# (grpcio-tools, in the dev extra). Run from anywhere; commit the generated files with the .proto.
# The test tests/unit/test_sp5_messages.py::test_the_committed_generated_code_matches_the_proto fails until it is run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-.venv/bin/python}"
env -u PYTHONPATH "$PY" -m grpc_tools.protoc -I src --python_out=src --grpc_python_out=src \
    src/colosseum/net/proto/colosseum_v1.proto
echo "generated: src/colosseum/net/proto/colosseum_v1_pb2.py, colosseum_v1_pb2_grpc.py"
```
Run: `scripts/gen_proto.sh`, then `head -20 src/colosseum/net/proto/colosseum_v1_pb2.py` and `grep GRPC_GENERATED_VERSION src/colosseum/net/proto/colosseum_v1_pb2_grpc.py`.
Expected: `# Protobuf Python Version: 7.35.1` (gencode `7, 35, 1`) and `GRPC_GENERATED_VERSION = '1.84.0'` with the installed grpcio-tools 1.84.0 — the floors of Step 6. The `_pb2_grpc.py` imports `from colosseum.net.proto import colosseum_v1_pb2` (because of `-I src`).

- [ ] **Step 5: Write the converters**

Create `src/colosseum/net/messages.py`:
```python
"""Protocol v1 constants and the only Python <-> protobuf converters (SP5 spec block 4.9 "Управляющие сообщения").

Session messages are used as their generated classes (``colosseum.net.proto.colosseum_v1_pb2``) everywhere; only the
types that the SP3 / SP4 code keeps as dataclasses are converted here:
- ``MatchResult`` / ``SeatResult`` / ``TeamResult`` <-> ``MatchResultProto`` (every SP4 field of the match record, plus
  ``MatchResult.node`` and ``SeatResult.generation``); ``None`` <-> an unset ``optional`` field;
- ``Lineup`` / ``SeatAssignment`` <-> ``LineupProto`` (``seed`` / ``block`` / ``uid`` / ``epoch`` optional);
- :class:`VersionInfo` (the hub's version registry entry, spec 4.5) <-> ``VersionInfo``.

The converters copy values verbatim and never raise on a well-formed protobuf: a result from a node is checked by
the hub's intake (``colosseum.hub.state.ResultIntake.check``), not here. ``to_proto`` raises ValueError for values a
field cannot hold (an int beyond int64).

``PROTOCOL_VERSION`` = (major, minor): a peer with another major is refused at ``Bootstrap/Info`` and ``Hello``
(:func:`protocol_compatible`); a minor bump adds fields old peers ignore.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from colosseum.core.types import Lineup, MatchResult, SeatAssignment, SeatResult, TeamResult
from colosseum.net.proto import colosseum_v1_pb2 as pb

PROTOCOL_VERSION = (1, 0)


def protocol_compatible(major: int, minor: int) -> bool:
    """Whether a peer speaking ``major.minor`` can talk to this code (same major; any minor)."""
    return int(major) == PROTOCOL_VERSION[0]


@dataclass(frozen=True)
class VersionInfo:
    """One announced weights version of an agent (spec 4.5): where a node fetches it from. ``digest`` is the blake2b-128
    hex of the weights frame, ``node`` the learner's node number, ``addresses`` its data-port addresses (empty or
    ``relay_only``: through the hub)."""

    agent_id: str
    generation: int
    version: int
    size: int
    digest: str
    node: int
    addresses: tuple[str, ...] = ()
    relay_only: bool = False
    teacher_active: bool = False


def _opt(message, field: str):
    return getattr(message, field) if message.HasField(field) else None


def _set_opt(message, field: str, value) -> None:
    if value is not None:
        setattr(message, field, value)


def seat_result_to_proto(s: SeatResult) -> pb.SeatResultProto:
    p = pb.SeatResultProto(seat=s.seat, role=s.role, team=s.team, agent_id=s.agent_id, network_id=s.network_id,
                           reward=float(s.reward), source=s.source)
    _set_opt(p, "eliminated_step", s.eliminated_step)
    _set_opt(p, "version_start", s.version_start)
    _set_opt(p, "version_end", s.version_end)
    _set_opt(p, "generation", s.generation)
    return p


def seat_result_from_proto(p: pb.SeatResultProto) -> SeatResult:
    return SeatResult(seat=p.seat, role=p.role, team=p.team, agent_id=p.agent_id, network_id=p.network_id,
                      reward=p.reward, eliminated_step=_opt(p, "eliminated_step"), source=p.source,
                      version_start=_opt(p, "version_start"), version_end=_opt(p, "version_end"),
                      generation=_opt(p, "generation"))


def result_to_proto(r: MatchResult) -> pb.MatchResultProto:
    p = pb.MatchResultProto(match_id=r.match_id, layout=r.layout, outcome_kind=r.outcome_kind,
                            seats=[seat_result_to_proto(s) for s in r.seats],
                            teams=[pb.TeamResultProto(team=t.team, rank=float(t.rank), score=float(t.score))
                                   for t in r.teams],
                            episode_length=r.episode_length, uid=r.uid, source=r.source)
    _set_opt(p, "seed", r.seed)
    _set_opt(p, "env_steps", r.env_steps)
    _set_opt(p, "block", r.block)
    _set_opt(p, "node", r.node)
    return p


def result_from_proto(p: pb.MatchResultProto) -> MatchResult:
    return MatchResult(match_id=p.match_id, layout=p.layout, outcome_kind=p.outcome_kind,  # type: ignore[arg-type]
                       seats=[seat_result_from_proto(s) for s in p.seats],
                       teams=[TeamResult(team=t.team, rank=t.rank, score=t.score) for t in p.teams],
                       episode_length=p.episode_length, uid=p.uid, source=p.source, seed=_opt(p, "seed"),
                       env_steps=_opt(p, "env_steps"), block=_opt(p, "block"), node=_opt(p, "node"))


def results_to_batch(seq: int, results: Iterable[MatchResult]) -> pb.ResultBatch:
    return pb.ResultBatch(seq=int(seq), results=[result_to_proto(r) for r in results])


def results_from_batch(p: pb.ResultBatch) -> list[MatchResult]:
    return [result_from_proto(r) for r in p.results]


def lineup_to_proto(lineup: Lineup) -> pb.LineupProto:
    p = pb.LineupProto(layout=lineup.layout,
                       seats=[pb.SeatProto(agent_id=s.agent_id, network_id=s.network_id, collect=s.collect,
                                           source=s.source) for s in lineup.seats])
    _set_opt(p, "seed", lineup.seed)
    _set_opt(p, "block", lineup.block)
    _set_opt(p, "uid", lineup.uid)
    _set_opt(p, "epoch", lineup.epoch)
    return p


def lineup_from_proto(p: pb.LineupProto) -> Lineup:
    return Lineup(layout=p.layout,
                  seats=[SeatAssignment(agent_id=s.agent_id, network_id=s.network_id, collect=s.collect,
                                        source=s.source) for s in p.seats],
                  seed=_opt(p, "seed"), block=_opt(p, "block"), uid=_opt(p, "uid"), epoch=_opt(p, "epoch"))


def version_to_proto(v: VersionInfo) -> pb.VersionInfo:
    return pb.VersionInfo(agent_id=v.agent_id, generation=v.generation, version=v.version, size=v.size,
                          digest=v.digest, node=v.node, addresses=list(v.addresses), relay_only=v.relay_only,
                          teacher_active=v.teacher_active)


def version_from_proto(p: pb.VersionInfo) -> VersionInfo:
    return VersionInfo(agent_id=p.agent_id, generation=p.generation, version=p.version, size=p.size,
                       digest=p.digest, node=p.node, addresses=tuple(p.addresses), relay_only=p.relay_only,
                       teacher_active=p.teacher_active)
```

- [ ] **Step 6: Dependencies, extras, ruff excludes, setup script**

`pyproject.toml`:
- `[project].dependencies` gains `"grpcio>=1.84"` and `"protobuf>=7.35.1"` (the floors the generated files check: `GRPC_GENERATED_VERSION` and the gencode version; replace with what Step 4 printed if it differs).
- The `grpc` extra is removed; `grpcio-tools>=1.84` moves into the `dev` extra (`dev = ["pytest>=8.0", "pytest-timeout>=2.3", "ruff>=0.6", "grpcio-tools>=1.84"]`). Keep the comment above it, reworded: `# grpcio-tools regenerates colosseum/net/proto (scripts/gen_proto.sh); the old colosseum/transport/colosseum_pb2*.py (SP1, removed in T7.2) was generated by 1.78.`
- `[tool.ruff].extend-exclude` adds `"src/colosseum/net/proto/colosseum_v1_pb2.py"`, `"src/colosseum/net/proto/colosseum_v1_pb2_grpc.py"` (the old transport entries stay until T7.2).

`scripts/setup-dev.sh`: install `.[dev,examples]` instead of `.[grpc,dev,examples]` (both the `uv pip install` line and the echo line and the header comment's "with the grpc, dev and examples extras" → "with the dev and examples extras").

Then: `env -u PYTHONPATH .venv/bin/python -c "import grpc, google.protobuf; print(grpc.__version__, google.protobuf.__version__)"` — the shared venv already has grpcio 1.84.0 / protobuf 7.36.2 (no reinstall).

- [ ] **Step 7: Run the focused tests (GREEN)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_messages.py tests/unit/test_sp4_match_records.py tests/unit/test_sp4_sp3_compat.py tests/unit/test_sp4_matchlog.py tests/contract/test_sp4_result_flow.py -q -rw`
Expected: all pass (18 new), zero warnings.

- [ ] **Step 8: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings; ruff clean (the generated files are excluded).

- [ ] **Step 9: Commit and push**

```bash
git add src/colosseum/net/proto scripts/gen_proto.sh src/colosseum/net/messages.py src/colosseum/core/types.py \
        pyproject.toml scripts/setup-dev.sh tests/unit/test_sp5_messages.py
git commit -m "feat: SP5 protocol v1 (protobuf, generated code, converters), record fields, grpc as a runtime dependency"
git push
```
(If `src/colosseum/net/__init__.py` does not exist in this worktree because T1.1 is not integrated yet, create it with T1.1's content — identical text, the cherry-pick then has no conflict — and add it to this commit.)

---

### Task T1.3: Security — identities, token, certificates, node book, credentials, interceptors; identity-dir isolation

Spec 2.1 (identity per hub machine, token, node certificates, `--token-file` / `COLOSSEUM_TOKEN`), 2.2 (`--ban` by node id and machine id, `--rotate`), 3.2 (TLS, fixed server names), 3.3 (`--insecure`, loopback), 3.4 (threat model), 3.6 (`cryptography`); rulings R1 (application-level node authentication on the hub port, mutual TLS on node data ports) and R15 (the CA certificate is the hub's TLS server certificate). Criterion 7 is tested at the library level on real loopback gRPC servers with raw-bytes handlers on the contract's method paths. The CLI (`hub token`, `join`, `nodes remove --ban`) is T6.3; the Register servicer is T2.3.

**Files:**
- Create: `src/colosseum/net/security.py`, `src/colosseum/net/grpc_util.py` (T2.3 adds the blob-part helpers), `tests/net_kit.py`
- Modify: `tests/conftest.py` (`_isolate_colosseum_dirs`), `pyproject.toml` (`cryptography>=42` in `[project].dependencies`; `net_kit` in `known-first-party`)
- Test: `tests/unit/test_sp5_security.py`

**Interfaces:**
- Consumes: nothing of the repo but `colosseum.net` (package); grpc, cryptography, stdlib `ssl`.
- Produces: everything of the contract's `colosseum.net.security` / `colosseum.net.grpc_util` (T1.3 part) plus (contract notes 6–7) `NodeRecord`, `NodeBook.{records, get, reload, set_cert_serial}`, `register(..., persistent=True, run_id="")`, `HubIdentity.key_pem`, `NodeIdentity.{key_pem, cert_pem, registered, hub_addresses, forget}`, `save_registration(reply, *, hub_addresses=())`, `private_dir`, `write_private`, `new_secret`, `TOKEN_RE`, `NODE_ID_RE`, `NODE_NAME_RE`; `grpc_util.{KEEPALIVE_MS, OPEN_METHODS, TOKEN_METHODS, token_metadata, node_metadata, metadata_value, node_id_of, peer_common_name, DataPortInterceptor}`; `tests/net_kit.{hub_identity, node_identity, register, echo_handlers, loopback_server, channel, call, call_stream}`.

- [ ] **Step 1: Isolate identity dirs, caches and the weight cache in every test**

Add to `tests/conftest.py` (after `_isolate_filesystem`; `os` is imported already):
```python
@pytest.fixture(autouse=True)
def _isolate_colosseum_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SP5 (T1.3): the hub / node identity dirs (``XDG_DATA_HOME``), bundles, envs and node logs
    (``XDG_CACHE_HOME``) and the shared-memory weight cache (``COLOSSEUM_SHM_DIR``) of every test, and of every
    process it starts (children inherit the environment: ``cli_runner.child_env``), live under ``tmp_path``."""
    for var, sub in (("XDG_DATA_HOME", "xdg-data"), ("XDG_CACHE_HOME", "xdg-cache"), ("COLOSSEUM_SHM_DIR", "shm")):
        path = tmp_path / sub
        path.mkdir(exist_ok=True)
        monkeypatch.setenv(var, str(path))
```
(`_repo_root_stays_clean` is unaffected: the dirs are under `tmp_path`.)

- [ ] **Step 2: Write the test support module and the failing tests**

Create `tests/net_kit.py`:
```python
"""Test support for the SP5 wire layer (T1.3): hub / node identities under ``tmp_path``, registration without a hub
process, loopback gRPC servers on port 0 with echo handlers of the ``colosseum.v1`` method paths, and channels.

The servers bind ``127.0.0.1:0`` themselves (no bind-then-release race); ``stop_all`` in a ``finally`` (or the
``servers`` fixture of a test module) stops them."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

import grpc

from colosseum.net.grpc_util import channel_options, server_options
from colosseum.net.security import HUB_SERVER_NAME, HubIdentity, NodeIdentity, NodeRecord

MAX_MB = 4


def hub_identity(tmp_path: Path, name: str = "hub-identity") -> HubIdentity:
    return HubIdentity.load_or_create(Path(tmp_path) / name)


def node_identity(tmp_path: Path, ca_fingerprint: str, name: str = "node") -> NodeIdentity:
    return NodeIdentity.load_or_create(ca_fingerprint, root=Path(tmp_path) / f"{name}-identity")


def register(hub: HubIdentity, node: NodeIdentity, *, name: str = "laptop", machine_id: str = "machine-1",
             tags: Sequence[str] = ()) -> NodeRecord:
    """What the hub's Register handler does (T2.3), without a server: book the node, issue its certificate, and let
    the node keep the reply."""
    record, secret = hub.nodes.register(machine_id=machine_id, name=name, tags=tags, node_id=node.node_id)
    cert = hub.issue_node_certificate(record.node_id, node.public_key_pem())
    reply = SimpleNamespace(node_id=record.node_id, number=record.number, name=record.name,
                            certificate_pem=cert.decode("ascii"), node_secret=secret, ca_pem=hub.ca_pem.decode("ascii"))
    node.save_registration(reply)
    return hub.nodes.get(record.node_id)


def _echo(request: bytes, context) -> bytes:
    return b"ok:" + request


def _echo_stream(requests: Iterable[bytes], context) -> Iterable[bytes]:
    for request in requests:
        yield b"ok:" + request


def _count(requests: Iterable[bytes], context) -> bytes:
    return str(sum(1 for _ in requests)).encode()


def _two(request: bytes, context) -> Iterable[bytes]:
    yield b"part1"
    yield b"part2"


def echo_handlers() -> list[grpc.GenericRpcHandler]:
    """Raw-bytes handlers on the contract's method paths (one of each streaming kind)."""
    def h(kind, fn):
        return getattr(grpc, f"{kind}_rpc_method_handler")(fn)

    return [
        grpc.method_handlers_generic_handler("colosseum.v1.Bootstrap", {
            "Info": h("unary_unary", _echo), "Script": h("unary_unary", _echo), "Bundle": h("unary_stream", _two)}),
        grpc.method_handlers_generic_handler("colosseum.v1.Hub", {
            "Register": h("unary_unary", _echo), "Admin": h("unary_unary", _echo),
            "AdminStream": h("unary_stream", _two), "Session": h("stream_stream", _echo_stream),
            "Upload": h("stream_unary", _count), "Download": h("unary_stream", _two)}),
        grpc.method_handlers_generic_handler("colosseum.v1.NodeData", {
            "Chunks": h("stream_unary", _count), "GetLatest": h("unary_stream", _two)}),
    ]


def loopback_server(credentials: grpc.ServerCredentials | None = None, *,
                    interceptors: Sequence[grpc.ServerInterceptor] = (),
                    handlers: Sequence[grpc.GenericRpcHandler] | None = None) -> tuple[grpc.Server, int]:
    """A started server on ``127.0.0.1:0`` (plain without ``credentials``); returns ``(server, port)``."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), interceptors=list(interceptors),
                         options=server_options(MAX_MB))
    server.add_generic_rpc_handlers(list(handlers if handlers is not None else echo_handlers()))
    address = "127.0.0.1:0"
    if credentials is None:
        port = server.add_insecure_port(address)
    else:
        port = server.add_secure_port(address, credentials)
    server.start()
    return server, port


def channel(port: int, credentials: grpc.ChannelCredentials | None = None,
            server_name: str | None = HUB_SERVER_NAME) -> grpc.Channel:
    target = f"127.0.0.1:{port}"
    if credentials is None:
        return grpc.insecure_channel(target, options=channel_options(MAX_MB))
    return grpc.secure_channel(target, credentials, options=channel_options(MAX_MB, server_name=server_name))


def call(ch: grpc.Channel, method: str, payload: bytes = b"x", *, metadata=(), timeout: float = 5.0) -> bytes:
    """One unary call of a raw-bytes method (``/colosseum.v1.Hub/Register`` style)."""
    return ch.unary_unary(method)(payload, metadata=tuple(metadata), timeout=timeout)


def call_stream(ch: grpc.Channel, method: str, payloads: Sequence[bytes] = (b"a", b"b"), *, metadata=(),
                timeout: float = 5.0) -> list[bytes]:
    return list(ch.stream_stream(method)(iter(payloads), metadata=tuple(metadata), timeout=timeout))
```
Add `"net_kit"` to `known-first-party` in `pyproject.toml` and `"cryptography>=42"` to `[project].dependencies` (the controller installed cryptography 50.0.2 into the shared `.venv`; no install here).

Create `tests/unit/test_sp5_security.py`:
```python
"""Security of the SP5 wire layer (T1.3, spec blocks 2.1 and 3.2-3.4, rulings R1 and R15; criterion 7 at the library
level): the token, the hub and node identities with private file modes, the node book (numbers, names, secret
hashes, ban, remove), rotation, node certificates, and every refusal on a real loopback gRPC server: no token, a
wrong token, a wrong fingerprint, another CA, plain against TLS, --insecure still needing the token, node secrets,
bans, and mutual TLS on a node data port."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import stat

import grpc
import pytest
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from colosseum.net.grpc_util import AuthInterceptor, DataPortInterceptor, node_metadata, token_metadata
from colosseum.net.security import (
    HUB_SERVER_NAME,
    NODE_SERVER_NAME,
    TOKEN_ENV,
    HubIdentity,
    NodeIdentity,
    SecurityError,
    Token,
    cache_home,
    curl_pin,
    data_channel_credentials,
    data_home,
    data_server_credentials,
    fetch_hub_ca,
    hub_channel_credentials,
    hub_identity_dir,
    parse_token,
    resolve_token,
    spki_fingerprint,
)
from net_kit import call, call_stream, channel, hub_identity, loopback_server, node_identity, register


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def servers():
    started: list[grpc.Server] = []
    yield started
    for server in started:
        server.stop(None)


def _hub_server(hub: HubIdentity, servers, *, tls: bool = True) -> int:
    server, port = loopback_server(hub.server_credentials() if tls else None,
                                   interceptors=[AuthInterceptor(lambda: hub.token.secret, hub.nodes)])
    servers.append(server)
    return port


def _tls_channel(hub: HubIdentity, port: int) -> grpc.Channel:
    ca = fetch_hub_ca("127.0.0.1", port, fingerprint=hub.fingerprint, timeout=5.0)
    return channel(port, hub_channel_credentials(ca), HUB_SERVER_NAME)


def _code(fn, *args, **kwargs) -> grpc.StatusCode:
    with pytest.raises(grpc.RpcError) as info:
        fn(*args, **kwargs)
    return info.value.code()


# ---------------------------------------------------------------------------
# Directories, token, fingerprints
# ---------------------------------------------------------------------------


def test_identity_dirs_are_redirected_into_tmp_path(tmp_path):
    """tests/conftest.py's autouse fixture: no test touches the real ~/.local/share or ~/.cache."""
    assert data_home().is_relative_to(tmp_path) and cache_home().is_relative_to(tmp_path)
    assert hub_identity_dir() == data_home() / "colosseum" / "hub"
    assert os.environ["COLOSSEUM_SHM_DIR"].startswith(str(tmp_path))


def test_the_token_format_and_its_resolution(tmp_path):
    token = Token("A" * 43, "0f" * 32)
    assert str(token) == f"cst1.{'A' * 43}.{'0f' * 32}" and parse_token(str(token) + "\n") == token
    assert "A" * 43 not in repr(token)                                          # never logged by accident
    for bad in ("cst2." + "A" * 43 + "." + "0" * 64, "cst1." + "A" * 42 + "." + "0" * 64,
                "cst1." + "A" * 43 + "." + "0" * 63, "cst1." + "A" * 43 + "." + "G" * 64, ""):
        with pytest.raises(SecurityError, match="malformed token"):
            parse_token(bad)
    other = Token("B" * 43, "1a" * 32)
    path = tmp_path / "token.txt"
    path.write_text(f"{other}\n")
    env = {TOKEN_ENV: str(Token("C" * 43, "2b" * 32))}
    assert resolve_token(str(token), str(path), env=env) == token                # --token first
    assert resolve_token(None, str(path), env=env) == other                      # then --token-file
    assert resolve_token(None, None, env=env).secret == "C" * 43                 # then COLOSSEUM_TOKEN
    with pytest.raises(SecurityError, match="--token-file"):
        resolve_token(None, None, env={})
    with pytest.raises(SecurityError, match="cannot read --token-file"):
        resolve_token(None, str(tmp_path / "missing"), env={})


def test_the_hub_identity_is_created_once_with_private_modes(tmp_path):
    hub = hub_identity(tmp_path)
    assert _mode(hub.path) == 0o700
    assert _mode(hub.path / "ca.key") == 0o600 and _mode(hub.path / "secret") == 0o600
    again = HubIdentity.load_or_create(hub.path)
    assert again.fingerprint == hub.fingerprint and again.token == hub.token
    assert hub.fingerprint == spki_fingerprint((hub.path / "ca.crt").read_bytes()) and len(hub.fingerprint) == 64
    cert = x509.load_pem_x509_certificate(hub.ca_pem)
    assert cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    assert cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(
        x509.DNSName) == [HUB_SERVER_NAME]
    assert isinstance(cert.public_key(), ec.EllipticCurvePublicKey) and cert.public_key().curve.name == "secp256r1"
    (hub.path / "secret").unlink()
    with pytest.raises(SecurityError, match="incomplete"):
        HubIdentity.load_or_create(hub.path)


def test_the_curl_pin_is_the_base64_of_the_same_digest(tmp_path):
    hub = hub_identity(tmp_path)
    pin = curl_pin(hub.fingerprint)
    assert pin.startswith("sha256//") and base64.b64decode(pin.removeprefix("sha256//")).hex() == hub.fingerprint


# ---------------------------------------------------------------------------
# Node book, bans, rotation, node identities
# ---------------------------------------------------------------------------


def test_node_numbers_are_never_reused_names_are_unique_and_only_secret_hashes_are_kept(tmp_path):
    hub = hub_identity(tmp_path)
    a, secret_a = hub.nodes.register(machine_id="m1", name="laptop", tags=["cpu"], node_id=None)
    b, _ = hub.nodes.register(machine_id="m2", name="laptop", tags=[], node_id=None)
    assert (a.number, b.number) == (1, 2) and (a.name, b.name) == ("laptop", "laptop-2")
    assert len(a.node_id) == 32 and hub.nodes.verify(a.node_id, secret_a).name == "laptop"
    again, secret_again = hub.nodes.register(machine_id="m1", name="laptop", tags=["gpu"], node_id=a.node_id)
    assert again.number == 1 and again.tags == ["gpu"]
    assert hub.nodes.verify(a.node_id, secret_a) is None and hub.nodes.verify(a.node_id, secret_again)   # new secret
    hub.nodes.remove(b.node_id)
    c, _ = hub.nodes.register(machine_id="m3", name="pc", tags=[], node_id=None)
    assert c.number == 3                                                         # 2 is not reused
    text = (hub.path / "nodes.json").read_text()
    assert secret_again not in text and hashlib.sha256(secret_again.encode()).hexdigest() in text
    assert _mode(hub.path / "nodes.json") == 0o600 and json.loads(text)["next_number"] == 4
    assert [r.number for r in HubIdentity.load_or_create(hub.path).nodes.records()] == [1, 3]
    with pytest.raises(SecurityError, match="invalid node name"):
        hub.nodes.register(machine_id="m4", name="bad name", tags=[], node_id=None)


def test_a_ban_refuses_the_node_id_and_the_machine_id(tmp_path):
    hub = hub_identity(tmp_path)
    a, secret = hub.nodes.register(machine_id="m1", name="laptop", tags=[], node_id=None)
    hub.nodes.ban(a.node_id)
    assert hub.nodes.verify(a.node_id, secret) is None and hub.nodes.is_banned(None, "m1")
    with pytest.raises(SecurityError, match="banned"):
        hub.nodes.register(machine_id="m1", name="laptop", tags=[], node_id=None)          # same machine, new id
    with pytest.raises(SecurityError, match="banned"):
        hub.nodes.register(machine_id="m9", name="other", tags=[], node_id=a.node_id)      # same id
    assert hub.nodes.register(machine_id="m2", name="pc", tags=[], node_id=None)[0].number == 2
    assert json.loads((hub.path / "banned.json").read_text())["banned"][0]["machine_id"] == "m1"


def test_rotate_makes_a_new_ca_and_secret_and_revokes_every_node(tmp_path):
    hub = hub_identity(tmp_path)
    book = hub.nodes
    old_fp, old_token = hub.fingerprint, hub.token
    a, secret = hub.nodes.register(machine_id="m1", name="laptop", tags=[], node_id=None)
    hub.nodes.ban(hub.nodes.register(machine_id="m2", name="pc", tags=[], node_id=None)[0].node_id)
    hub.rotate()
    assert hub.fingerprint != old_fp and hub.token.secret != old_token.secret and hub.nodes is book
    assert hub.nodes.records() == [] and hub.nodes.verify(a.node_id, secret) is None
    assert not hub.nodes.is_banned(None, "m2")                                  # the ban list starts over too
    assert spki_fingerprint((hub.path / "rotated" / "1" / "ca.crt").read_bytes()) == old_fp
    assert HubIdentity.load_or_create(hub.path).fingerprint == hub.fingerprint


def test_a_node_identity_keeps_its_key_and_registration(tmp_path):
    hub = hub_identity(tmp_path)
    node = node_identity(tmp_path, hub.fingerprint)
    assert _mode(node.path) == 0o700 and _mode(node.path / "node.key") == 0o600 and not node.registered
    record = register(hub, node, name="laptop")
    again = NodeIdentity.load_or_create(hub.fingerprint, root=tmp_path / "node-identity")
    assert (again.node_id, again.number, again.name, again.secret) == (record.node_id, 1, "laptop", node.secret)
    assert again.registered and _mode(again.path / "node.json") == 0o600
    assert again.public_key_pem() == node.public_key_pem()
    cert = x509.load_pem_x509_certificate(again.cert_pem)
    assert cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == record.node_id
    assert cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value.get_values_for_type(
        x509.DNSName) == [NODE_SERVER_NAME]
    cert.verify_directly_issued_by(x509.load_pem_x509_certificate(hub.ca_pem))
    assert hub.nodes.get(record.node_id).cert_serial == format(cert.serial_number, "x")
    again.forget()
    assert not NodeIdentity.load_or_create(hub.fingerprint, root=tmp_path / "node-identity").registered
    with pytest.raises(SecurityError, match="invalid node id"):
        hub.issue_node_certificate("x", node.public_key_pem())
    with pytest.raises(SecurityError, match="PEM"):
        hub.issue_node_certificate("0" * 32, b"not a key")


# ---------------------------------------------------------------------------
# The hub port on a real loopback server (criterion 7)
# ---------------------------------------------------------------------------


def test_the_tls_hub_port_accepts_the_pinned_ca_and_the_token(tmp_path, servers):
    hub = hub_identity(tmp_path)
    port = _hub_server(hub, servers)
    ch = _tls_channel(hub, port)
    assert call(ch, "/colosseum.v1.Bootstrap/Info") == b"ok:x"                    # open
    assert call(ch, "/colosseum.v1.Hub/Register", metadata=token_metadata(hub.token)) == b"ok:x"
    assert call(ch, "/colosseum.v1.Hub/Admin", metadata=token_metadata(hub.token)) == b"ok:x"
    assert list(ch.unary_stream("/colosseum.v1.Bootstrap/Bundle")(b"", metadata=token_metadata(hub.token),
                                                                    timeout=5)) == [b"part1", b"part2"]


def test_calls_without_the_token_or_with_a_wrong_one_are_refused(tmp_path, servers):
    hub = hub_identity(tmp_path)
    port = _hub_server(hub, servers)
    ch = _tls_channel(hub, port)
    wrong = Token("Z" * 43, hub.fingerprint)
    bundle = ch.unary_stream("/colosseum.v1.Bootstrap/Bundle")
    for metadata in ((), token_metadata(wrong)):
        for method in ("/colosseum.v1.Hub/Register", "/colosseum.v1.Hub/Admin"):
            assert _code(call, ch, method, metadata=metadata) == grpc.StatusCode.UNAUTHENTICATED
        assert _code(lambda md: list(bundle(b"", metadata=md, timeout=5)), metadata) == grpc.StatusCode.UNAUTHENTICATED
    with pytest.raises(grpc.RpcError) as info:
        call(ch, "/colosseum.v1.Hub/Register")
    assert "colosseum hub token" in info.value.details()


def test_a_wrong_fingerprint_is_refused_before_any_call(tmp_path, servers):
    hub, other = hub_identity(tmp_path), hub_identity(tmp_path, "other-hub")
    port = _hub_server(hub, servers)
    with pytest.raises(SecurityError, match="fingerprint"):
        fetch_hub_ca("127.0.0.1", port, fingerprint=other.fingerprint, timeout=5.0)


def test_a_client_trusting_another_ca_or_no_tls_cannot_connect(tmp_path, servers):
    hub, other = hub_identity(tmp_path), hub_identity(tmp_path, "other-hub")
    port = _hub_server(hub, servers)
    foreign = channel(port, hub_channel_credentials(other.ca_pem), HUB_SERVER_NAME)
    assert _code(lambda: call(foreign, "/colosseum.v1.Bootstrap/Info", timeout=3)) == grpc.StatusCode.UNAVAILABLE
    plain = channel(port)                                                        # no TLS against the TLS port
    assert _code(lambda: call(plain, "/colosseum.v1.Bootstrap/Info", timeout=3)) == grpc.StatusCode.UNAVAILABLE
    ca = fetch_hub_ca("127.0.0.1", port, fingerprint=hub.fingerprint, timeout=5.0)
    unnamed = channel(port, hub_channel_credentials(ca), server_name=None)       # 127.0.0.1 is not colosseum-hub
    assert _code(lambda: call(unnamed, "/colosseum.v1.Bootstrap/Info", timeout=3)) == grpc.StatusCode.UNAVAILABLE


def test_insecure_mode_still_requires_the_token_and_the_node_secret(tmp_path, servers):
    hub = hub_identity(tmp_path)
    port = _hub_server(hub, servers, tls=False)
    ch = channel(port)
    assert _code(lambda: call(ch, "/colosseum.v1.Hub/Register")) == grpc.StatusCode.UNAUTHENTICATED
    assert call(ch, "/colosseum.v1.Hub/Register", metadata=token_metadata(hub.token)) == b"ok:x"
    assert _code(lambda: call_stream(ch, "/colosseum.v1.Hub/Session")) == grpc.StatusCode.UNAUTHENTICATED


def test_node_calls_need_a_registered_not_banned_node_and_its_secret(tmp_path, servers):
    hub = hub_identity(tmp_path)
    node = node_identity(tmp_path, hub.fingerprint)
    record = register(hub, node)
    port = _hub_server(hub, servers)
    ch = _tls_channel(hub, port)
    good = node_metadata(record.node_id, node.secret)
    assert call_stream(ch, "/colosseum.v1.Hub/Session", metadata=good) == [b"ok:a", b"ok:b"]
    assert list(ch.unary_stream("/colosseum.v1.Hub/Download")(b"", metadata=good, timeout=5)) == [b"part1", b"part2"]
    assert ch.stream_unary("/colosseum.v1.Hub/Upload")(iter([b"1", b"2", b"3"]), metadata=good, timeout=5) == b"3"
    for metadata in ((), token_metadata(hub.token), node_metadata(record.node_id, "wrong"),
                     node_metadata("f" * 32, node.secret)):
        assert _code(call_stream, ch, "/colosseum.v1.Hub/Session", metadata=metadata) == grpc.StatusCode.UNAUTHENTICATED
    hub.nodes.ban(record.node_id)
    assert _code(lambda: call_stream(ch, "/colosseum.v1.Hub/Session", metadata=good)) \
        == grpc.StatusCode.PERMISSION_DENIED


def test_rotation_revokes_registered_nodes_and_the_old_token_at_once(tmp_path, servers):
    hub = hub_identity(tmp_path)
    node = node_identity(tmp_path, hub.fingerprint)
    record = register(hub, node)
    port = _hub_server(hub, servers, tls=False)                                 # the CA change needs a restart
    ch = channel(port)
    old_token = hub.token
    hub.rotate()
    assert _code(lambda: call_stream(ch, "/colosseum.v1.Hub/Session",
                                     metadata=node_metadata(record.node_id, node.secret))) \
        == grpc.StatusCode.UNAUTHENTICATED
    assert _code(lambda: call(ch, "/colosseum.v1.Hub/Register", metadata=token_metadata(old_token))) \
        == grpc.StatusCode.UNAUTHENTICATED
    assert call(ch, "/colosseum.v1.Hub/Register", metadata=token_metadata(hub.token)) == b"ok:x"


# ---------------------------------------------------------------------------
# A node data port (mutual TLS)
# ---------------------------------------------------------------------------


def _data_server(hub: HubIdentity, owner: NodeIdentity, servers) -> int:
    allowed = lambda cn: hub.nodes.get(cn) is not None and not hub.nodes.is_banned(cn, None)  # noqa: E731
    server, port = loopback_server(data_server_credentials(owner, hub.ca_pem),
                                   interceptors=[DataPortInterceptor(allowed)])
    servers.append(server)
    return port


def test_a_data_port_takes_only_node_certificates_of_the_run(tmp_path, servers):
    hub, other = hub_identity(tmp_path), hub_identity(tmp_path, "other-hub")
    a, b = node_identity(tmp_path, hub.fingerprint, "a"), node_identity(tmp_path, hub.fingerprint, "b")
    register(hub, a, name="a", machine_id="ma")
    record_b = register(hub, b, name="b", machine_id="mb")
    stranger = node_identity(tmp_path, other.fingerprint, "stranger")
    register(other, stranger, name="s", machine_id="ms")
    port = _data_server(hub, a, servers)
    good = channel(port, data_channel_credentials(b, hub.ca_pem), NODE_SERVER_NAME)
    assert good.stream_unary("/colosseum.v1.NodeData/Chunks")(iter([b"f1", b"f2"]), timeout=5) == b"2"
    assert list(good.unary_stream("/colosseum.v1.NodeData/GetLatest")(b"", timeout=5)) == [b"part1", b"part2"]
    no_cert = channel(port, hub_channel_credentials(hub.ca_pem), NODE_SERVER_NAME)
    assert _code(lambda: no_cert.stream_unary("/colosseum.v1.NodeData/Chunks")(iter([b"f"]), timeout=3)) \
        in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNKNOWN)
    foreign = channel(port, grpc.ssl_channel_credentials(root_certificates=hub.ca_pem,
                                                         private_key=stranger.key_pem,
                                                         certificate_chain=stranger.cert_pem), NODE_SERVER_NAME)
    assert _code(lambda: foreign.stream_unary("/colosseum.v1.NodeData/Chunks")(iter([b"f"]), timeout=3)) \
        in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNKNOWN)
    assert _code(lambda: call(good, "/colosseum.v1.Hub/Register", metadata=token_metadata(hub.token))) \
        == grpc.StatusCode.PERMISSION_DENIED                                    # a data port serves NodeData only
    hub.nodes.ban(record_b.node_id)
    assert _code(lambda: good.stream_unary("/colosseum.v1.NodeData/Chunks")(iter([b"f"]), timeout=5)) \
        == grpc.StatusCode.PERMISSION_DENIED


def test_node_data_is_never_served_on_the_hub_port(tmp_path, servers):
    hub = hub_identity(tmp_path)
    node = node_identity(tmp_path, hub.fingerprint)
    record = register(hub, node)
    ch = _tls_channel(hub, _hub_server(hub, servers))
    assert _code(lambda: ch.stream_unary("/colosseum.v1.NodeData/Chunks")(
        iter([b"f"]), metadata=node_metadata(record.node_id, node.secret), timeout=5)) \
        == grpc.StatusCode.PERMISSION_DENIED
```

- [ ] **Step 3: Run them (RED)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_security.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.net'` (or `colosseum.net.grpc_util` once T1.1 / T1.2 are integrated; create `src/colosseum/net/__init__.py` with T1.1's text if it is missing in this worktree).

- [ ] **Step 4: Write `colosseum.net.security`**

Create `src/colosseum/net/security.py`:
```python
"""Identities, tokens, certificates and node authentication (SP5 spec blocks 2.1, 3.2-3.4; rulings R1, R15).

Trust is per hub machine, not per run (spec 2.1): ``<data_home>/colosseum/hub/`` (``0700``) holds the hub's root of
trust — an ECDSA P-256 key ``ca.key`` (``0600``) and its self-signed certificate ``ca.crt`` (CA, SAN
``colosseum-hub``), which is also the hub's TLS server certificate (R15: a node fetches it unverified with
``ssl.get_server_certificate``, checks its SPKI fingerprint against the token and then uses it as the gRPC root; the
curl one-liner pins the same key) — the token secret ``secret`` (``0600``), the node book ``nodes.json`` and the
ban list ``banned.json``. ``rotate()`` moves all of it to ``rotated/<n>/`` and starts a new CA and secret: every
node is revoked.

The token ``cst1.<secret>.<fp>`` = 32 random bytes base64url without padding + the lowercase hex SHA-256 of the CA
certificate's SubjectPublicKeyInfo (DER). It is the owner's full access (registration and admin commands).

Node authentication (R1): grpc-python cannot ask for an optional client certificate, and the hub's one port must also
serve clients without one (the installer, the first registration), so the hub port is server-side TLS only and every
hub RPC after registration carries the node id and a per-node secret in metadata (``x-colosseum-node``,
``x-colosseum-node-secret``); the hub keeps only the secret's SHA-256 (``nodes.json``). Registration and admin RPCs
carry the token secret (``x-colosseum-token``). Node data ports (node <-> node) are full mutual TLS with node
certificates issued by the CA (CN = node id, SAN ``colosseum-node``). ``ban`` and ``rotate`` revoke both kinds.
``--insecure`` drops TLS but keeps the token and node secrets; loopback (``train`` without ``--listen``) is plain.

A node's identity per hub CA lives in ``<data_home>/colosseum/node/<ca fingerprint>/`` (``0700``): ``node.key``
(``0600``), ``node.crt``, ``ca.crt`` (the pinned CA) and ``node.json`` (``0600``: node id, number, name, secret,
hub addresses).

Every private key and secret is written ``0600`` (created with that mode, then replaced atomically); every
identity directory is ``0700``. Refusals raise :class:`SecurityError` with a hint (exit 1 at the CLI).
"""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import ssl
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import grpc
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

TOKEN_PREFIX = "cst1"
TOKEN_ENV = "COLOSSEUM_TOKEN"
HUB_SERVER_NAME = "colosseum-hub"
NODE_SERVER_NAME = "colosseum-node"
MD_TOKEN, MD_NODE_ID, MD_NODE_SECRET = "x-colosseum-token", "x-colosseum-node", "x-colosseum-node-secret"
TOKEN_RE = re.compile(r"^cst1\.([A-Za-z0-9_-]{43})\.([0-9a-f]{64})$")
NODE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
NODE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
BOOK_VERSION = 1
CERT_BACKDATE = datetime.timedelta(days=7)          # tolerate machine clocks a few days apart
CA_LIFETIME = datetime.timedelta(days=20 * 365)
NODE_LIFETIME = datetime.timedelta(days=10 * 365)


class SecurityError(Exception):
    """A refusal or a broken identity (exit 1 at the CLI); the message says what to do."""


# ---------------------------------------------------------------------------
# Directories and files
# ---------------------------------------------------------------------------


def data_home() -> Path:
    """``$XDG_DATA_HOME`` or ``~/.local/share``."""
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")


def cache_home() -> Path:
    """``$XDG_CACHE_HOME`` or ``~/.cache``."""
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")


def hub_identity_dir() -> Path:
    return data_home() / "colosseum" / "hub"


def node_identity_dir(ca_fingerprint: str, root: Path | None = None) -> Path:
    """The node identity for the hub CA ``ca_fingerprint``; ``root`` replaces ``<data_home>/colosseum/node``."""
    return (Path(root) if root is not None else data_home() / "colosseum" / "node") / ca_fingerprint


def private_dir(path: Path) -> Path:
    """``path`` (and missing parents) as a directory with mode ``0700``."""
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, 0o700)
    return path


def write_private(path: Path, data: bytes | str) -> Path:
    """Atomically (re)write ``path`` with mode ``0600`` (the temp file is created ``0600``)."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        raise SecurityError(f"{path} is unreadable ({e}); move it away to start over") from None
    if not isinstance(data, dict):
        raise SecurityError(f"{path} is not a JSON object; move it away to start over")
    return data


# ---------------------------------------------------------------------------
# Token and fingerprints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    secret: str
    ca_fingerprint: str

    def __str__(self) -> str:
        return f"{TOKEN_PREFIX}.{self.secret}.{self.ca_fingerprint}"

    def __repr__(self) -> str:                      # never print the secret by accident
        return f"Token(fingerprint={self.ca_fingerprint[:8]}...)"


def new_secret() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")


def parse_token(text: str) -> Token:
    m = TOKEN_RE.match(text.strip())
    if m is None:
        raise SecurityError("malformed token: expected cst1.<43 base64url chars>.<64 hex chars> (print it with "
                            "`colosseum hub token` on the hub machine)")
    return Token(m.group(1), m.group(2))


def resolve_token(token: str | None, token_file: str | None, *, env: Mapping[str, str] = os.environ) -> Token:
    """The token from ``--token`` > ``--token-file`` > ``$COLOSSEUM_TOKEN``."""
    if token:
        return parse_token(token)
    if token_file:
        try:
            return parse_token(Path(token_file).read_text())
        except OSError as e:
            raise SecurityError(f"cannot read --token-file {token_file}: {e}") from None
    if env.get(TOKEN_ENV):
        return parse_token(env[TOKEN_ENV])
    raise SecurityError(f"no token: pass --token, --token-file or set {TOKEN_ENV} (print it with `colosseum hub "
                        f"token` on the hub machine)")


def spki_fingerprint(cert_pem: bytes) -> str:
    """Lowercase hex SHA-256 of the certificate's SubjectPublicKeyInfo (DER)."""
    cert = x509.load_pem_x509_certificate(cert_pem)
    spki = cert.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return hashlib.sha256(spki).hexdigest()


def curl_pin(fingerprint: str) -> str:
    """``sha256//<base64>``: the same SPKI digest for ``curl --pinnedpubkey``."""
    return "sha256//" + base64.b64encode(bytes.fromhex(fingerprint)).decode("ascii")


def _key_pem(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption())


def _load_key(path: Path) -> ec.EllipticCurvePrivateKey:
    try:
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    except (OSError, ValueError, TypeError) as e:
        raise SecurityError(f"{path} is not a readable private key ({e})") from None
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise SecurityError(f"{path} is not an ECDSA key")
    return key


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _make_ca(key: ec.EllipticCurvePrivateKey) -> x509.Certificate:
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, HUB_SERVER_NAME)])
    now = _now()
    usage = x509.KeyUsage(digital_signature=True, key_cert_sign=True, crl_sign=True, content_commitment=False,
                          key_encipherment=False, data_encipherment=False, key_agreement=False, encipher_only=False,
                          decipher_only=False)
    return (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - CERT_BACKDATE)
            .not_valid_after(now + CA_LIFETIME)
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(usage, critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(HUB_SERVER_NAME)]), critical=False)
            .sign(key, hashes.SHA256()))


# ---------------------------------------------------------------------------
# The node book (nodes.json + banned.json)
# ---------------------------------------------------------------------------


@dataclass
class NodeRecord:
    node_id: str
    number: int
    name: str
    machine_id: str
    tags: list[str] = field(default_factory=list)
    secret_sha256: str = ""
    cert_serial: str = ""
    persistent: bool = True
    registered_run: str = ""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class NodeBook:
    """Registered and banned nodes of one hub identity (``nodes.json``, ``banned.json``; both ``0600``).

    ``nodes.json``: ``{"version": 1, "next_number", "nodes": {node_id: NodeRecord fields}}`` (node ids are 32 hex
    chars chosen here; numbers >= 1 are never reused; names are unique, ``-2``, ``-3``, ... on a collision);
    ``banned.json``: ``{"version": 1, "banned": [{"node_id", "machine_id", "name"}]}``. Thread-safe (servicer
    threads verify while the main loop registers)."""

    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        """Re-read both files (after ``HubIdentity.rotate``; the object stays the one interceptors hold)."""
        data = _read_json(self._dir / "nodes.json") or {"version": BOOK_VERSION, "next_number": 1, "nodes": {}}
        banned = _read_json(self._dir / "banned.json") or {"version": BOOK_VERSION, "banned": []}
        if data.get("version") != BOOK_VERSION or banned.get("version") != BOOK_VERSION:
            raise SecurityError(f"{self._dir}: node book of another version; move nodes.json / banned.json away")
        try:
            next_number = int(data["next_number"])
            nodes = {nid: NodeRecord(**rec) for nid, rec in data["nodes"].items()}
            banned_list = [dict(entry) for entry in banned["banned"]]
        except (KeyError, TypeError, ValueError) as e:
            raise SecurityError(f"{self._dir}/nodes.json is malformed ({e}); move it away to start over") from None
        with self._lock:
            self._next, self._nodes, self._banned = next_number, nodes, banned_list

    def _save(self) -> None:
        nodes = {"version": BOOK_VERSION, "next_number": self._next,
                 "nodes": {nid: asdict(rec) for nid, rec in self._nodes.items()}}
        write_private(self._dir / "nodes.json", json.dumps(nodes, indent=2, sort_keys=True))
        write_private(self._dir / "banned.json",
                      json.dumps({"version": BOOK_VERSION, "banned": self._banned}, indent=2, sort_keys=True))

    def records(self) -> list[NodeRecord]:
        with self._lock:
            return sorted((NodeRecord(**asdict(r)) for r in self._nodes.values()), key=lambda r: r.number)

    def get(self, node_id: str) -> NodeRecord | None:
        with self._lock:
            rec = self._nodes.get(node_id)
            return None if rec is None else NodeRecord(**asdict(rec))

    def is_banned(self, node_id: str | None, machine_id: str | None) -> bool:
        with self._lock:
            return any((node_id and e.get("node_id") == node_id) or (machine_id and e.get("machine_id") == machine_id)
                       for e in self._banned)

    def _unique_name(self, wanted: str, node_id: str) -> str:
        taken = {r.name for nid, r in self._nodes.items() if nid != node_id}
        if wanted not in taken:
            return wanted
        k = 2
        while f"{wanted}-{k}" in taken:
            k += 1
        return f"{wanted}-{k}"

    def register(self, *, machine_id: str, name: str, tags: Sequence[str], node_id: str | None,
                 persistent: bool = True, run_id: str = "") -> tuple[NodeRecord, str]:
        """Register a new node, or re-register a known ``node_id`` (same number; the name stays unless it was asked
        for anew); returns ``(record, new secret)``. A banned node id or machine id is a SecurityError."""
        if not NODE_NAME_RE.match(name):
            raise SecurityError(f"invalid node name {name!r} (letters, digits, '_', '.', '-'; at most 63 chars)")
        with self._lock:
            if any((node_id and e.get("node_id") == node_id) or (machine_id and e.get("machine_id") == machine_id)
                   for e in self._banned):
                raise SecurityError(f"node {name!r} (machine {machine_id[:12]}) is banned on this hub; "
                                    f"`colosseum hub token --rotate` revokes every node and clears the ban list")
            secret = new_secret()
            rec = self._nodes.get(node_id) if node_id else None
            if rec is None:
                node_id = secrets.token_hex(16)
                rec = NodeRecord(node_id=node_id, number=self._next, name=self._unique_name(name, node_id),
                                 machine_id=machine_id)
                self._next += 1
                self._nodes[node_id] = rec
            elif rec.name != name:
                rec.name = self._unique_name(name, rec.node_id)
            rec.machine_id, rec.tags = machine_id, list(tags)
            rec.secret_sha256, rec.persistent, rec.registered_run = _sha256(secret), bool(persistent), run_id
            self._save()
            return NodeRecord(**asdict(rec)), secret

    def set_cert_serial(self, node_id: str, serial: str) -> None:
        with self._lock:
            self._nodes[node_id].cert_serial = serial
            self._save()

    def verify(self, node_id: str, secret: str) -> NodeRecord | None:
        """The node's record if ``secret`` is its current secret and it is not banned; else None."""
        with self._lock:
            rec = self._nodes.get(node_id)
            if rec is None or not rec.secret_sha256:
                return None
            if not hmac.compare_digest(rec.secret_sha256, _sha256(secret)):
                return None
            if any(e.get("node_id") == node_id or (rec.machine_id and e.get("machine_id") == rec.machine_id)
                   for e in self._banned):
                return None
            return NodeRecord(**asdict(rec))

    def ban(self, node_id: str) -> None:
        """Forget the node and refuse its node id and machine id from now on."""
        with self._lock:
            rec = self._nodes.pop(node_id, None)
            entry = {"node_id": node_id, "machine_id": rec.machine_id if rec else "", "name": rec.name if rec else ""}
            self._banned.append(entry)
            self._save()

    def remove(self, node_id: str) -> None:
        """Forget the node (it may register again with the token; its number is not reused)."""
        with self._lock:
            if self._nodes.pop(node_id, None) is not None:
                self._save()


# ---------------------------------------------------------------------------
# The hub identity
# ---------------------------------------------------------------------------


class HubIdentity:
    """The hub machine's CA, token secret and node book (module docstring). ``load_or_create`` makes them once."""

    def __init__(self, path: Path, key: ec.EllipticCurvePrivateKey, ca_pem: bytes, secret: str) -> None:
        self.path = path
        self._key = key
        self.ca_pem = ca_pem
        self.fingerprint = spki_fingerprint(ca_pem)
        self.token = Token(secret, self.fingerprint)
        self.nodes = NodeBook(path)

    @classmethod
    def load_or_create(cls, path: Path | None = None) -> HubIdentity:
        path = private_dir(Path(path) if path is not None else hub_identity_dir())
        key_path, ca_path, secret_path = path / "ca.key", path / "ca.crt", path / "secret"
        present = [p.exists() for p in (key_path, ca_path, secret_path)]
        if any(present) and not all(present):
            missing = [p.name for p, ok in zip((key_path, ca_path, secret_path), present) if not ok]
            raise SecurityError(f"hub identity {path} is incomplete (missing {missing}); restore it, or move the "
                                f"directory away to create a new one (every node must then join again)")
        if not any(present):
            key = ec.generate_private_key(ec.SECP256R1())
            write_private(key_path, _key_pem(key))
            ca_path.write_bytes(_make_ca(key).public_bytes(serialization.Encoding.PEM))
            write_private(secret_path, new_secret() + "\n")
        key = _load_key(key_path)
        ca_pem = ca_path.read_bytes()
        secret = secret_path.read_text().strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{43}", secret):
            raise SecurityError(f"{secret_path} is malformed; `colosseum hub token --rotate` makes a new one")
        cert = x509.load_pem_x509_certificate(ca_pem)
        if cert.public_key().public_numbers() != key.public_key().public_numbers():
            raise SecurityError(f"{ca_path} does not belong to {key_path}; restore the pair or rotate")
        return cls(path, key, ca_pem, secret)

    @property
    def key_pem(self) -> bytes:
        return _key_pem(self._key)

    def rotate(self) -> None:
        """A new CA and secret; the old identity (and with it every node and ban) moves to ``rotated/<n>/``."""
        rotated = private_dir(self.path / "rotated")
        n = 1 + max((int(p.name) for p in rotated.iterdir() if p.name.isdigit()), default=0)
        target = private_dir(rotated / str(n))
        for name in ("ca.key", "ca.crt", "secret", "nodes.json", "banned.json"):
            if (self.path / name).exists():
                shutil.move(str(self.path / name), str(target / name))
        fresh = HubIdentity.load_or_create(self.path)
        self._key, self.ca_pem, self.fingerprint, self.token = fresh._key, fresh.ca_pem, fresh.fingerprint, fresh.token
        self.nodes.reload()                       # the same object: interceptors holding it see the empty book

    def issue_node_certificate(self, node_id: str, public_key_pem: bytes) -> bytes:
        """A client + server certificate for the node's key (CN = node id, SAN ``colosseum-node``), signed by the CA."""
        if not NODE_ID_RE.match(node_id):
            raise SecurityError(f"invalid node id {node_id!r}")
        try:
            public = serialization.load_pem_public_key(public_key_pem)
        except (ValueError, TypeError) as e:
            raise SecurityError(f"the node's public key is not a PEM key ({e})") from None
        if not isinstance(public, ec.EllipticCurvePublicKey):
            raise SecurityError("the node's key must be an ECDSA key")
        ca = x509.load_pem_x509_certificate(self.ca_pem)
        now = _now()
        cert = (x509.CertificateBuilder()
                .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node_id)]))
                .issuer_name(ca.subject).public_key(public).serial_number(x509.random_serial_number())
                .not_valid_before(now - CERT_BACKDATE).not_valid_after(now + NODE_LIFETIME)
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(x509.SubjectAlternativeName([x509.DNSName(NODE_SERVER_NAME)]), critical=False)
                .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH,
                                                      ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False)
                .sign(self._key, hashes.SHA256()))
        if self.nodes.get(node_id) is not None:
            self.nodes.set_cert_serial(node_id, format(cert.serial_number, "x"))
        return cert.public_bytes(serialization.Encoding.PEM)

    def server_credentials(self) -> grpc.ServerCredentials:
        """TLS for the hub port with the CA certificate as the server certificate; no client certificates (R1)."""
        return grpc.ssl_server_credentials([(self.key_pem, self.ca_pem)], require_client_auth=False)


# ---------------------------------------------------------------------------
# The node identity
# ---------------------------------------------------------------------------


class NodeIdentity:
    """A node's key, certificate and registration for one hub CA (module docstring)."""

    def __init__(self, path: Path, key: ec.EllipticCurvePrivateKey, info: dict) -> None:
        self.path = path
        self._key = key
        self.node_id: str | None = info.get("node_id")
        self.number: int | None = info.get("number")
        self.name: str | None = info.get("name")
        self.secret: str | None = info.get("secret")
        self.hub_addresses: list[str] = list(info.get("hub_addresses", []))

    @classmethod
    def load_or_create(cls, ca_fingerprint: str, root: Path | None = None) -> NodeIdentity:
        if not re.fullmatch(r"[0-9a-f]{64}", ca_fingerprint):
            raise SecurityError(f"invalid CA fingerprint {ca_fingerprint!r}")
        path = private_dir(node_identity_dir(ca_fingerprint, root))
        key_path = path / "node.key"
        if not key_path.exists():
            write_private(key_path, _key_pem(ec.generate_private_key(ec.SECP256R1())))
        info = _read_json(path / "node.json") or {}
        return cls(path, _load_key(key_path), info)

    def public_key_pem(self) -> bytes:
        return self._key.public_key().public_bytes(serialization.Encoding.PEM,
                                                   serialization.PublicFormat.SubjectPublicKeyInfo)

    @property
    def key_pem(self) -> bytes:
        return _key_pem(self._key)

    @property
    def cert_pem(self) -> bytes | None:
        path = self.path / "node.crt"
        return path.read_bytes() if path.exists() else None

    @property
    def registered(self) -> bool:
        return bool(self.node_id and self.secret and self.cert_pem)

    def save_registration(self, reply: Any, *, hub_addresses: Sequence[str] = ()) -> None:
        """Keep a ``RegisterReply`` (node id, number, name, certificate, secret, CA) for the next connections."""
        self.node_id, self.number, self.name, self.secret = reply.node_id, int(reply.number), reply.name, \
            reply.node_secret
        self.hub_addresses = list(dict.fromkeys([*hub_addresses, *self.hub_addresses]))
        (self.path / "node.crt").write_bytes(reply.certificate_pem.encode("ascii"))
        (self.path / "ca.crt").write_bytes(reply.ca_pem.encode("ascii"))
        write_private(self.path / "node.json", json.dumps({
            "node_id": self.node_id, "number": self.number, "name": self.name, "secret": self.secret,
            "hub_addresses": self.hub_addresses}, indent=2))

    def forget(self) -> None:
        """Drop the registration (after a ban or a rotation); the key stays."""
        for name in ("node.json", "node.crt", "ca.crt"):
            (self.path / name).unlink(missing_ok=True)
        self.node_id = self.number = self.name = self.secret = None


# ---------------------------------------------------------------------------
# Fetching the hub CA and gRPC credentials
# ---------------------------------------------------------------------------


def fetch_hub_ca(host: str, port: int, *, fingerprint: str, timeout: float) -> bytes:
    """The hub's certificate (= its CA, R15) from an unverified TLS handshake, accepted only if its SPKI fingerprint
    is the token's. SecurityError otherwise (a wrong token, another hub, or interception)."""
    try:
        pem = ssl.get_server_certificate((host, int(port)), timeout=timeout).encode("ascii")
    except (OSError, ssl.SSLError, TimeoutError) as e:
        raise SecurityError(f"cannot fetch the hub certificate from {host}:{port} ({type(e).__name__}: {e}); is the "
                            f"hub running with TLS on that address? (a hub started with --insecure needs --insecure "
                            f"here)") from None
    actual = spki_fingerprint(pem)
    if not hmac.compare_digest(actual, fingerprint):
        raise SecurityError(f"the certificate at {host}:{port} has fingerprint {actual[:16]}..., the token names "
                            f"{fingerprint[:16]}...: a wrong token, another hub on that address, or someone in "
                            f"between; check the token with `colosseum hub token` on the hub machine")
    return pem


def hub_channel_credentials(ca_pem: bytes) -> grpc.ChannelCredentials:
    """TLS to the hub port, trusting only the pinned CA (use with ``grpc_util.channel_options(server_name=
    HUB_SERVER_NAME)``)."""
    return grpc.ssl_channel_credentials(root_certificates=ca_pem)


def data_server_credentials(node: NodeIdentity, ca_pem: bytes) -> grpc.ServerCredentials:
    """Mutual TLS for a node's data port: the node's certificate, clients must present a certificate of the CA."""
    if node.cert_pem is None:
        raise SecurityError("the node has no certificate yet: register with the hub first")
    return grpc.ssl_server_credentials([(node.key_pem, node.cert_pem)], root_certificates=ca_pem,
                                       require_client_auth=True)


def data_channel_credentials(node: NodeIdentity, ca_pem: bytes) -> grpc.ChannelCredentials:
    """Mutual TLS to another node's data port (with ``channel_options(server_name=NODE_SERVER_NAME)``)."""
    if node.cert_pem is None:
        raise SecurityError("the node has no certificate yet: register with the hub first")
    return grpc.ssl_channel_credentials(root_certificates=ca_pem, private_key=node.key_pem,
                                        certificate_chain=node.cert_pem)
```

- [ ] **Step 5: Write `colosseum.net.grpc_util` (options, metadata, interceptors)**

Create `src/colosseum/net/grpc_util.py`:
```python
"""gRPC helpers of the SP5 wire layer: channel / server options, auth metadata and the server-side auth interceptor
(ruling R1). T2.3 adds blob streaming in parts (``PART_BYTES``, ``send_parts``, ``receive_parts``).

Which RPC needs what (:class:`AuthInterceptor`; refusals are UNAUTHENTICATED / PERMISSION_DENIED with a hint):
- open: ``Bootstrap/Info`` and ``Bootstrap/Script`` (the installer: version check and script, no secrets);
- the token secret (``x-colosseum-token``): ``Bootstrap/Bundle`` (the project code), ``Hub/Register``,
  ``Hub/Admin``, ``Hub/AdminStream``;
- a node id + its node secret (``x-colosseum-node``, ``x-colosseum-node-secret``), not banned: every other ``Hub``
  method (deny by default);
- ``NodeData`` (a node's data port, mutual TLS, :class:`DataPortInterceptor`): the client certificate's CN must be a
  node id the port accepts (the node agent's current set of the run's nodes, from the hub; T3.2).
Services outside ``colosseum.v1`` are passed through untouched; ``NodeData`` on the hub port is refused.
"""

from __future__ import annotations

import hmac
import logging
from collections.abc import Callable
from typing import Any

import grpc

from colosseum.net.security import MD_NODE_ID, MD_NODE_SECRET, MD_TOKEN, NodeBook, Token

logger = logging.getLogger(__name__)

KEEPALIVE_MS = 10_000                    # spec 4.1: a dead peer is noticed within ~10 s
KEEPALIVE_TIMEOUT_MS = 5_000
SERVICE_PREFIX = "/colosseum.v1."
OPEN_METHODS = frozenset({"/colosseum.v1.Bootstrap/Info", "/colosseum.v1.Bootstrap/Script"})
TOKEN_METHODS = frozenset({"/colosseum.v1.Bootstrap/Bundle", "/colosseum.v1.Hub/Register", "/colosseum.v1.Hub/Admin",
                           "/colosseum.v1.Hub/AdminStream"})
DATA_SERVICE = "/colosseum.v1.NodeData/"


def _sizes(max_message_mb: int) -> list[tuple[str, Any]]:
    limit = int(max_message_mb) * 2**20
    return [("grpc.max_send_message_length", limit), ("grpc.max_receive_message_length", limit)]


def channel_options(max_message_mb: int, *, server_name: str | None = None) -> list[tuple[str, Any]]:
    """Client options: message sizes, keepalive pings every 10 s (also without calls), the fixed TLS server name."""
    options = _sizes(max_message_mb) + [
        ("grpc.keepalive_time_ms", KEEPALIVE_MS), ("grpc.keepalive_timeout_ms", KEEPALIVE_TIMEOUT_MS),
        ("grpc.keepalive_permit_without_calls", 1), ("grpc.http2.max_pings_without_data", 0),
    ]
    if server_name is not None:
        options.append(("grpc.ssl_target_name_override", server_name))
    return options


def server_options(max_message_mb: int) -> list[tuple[str, Any]]:
    """Server options: message sizes, and clients may ping every 10 s without calls."""
    return _sizes(max_message_mb) + [
        ("grpc.keepalive_time_ms", KEEPALIVE_MS), ("grpc.keepalive_timeout_ms", KEEPALIVE_TIMEOUT_MS),
        ("grpc.keepalive_permit_without_calls", 1), ("grpc.http2.min_ping_interval_without_data_ms", 5_000),
        ("grpc.http2.max_ping_strikes", 0),
    ]


def token_metadata(token: Token) -> tuple[tuple[str, str], ...]:
    return ((MD_TOKEN, token.secret),)


def node_metadata(node_id: str, node_secret: str) -> tuple[tuple[str, str], ...]:
    return ((MD_NODE_ID, node_id), (MD_NODE_SECRET, node_secret))


def metadata_value(metadata: Any, key: str) -> str | None:
    """The value of ``key`` in invocation metadata (a sequence of (key, value) pairs), or None."""
    for item in metadata or ():
        k, v = (item.key, item.value) if hasattr(item, "key") else item
        if k == key:
            return v if isinstance(v, str) else None
    return None


def node_id_of(context: grpc.ServicerContext) -> str | None:
    """The node id an authenticated hub RPC was made with (the interceptor checked it)."""
    return metadata_value(context.invocation_metadata(), MD_NODE_ID)


def peer_common_name(context: grpc.ServicerContext) -> str | None:
    """The CN of the client certificate of a mutual-TLS call, or None."""
    names = context.auth_context().get("x509_common_name") or []
    return names[0].decode("utf-8", "replace") if names else None


def _refusing(handler: grpc.RpcMethodHandler, code: grpc.StatusCode, message: str) -> grpc.RpcMethodHandler:
    """A handler of the same kind as ``handler`` that aborts every call with ``code``."""

    def abort(request_or_iterator: Any, context: grpc.ServicerContext) -> Any:
        context.abort(code, message)

    def abort_stream(request_or_iterator: Any, context: grpc.ServicerContext) -> Any:
        context.abort(code, message)
        yield  # pragma: no cover - abort raises

    kwargs = {"request_deserializer": handler.request_deserializer, "response_serializer": handler.response_serializer}
    if handler.request_streaming and handler.response_streaming:
        return grpc.stream_stream_rpc_method_handler(abort_stream, **kwargs)
    if handler.request_streaming:
        return grpc.stream_unary_rpc_method_handler(abort, **kwargs)
    if handler.response_streaming:
        return grpc.unary_stream_rpc_method_handler(abort_stream, **kwargs)
    return grpc.unary_unary_rpc_method_handler(abort, **kwargs)


def _checking_peer(handler: grpc.RpcMethodHandler, allow: Callable[[grpc.ServicerContext], str | None]
                   ) -> grpc.RpcMethodHandler:
    """``handler`` with a check of the caller's TLS identity first (``allow`` returns a refusal message or None)."""
    kwargs = {"request_deserializer": handler.request_deserializer, "response_serializer": handler.response_serializer}

    def guard(context: grpc.ServicerContext) -> None:
        refusal = allow(context)
        if refusal is not None:
            context.abort(grpc.StatusCode.PERMISSION_DENIED, refusal)

    if handler.request_streaming and handler.response_streaming:
        def ss(it, ctx):
            guard(ctx)
            yield from handler.stream_stream(it, ctx)
        return grpc.stream_stream_rpc_method_handler(ss, **kwargs)
    if handler.request_streaming:
        def su(it, ctx):
            guard(ctx)
            return handler.stream_unary(it, ctx)
        return grpc.stream_unary_rpc_method_handler(su, **kwargs)
    if handler.response_streaming:
        def us(req, ctx):
            guard(ctx)
            yield from handler.unary_stream(req, ctx)
        return grpc.unary_stream_rpc_method_handler(us, **kwargs)

    def uu(req, ctx):
        guard(ctx)
        return handler.unary_unary(req, ctx)
    return grpc.unary_unary_rpc_method_handler(uu, **kwargs)


class AuthInterceptor(grpc.ServerInterceptor):
    """R1 authentication of every ``colosseum.v1`` RPC (module docstring). ``token_secret`` is read on every call
    (a rotated secret applies at once); ``nodes`` is the hub identity's node book."""

    def __init__(self, token_secret: Callable[[], str], nodes: NodeBook) -> None:
        self._token_secret = token_secret
        self._nodes = nodes

    def intercept_service(self, continuation: Callable, handler_call_details: grpc.HandlerCallDetails) -> Any:
        method = handler_call_details.method
        handler = continuation(handler_call_details)
        if handler is None or not method.startswith(SERVICE_PREFIX) or method in OPEN_METHODS:
            return handler
        metadata = handler_call_details.invocation_metadata
        if method.startswith(DATA_SERVICE):
            return _refusing(handler, grpc.StatusCode.PERMISSION_DENIED, "NodeData is served on node data ports only")
        if method in TOKEN_METHODS:
            secret = metadata_value(metadata, MD_TOKEN)
            if secret is None or not hmac.compare_digest(secret.encode(), self._token_secret().encode()):
                return _refusing(handler, grpc.StatusCode.UNAUTHENTICATED,
                                 "a valid token is required (x-colosseum-token); print it with `colosseum hub token` "
                                 "on the hub machine")
            return handler
        node_id, node_secret = metadata_value(metadata, MD_NODE_ID), metadata_value(metadata, MD_NODE_SECRET)
        if node_id is None or node_secret is None:
            return _refusing(handler, grpc.StatusCode.UNAUTHENTICATED,
                             "this call needs a registered node (x-colosseum-node / x-colosseum-node-secret)")
        if self._nodes.is_banned(node_id, None):
            return _refusing(handler, grpc.StatusCode.PERMISSION_DENIED, "this node is banned on this hub")
        if self._nodes.verify(node_id, node_secret) is None:
            return _refusing(handler, grpc.StatusCode.UNAUTHENTICATED,
                             "unknown node or wrong node secret: register again with the token (the hub's identity "
                             "may have been rotated)")
        return handler


class DataPortInterceptor(grpc.ServerInterceptor):
    """A node data port (mutual TLS): ``NodeData`` calls only from a client certificate whose CN ``allowed(cn)`` accepts
    (a registered, not banned node of the run); every other ``colosseum.v1`` method is refused."""

    def __init__(self, allowed: Callable[[str], bool]) -> None:
        self._allowed = allowed

    def intercept_service(self, continuation: Callable, handler_call_details: grpc.HandlerCallDetails) -> Any:
        method = handler_call_details.method
        handler = continuation(handler_call_details)
        if handler is None or not method.startswith(SERVICE_PREFIX):
            return handler
        if not method.startswith(DATA_SERVICE):
            return _refusing(handler, grpc.StatusCode.PERMISSION_DENIED, "a node data port serves NodeData only")
        return _checking_peer(handler, self._refusal)

    def _refusal(self, context: grpc.ServicerContext) -> str | None:
        name = peer_common_name(context)
        if name is None:
            return "a node certificate of this hub is required"
        if not self._allowed(name):
            return f"node {name[:12]} is not a node of this run (removed or banned)"
        return None
```

- [ ] **Step 6: Run the focused tests (GREEN)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_security.py -q -rw`
Expected: 17 passed in about 1 s, zero warnings (grpc's C core may print an INFO "Got goaway" line to stderr when a test server stops — that is not a Python warning). Then a test of the fixture itself: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_security.py::test_identity_dirs_are_redirected_into_tmp_path -q` passes, and `ls ~/.local/share/colosseum 2>/dev/null` shows nothing new.

- [ ] **Step 7: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings; ruff clean.

- [ ] **Step 8: Commit and push**

```bash
git add src/colosseum/net/security.py src/colosseum/net/grpc_util.py tests/net_kit.py tests/conftest.py pyproject.toml \
        tests/unit/test_sp5_security.py
git commit -m "feat: SP5 security (hub and node identities, token, node certificates, node book, credentials, interceptors)"
git push
```

---

### Task T1.4: Config — the `distributed` section, learner keys, the `transport` translation, the benchmark config

Spec block 9.1 (every key and default verbatim), 4.10 (`transport.mode` / `grpc_port` accepted with a WARNING and ignored, `grpc_max_message_mb` → `distributed.max_message_mb`, old + new = `ConfigError`), block 1 (`learner.device` `auto` / `cpu` / `cuda[:N]`; `mps` from 2.6), 3.1 (`port` ≠ `data_port`). `TransportConfig`, `TransportMode` and `ColosseumConfig.transport` are removed; `distributed.py` (until T7.2) reads `config.distributed.max_message_mb`.

**Files:**
- Modify: `src/colosseum/core/config.py`, `src/colosseum/distributed.py` (two lines), `scripts/bench_throughput.py` (`_make_config`), `tests/unit/test_bench_throughput.py` (guard)
- Test: `tests/unit/test_sp5_config.py`

**Interfaces:**
- Consumes: `StrictModel`, `apply_overrides` / `LEGACY_OVERRIDE_KEYS`, `get_agent_config` (learner overrides deep-merged).
- Produces: `DistributedBundleConfig`, `DistributedConfig`, `LearnerPlacementConfig`, `LearnerConfig.{max_policy_lag, recovery_interval_sec, placement}`, `ColosseumConfig.distributed`; *`TRANSPORT_IGNORED_KEYS`, *`TRANSPORT_RENAMED_KEYS`, *`translate_transport(data) -> dict` (contract note 8).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_sp5_config.py`:
```python
"""SP5 config (T1.4, spec blocks 9.1 and 4.10): the distributed section's defaults and checks (global only), the new
learner keys with their per-agent override, learner.device values, the transport translation (one WARNING; old +
new key = ConfigError), --set, and resolved configs of SP2-SP4 runs."""
from __future__ import annotations

import json
import logging

import pytest
import yaml

from colosseum.core.config import (
    ColosseumConfig,
    DistributedBundleConfig,
    DistributedConfig,
    LearnerPlacementConfig,
    load_config,
)
from colosseum.core.errors import ConfigError
from game_helpers import SP3_RUN

SPEC_DEFAULTS = {
    "port": 47100, "data_port": 47102, "insecure": False, "heartbeat_sec": 1.0, "node_lost_after_sec": 10.0,
    "learner_relocate_after_sec": 120.0, "hub_lost_timeout_sec": 1800.0, "hub_lost_collect_sec": 300.0,
    "shutdown_grace_sec": 20.0, "connect_timeout_sec": 5.0, "block_lease_sec": 600.0, "relay": True,
    "results_batch": 256, "results_flush_sec": 0.25, "node_result_buffer_mb": 256.0, "max_message_mb": 64,
    "bundle": {"include": [], "exclude": [], "max_file_mb": 10.0}, "requirements": None,
}


def _data(**sections) -> dict:
    return {"env": {"env_class": "game_helpers.TurnTakingGame"},
            "networks": {"model_class": "game_helpers.GameTestModel"}, **sections}


def _load(tmp_path, data: dict, overrides: dict | None = None) -> ColosseumConfig:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_config(path, overrides)


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and r.name.startswith("colosseum")]


def test_the_distributed_defaults_are_the_specs():
    cfg = ColosseumConfig.model_validate(_data())
    assert cfg.distributed.model_dump() == SPEC_DEFAULTS
    assert cfg.distributed.bundle == DistributedBundleConfig()
    assert not hasattr(cfg, "transport") and "transport" not in ColosseumConfig.model_fields


@pytest.mark.parametrize("bad, match", [
    ({"data_port": 47100}, "must differ"),
    ({"port": 0}, "port"),
    ({"data_port": 70000}, "data_port"),
    ({"heartbeat_sec": 0}, "heartbeat_sec"),
    ({"node_lost_after_sec": 1.0}, "larger than distributed.heartbeat_sec"),
    ({"learner_relocate_after_sec": 5.0}, "at least distributed.node_lost_after_sec"),
    ({"results_batch": 0}, "results_batch"),
    ({"max_message_mb": 0}, "max_message_mb"),
    ({"bundle": {"max_file_mb": 0}}, "max_file_mb"),
    ({"hub_port": 1}, "hub_port"),
])
def test_the_distributed_section_is_checked(bad, match):
    with pytest.raises(ValueError, match=match):
        ColosseumConfig.model_validate(_data(distributed=bad))


def test_the_distributed_section_is_global_only_and_set_reaches_it(tmp_path):
    cfg = _load(tmp_path, _data(), {"distributed.port": 50000, "distributed.bundle.exclude": ["data/"],
                                    "distributed.relay": False})
    assert (cfg.distributed.port, cfg.distributed.bundle.exclude, cfg.distributed.relay) == (50000, ["data/"], False)
    with pytest.raises(ConfigError):
        _load(tmp_path, _data(agents={"a": {}}), {"agents.a.distributed.port": 1})
    with pytest.raises(ConfigError, match="Unknown config key 'distributed.nonsense'"):
        _load(tmp_path, _data(), {"distributed.nonsense": 1})


def test_the_new_learner_keys_have_defaults_and_a_per_agent_override(tmp_path):
    cfg = _load(tmp_path, _data(agents={"a": {}, "b": {"learner": {"device": "cpu", "max_policy_lag": 4,
                                                                    "placement": {"node": "laptop"}}}}))
    a, b = cfg.get_agent_config("a").learner, cfg.get_agent_config("b").learner
    assert (a.max_policy_lag, a.recovery_interval_sec, a.placement) == (None, 300.0, LearnerPlacementConfig())
    assert a.placement.model_dump() == {"node": None, "tags": [], "strict": False}
    assert (b.device, b.max_policy_lag, b.placement.node, b.placement.strict) == ("cpu", 4, "laptop", False)
    over = _load(tmp_path, _data(agents={"a": {}}), {"agents.a.learner.placement.tags": ["gpu", "big"],
                                                     "agents.a.learner.placement.strict": True,
                                                     "learner.recovery_interval_sec": 60})
    learner = over.get_agent_config("a").learner
    assert (learner.placement.tags, learner.placement.strict, learner.recovery_interval_sec) == (["gpu", "big"], True,
                                                                                                 60.0)
    for bad in ({"max_policy_lag": -1}, {"recovery_interval_sec": 0}, {"placement": {"host": "x"}}):
        with pytest.raises(ValueError):
            ColosseumConfig.model_validate(_data(learner=bad))


@pytest.mark.parametrize("device, ok", [("auto", True), ("cpu", True), ("cuda", True), ("cuda:1", True),
                                         ("mps", True), ("gpu", False), ("cuda:x", False), ("CPU", False)])
def test_learner_device_values(device, ok):
    if ok:
        assert ColosseumConfig.model_validate(_data(learner={"device": device})).learner.device == device
    else:
        with pytest.raises(ValueError, match="auto, cpu, cuda, cuda:N or mps"):
            ColosseumConfig.model_validate(_data(learner={"device": device}))


def test_the_transport_section_is_translated_with_one_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        cfg = _load(tmp_path, _data(transport={"mode": "grpc", "grpc_port": 50051, "grpc_max_message_mb": 128}))
    assert cfg.distributed.max_message_mb == 128 and cfg.distributed.port == 47100
    (warning,) = _warnings(caplog)
    assert "transport.mode, transport.grpc_port ignored" in warning
    assert "transport.grpc_max_message_mb read as distributed.max_message_mb: 128" in warning
    assert "transport" not in cfg.model_dump() and "transport" not in json.dumps(cfg.model_dump(mode="json"))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        only_mode = _load(tmp_path, _data(transport={"mode": "local"}))
    assert only_mode.distributed.max_message_mb == 64 and len(_warnings(caplog)) == 1
    assert "grpc_max_message_mb" not in _warnings(caplog)[0]


def test_old_and_new_message_size_together_are_an_error(tmp_path):
    with pytest.raises(ConfigError, match="keep only distributed.max_message_mb"):
        _load(tmp_path, _data(transport={"grpc_max_message_mb": 32}, distributed={"max_message_mb": 16}))
    with pytest.raises(ConfigError, match="unknown keys"):
        _load(tmp_path, _data(transport={"grpc_host": "x"}))


def test_set_still_accepts_the_old_transport_keys(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        cfg = _load(tmp_path, _data(), {"transport.grpc_max_message_mb": 32, "transport.mode": "local"})
    assert cfg.distributed.max_message_mb == 32 and len(_warnings(caplog)) == 1


def test_a_resolved_config_of_an_sp3_run_loads_and_is_written_back_without_transport(tmp_path, caplog):
    """SP3 / SP4 run dirs hold config.resolved.yaml with a transport section: it loads (one WARNING) and the new
    resolved config has a distributed section instead."""
    with caplog.at_level(logging.WARNING):
        cfg = load_config(SP3_RUN / "config.resolved.yaml")
    assert len(_warnings(caplog)) == 1 and cfg.distributed.max_message_mb == 64
    dumped = cfg.model_dump(mode="json", by_alias=True)
    assert "transport" not in dumped and dumped["distributed"] == SPEC_DEFAULTS
    assert ColosseumConfig.model_validate(dumped).distributed == DistributedConfig()
```

Extend `tests/unit/test_bench_throughput.py::test_benchmark_config_sets_every_section_and_every_sp4_key_explicitly`: rename it `test_benchmark_config_sets_every_section_and_every_sp4_sp5_key_explicitly` and make its loop
```python
    for name in ("match_log", "ratings", "evaluation", "checkpoint", "distributed", "learner"):
        section = getattr(cfg, name)
        assert section.model_fields_set == set(type(section).model_fields), name
    assert cfg.distributed.bundle.model_fields_set == set(type(cfg.distributed.bundle).model_fields)
    assert cfg.learner.placement.model_fields_set == set(type(cfg.learner.placement).model_fields)
```

- [ ] **Step 2: Run them (RED)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_config.py tests/unit/test_bench_throughput.py -q`
Expected: `test_sp5_config.py` fails at collection (`ImportError: cannot import name 'DistributedBundleConfig' from 'colosseum.core.config'`); the bench guard fails with `AttributeError: 'ColosseumConfig' object has no attribute 'distributed'`.

- [ ] **Step 3: Edit `src/colosseum/core/config.py`**

1. Module docstring: in the first paragraph `metrics, transport, run` → `metrics, distributed, run`; append after the SP4 line:
```
SP5: the global ``distributed`` section and ``learner.max_policy_lag`` / ``recovery_interval_sec`` / ``placement``;
``transport`` is gone (``mode`` / ``grpc_port`` are ignored with a WARNING, ``grpc_max_message_mb`` is read as
``distributed.max_message_mb``).
```
2. Delete `class TransportMode` (the enum) and `class TransportConfig`.
3. Before `class LearnerConfig` add:
```python
class LearnerPlacementConfig(StrictModel):
    """Where the hub prefers to place an agent's learner (SP5 spec block 1)."""

    node: str | None = Field(default=None, description="Preferred node name (colosseum nodes list).")
    tags: list[str] = Field(default_factory=list, description="Preferred node tags (all of them must match).")
    strict: bool = Field(default=False, description="Never place or relocate the learner off the preference "
                                                     "(the agent waits as pending instead).")


_DEVICE_RE = re.compile(r"auto|cpu|cuda|cuda:\d+|mps")
```
4. At the end of `LearnerConfig` (after `torch_threads`) add:
```python
    max_policy_lag: int | None = Field(
        default=None, ge=0,
        description="Drop chunks whose policy_version is more than this many versions behind the learner's "
                    "(dropped_chunks.stale); null = off. The lag p50 / p95 / max is in the train records always.",
    )
    recovery_interval_sec: float = Field(
        default=300.0, gt=0,
        description="Seconds between recovery points (model + trainer state, stored by the hub, overwritten, not in "
                    "the snapshot pool); a restarted learner loses at most this much training.",
    )
    placement: LearnerPlacementConfig = Field(
        default_factory=LearnerPlacementConfig,
        description="Preferred node of the learner (by name and / or tags); strict: never on another node.",
    )

    @field_validator("device")
    @classmethod
    def _check_device(cls, device: str) -> str:
        if not _DEVICE_RE.fullmatch(device):
            raise ValueError(f"learner.device must be auto, cpu, cuda, cuda:N or mps, got {device!r}")
        return device
```
5. Where `TransportConfig` was (before the `# Agents (SP3 spec block 1)` banner) add:
```python
class DistributedBundleConfig(StrictModel):
    """What the hub's code bundle holds besides the colosseum package and the project dir (SP5 spec 2.3)."""

    include: list[str] = Field(default_factory=list, description="Extra paths (relative to the project dir).")
    exclude: list[str] = Field(default_factory=list, description="Paths (globs) of the project dir to leave out.")
    max_file_mb: float = Field(default=10.0, gt=0, description="Files of the project dir larger than this are left "
                                                               "out.")


class DistributedConfig(StrictModel):
    """The hub, its nodes and the wire (SP5 spec block 9.1). Global only. Every key has a default: a one-machine
    config needs none of them."""

    port: int = Field(default=47100, ge=1, le=65535, description="The hub's port (gRPC: sessions, blobs, the "
                                                                  "installer, the relay).")
    data_port: int = Field(default=47102, ge=1, le=65535,
                           description="A node's data port (chunks from other nodes, its learners' weights); the next "
                                       "free one when taken.")
    insecure: bool = Field(default=False, description="No TLS (trusted LAN only); the token stays required.")
    heartbeat_sec: float = Field(default=1.0, gt=0, description="Seconds between a node's heartbeats.")
    node_lost_after_sec: float = Field(default=10.0, gt=0, description="A node silent this long is lost.")
    learner_relocate_after_sec: float = Field(default=120.0, gt=0,
                                              description="A lost node's learners move after this long.")
    hub_lost_timeout_sec: float = Field(default=1800.0, gt=0,
                                        description="Nodes stop their processes after this long without the hub.")
    hub_lost_collect_sec: float = Field(default=300.0, gt=0,
                                        description="Without the hub, a learner reachable only through the relay gets "
                                                    "no data after this long.")
    shutdown_grace_sec: float = Field(default=20.0, gt=0, description="Distributed stop: final snapshots over the "
                                                                       "network within this time.")
    connect_timeout_sec: float = Field(default=5.0, gt=0, description="A direct data-port connection that takes "
                                                                       "longer goes through the hub's relay.")
    block_lease_sec: float = Field(default=600.0, gt=0, description="Lease of an evaluator block (hub clock).")
    relay: bool = Field(default=True, description="Relay chunks and weights through the hub for nodes whose data "
                                                  "port is unreachable.")
    results_batch: int = Field(default=256, ge=1, description="Results per batch a node sends.")
    results_flush_sec: float = Field(default=0.25, gt=0, description="A node sends a smaller batch after this long.")
    node_result_buffer_mb: float = Field(default=256.0, gt=0,
                                         description="Unacknowledged results a node keeps (also while the hub is "
                                                     "away).")
    max_message_mb: int = Field(default=64, ge=1, description="Largest gRPC message in MiB (was "
                                                              "transport.grpc_max_message_mb).")
    bundle: DistributedBundleConfig = Field(default_factory=DistributedBundleConfig)
    requirements: str | None = Field(default=None, description="A requirements file for the nodes' environment "
                                                                "instead of the hub environment's packages.")

    @model_validator(mode="after")
    def _check(self) -> DistributedConfig:
        if self.data_port == self.port:
            raise ValueError(f"distributed.data_port ({self.data_port}) must differ from distributed.port "
                             f"({self.port})")
        if self.node_lost_after_sec <= self.heartbeat_sec:
            raise ValueError(f"distributed.node_lost_after_sec ({self.node_lost_after_sec}) must be larger than "
                             f"distributed.heartbeat_sec ({self.heartbeat_sec})")
        if self.learner_relocate_after_sec < self.node_lost_after_sec:
            raise ValueError(f"distributed.learner_relocate_after_sec ({self.learner_relocate_after_sec}) must be at "
                             f"least distributed.node_lost_after_sec ({self.node_lost_after_sec})")
        return self


# SP2 transport keys (SP5 spec 4.10): accepted in the input YAML only, never stored.
TRANSPORT_IGNORED_KEYS = ("mode", "grpc_port")
TRANSPORT_RENAMED_KEYS = {"grpc_max_message_mb": "max_message_mb"}


def translate_transport(data: dict) -> dict:
    """``transport`` of an SP1-SP4 config -> ``distributed`` (one WARNING; ConfigError-worthy conflicts raise
    ValueError): ``mode`` / ``grpc_port`` are ignored, ``grpc_max_message_mb`` becomes ``distributed.max_message_mb``
    (the old and the new key together are an error)."""
    raw = data.get("transport")
    if raw is None:
        return {k: v for k, v in data.items() if k != "transport"}
    if not isinstance(raw, dict):
        raise ValueError(f"transport (SP2-SP4) must be a section, got {type(raw).__name__}; remove it (SP5 reads "
                         f"distributed.*)")
    unknown = sorted(set(raw) - set(TRANSPORT_IGNORED_KEYS) - set(TRANSPORT_RENAMED_KEYS))
    if unknown:
        raise ValueError(f"transport has unknown keys {unknown}; the section is gone in SP5 (see distributed.*)")
    out = {k: v for k, v in data.items() if k != "transport"}
    notes = []
    ignored = [k for k in TRANSPORT_IGNORED_KEYS if k in raw]
    if ignored:
        notes.append(f"{', '.join('transport.' + k for k in ignored)} ignored (the hub and nodes choose the transport)")
    if "grpc_max_message_mb" in raw:
        distributed = out.get("distributed")
        distributed = dict(distributed) if isinstance(distributed, dict) else {}
        if "max_message_mb" in distributed:
            raise ValueError("transport.grpc_max_message_mb (SP2-SP4) and distributed.max_message_mb are both set; "
                             "keep only distributed.max_message_mb")
        distributed["max_message_mb"] = raw["grpc_max_message_mb"]
        out["distributed"] = distributed
        notes.append(f"transport.grpc_max_message_mb read as distributed.max_message_mb: {raw['grpc_max_message_mb']}")
    if notes:
        logger.warning(f"the transport section is gone in SP5: {'; '.join(notes)} (remove transport from the config; "
                       f"it is not kept in config.resolved.yaml)")
    return out
```
6. In `ColosseumConfig`: replace `transport: TransportConfig = Field(default_factory=TransportConfig)` by `distributed: DistributedConfig = Field(default_factory=DistributedConfig)` (same place, before `run`), and add this validator right before `_translate_sp2_kickstart_knobs`:
```python
    @model_validator(mode="before")
    @classmethod
    def _translate_transport(cls, data: Any) -> Any:
        """SP2-SP4's ``transport`` section -> ``distributed`` (one WARNING; spec 4.10)."""
        if not isinstance(data, dict) or "transport" not in data:
            return data
        return translate_transport(data)
```
7. After `LEGACY_OVERRIDE_KEYS.update(f"training.{knob}" for knob in SP2_KICKSTART_KNOBS)` add:
```python
LEGACY_OVERRIDE_KEYS.update(f"transport.{key}" for key in (*TRANSPORT_IGNORED_KEYS, *TRANSPORT_RENAMED_KEYS))
```
8. Keep `distributed` global only: it is not in `_AGENT_SECTIONS`, so `agents.<id>.distributed` is rejected by `_check_override_path` and by the agent entries' `extra="forbid"` (the test pins it).

In `src/colosseum/distributed.py` replace both `config.transport.grpc_max_message_mb` by `config.distributed.max_message_mb`. `grep -rn "transport\.\(mode\|grpc_port\|grpc_max\)\|TransportConfig\|TransportMode" src scripts` must then find nothing but the translation in `config.py`.

- [ ] **Step 4: Pin the new keys in the benchmark config**

In `scripts/bench_throughput.py::_make_config`: the `learner` dict ends with
```python
            "torch_threads": None,  # auto: (cpu_count - num_workers * 1) // 1
            # SP5: no lag cut (default), the recovery cadence and placement of the one machine's learner pinned
            "max_policy_lag": None,
            "recovery_interval_sec": 300.0,
            "placement": {"node": None, "tags": [], "strict": False},
        },
```
and `transport={"mode": "local"},` (the last argument) is replaced by
```python
        # SP5 (spec 9.1): every key pinned. The hub and the local node talk over loopback; heartbeat_sec 0.25 is set
        # by T2.5 (counter quantization < 1 % of the window), until then the default.
        distributed={
            "port": 47100, "data_port": 47102, "insecure": False, "heartbeat_sec": 1.0, "node_lost_after_sec": 10.0,
            "learner_relocate_after_sec": 120.0, "hub_lost_timeout_sec": 1800.0, "hub_lost_collect_sec": 300.0,
            "shutdown_grace_sec": 20.0, "connect_timeout_sec": 5.0, "block_lease_sec": 600.0, "relay": True,
            "results_batch": 256, "results_flush_sec": 0.25, "node_result_buffer_mb": 256.0, "max_message_mb": 64,
            "bundle": {"include": [], "exclude": [], "max_file_mb": 10.0}, "requirements": None,
        },
```

- [ ] **Step 5: Run the focused tests (GREEN)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_config.py tests/unit/test_bench_throughput.py tests/unit/test_config_v2.py tests/unit/test_sp2_config_overrides.py tests/unit/test_example_configs.py tests/unit/test_sp3_example_configs_v3.py tests/unit/test_sp4_sp3_compat.py tests/unit/test_sp3_sp2_compat.py tests/unit/test_sp2_distributed_roles.py tests/unit/test_sp3_distributed_guards.py tests/integration/test_sp4_sp3_resume.py tests/integration/test_sp3_sp2_resume.py -q -rw`
Expected: all pass (25 new), zero warnings. (The SP2 / SP3 fixtures' `config.resolved.yaml` hold a `transport` section: their resumes now log the one WARNING; no test asserts on its absence.)

- [ ] **Step 6: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings; ruff clean.

- [ ] **Step 7: Commit and push**

```bash
git add src/colosseum/core/config.py src/colosseum/distributed.py scripts/bench_throughput.py \
        tests/unit/test_bench_throughput.py tests/unit/test_sp5_config.py
git commit -m "feat: SP5 config (distributed section, learner lag / recovery / placement keys, transport translation)"
git push
```

---

### Task T1.5: The shared-memory weight cache library; criterion-4 read-latency script

Spec block 4.6 ("Веса на машине": immutable versioned segments + pointer, `mmap`, names with the CA and run prefixes, a node lock with its pid, deleting only dead owners' segments, retry when the segment vanished, the space check with the `--shm-size` / `emptyDir` hint), 7.8 (`/dev/shm` on Linux, `mmap` of files in the temp dir on macOS; process liveness by pid + start time), ruling R10 (weights reach workers only through this cache), criterion 4 (the read half). Pure library; the node agent and the learner wire it in T2.2 / T2.4.

**Files:**
- Create: `src/colosseum/node/__init__.py`, `src/colosseum/node/weights.py`, `scripts/measure_weight_cache.py`
- Test: `tests/unit/test_sp5_weight_cache.py`

**Interfaces:**
- Consumes: `colosseum.net.chunks.{WeightsMeta, encode_weights, decode_weights}`, `colosseum.net.frame.FrameError` (T1.1 — integrate T1.1 first), `colosseum.core.run_state.process_start_ticks`.
- Produces: the contract's `colosseum.node.weights` plus (contract note 9) `CacheLockedError`, `release_node_lock(root, prefix)`, `segment_name(prefix, key, version=None)`, `pointer_name(prefix, agent_id)`, `frame_digest(frame)`, `KEEP_LATEST = 2`, `READ_RETRIES = 5`, `SPACE_MARGIN_BYTES = 64 MiB`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_sp5_weight_cache.py`:
```python
"""The shared-memory weight cache (SP5 T1.5, spec blocks 4.6 and 7.8; §6 "Узел с поддельным хабом: кэш весов"):
segments and pointers, the newest two versions kept, readers keep their views after a deletion and re-read the
pointer when its segment vanished, snapshots and blobs by id, the free-space check, prefixes of two hubs or runs never
meet, the node lock, and cleanup of dead owners only."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys

import numpy as np
import pytest

from colosseum.core.run_state import process_start_ticks
from colosseum.net.chunks import WeightsMeta, encode_weights
from colosseum.node import weights
from colosseum.node.weights import (
    SHM_DIR_ENV,
    CacheFullError,
    CacheKey,
    CacheLockedError,
    WeightCacheReader,
    WeightCacheWriter,
    acquire_node_lock,
    cache_prefix,
    cleanup_dead,
    shm_dir,
)

CA = "ab" * 32
RUN = "0123456789abcdef" * 2


def _frame(agent: str, version: int, value: float, *, kind: str = "latest", generation: int = 1,
           teacher_active: bool = False) -> bytes:
    sd = {"w": np.full((4, 4), value, np.float32), "b": np.arange(3, dtype=np.int64)}
    return encode_weights(sd, WeightsMeta(agent, kind, generation, version, teacher_active=teacher_active))


def _pair(tmp_path, node: int = 1, run: str = RUN, ca: str = CA):
    prefix = cache_prefix(ca, run, node)
    return WeightCacheWriter(tmp_path / "shm", prefix), WeightCacheReader(tmp_path / "shm", prefix)


def _files(tmp_path) -> list[str]:
    return sorted(os.listdir(tmp_path / "shm"))


def test_the_cache_dir_and_the_prefix(tmp_path, monkeypatch):
    assert shm_dir() == tmp_path / "shm"                                     # tests/conftest.py redirects it
    monkeypatch.delenv(SHM_DIR_ENV)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(weights.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    assert shm_dir() == tmp_path / "tmp" / "colosseum-shm"                   # macOS: files in the temp dir
    assert cache_prefix(CA, RUN, 3) == "colosseum-abababab-01234567-n3"
    for bad in (("xyz", RUN, 1), (CA, "run-1", 1), (CA, RUN, -1)):
        with pytest.raises(ValueError):
            cache_prefix(*bad)


def test_publish_and_read_the_latest_weights(tmp_path):
    writer, reader = _pair(tmp_path)
    assert reader.read_latest("main") is None and reader.latest_version("main") is None
    name = writer.publish_latest("main", _frame("main", 7, 1.5, generation=2, teacher_active=True), version=7,
                                 generation=2, teacher_active=True)
    assert name == "colosseum-abababab-01234567-n1-main-v7"
    assert reader.latest_version("main") == (7, 2)
    cached = reader.read_latest("main")
    assert cached.key == CacheKey("latest", "main") and cached.meta.policy_version == 7
    assert cached.meta.generation == 2 and cached.meta.teacher_active
    assert np.all(cached.state_dict["w"] == 1.5) and cached.state_dict["b"].tolist() == [0, 1, 2]
    assert not cached.state_dict["w"].flags.writeable                         # views of a read-only mapping
    assert reader.read_latest("main", newer_than=7) is None and reader.read_latest("main", newer_than=6) is not None
    pointer = json.loads((tmp_path / "shm" / "colosseum-abababab-01234567-n1-main.latest").read_text())
    assert set(pointer) == {"segment", "version", "generation", "size", "digest", "teacher_active"}
    assert pointer["size"] == cached.nbytes and len(pointer["digest"]) == 32
    assert stat.S_IMODE(os.stat(tmp_path / "shm" / name).st_mode) == 0o600


def test_the_newest_two_versions_are_kept_and_mapped_views_survive_deletion(tmp_path):
    writer, reader = _pair(tmp_path)
    for version in (1, 2, 3):
        writer.publish_latest("main", _frame("main", version, float(version)), version=version, generation=1,
                              teacher_active=False)
    held = reader.read_latest("main")                                         # maps v3
    for version in (4, 5):
        writer.publish_latest("main", _frame("main", version, float(version)), version=version, generation=1,
                              teacher_active=False)
    assert [f for f in _files(tmp_path) if "-main-v" in f] == ["colosseum-abababab-01234567-n1-main-v4",
                                                                 "colosseum-abababab-01234567-n1-main-v5"]
    assert held.meta.policy_version == 3 and np.all(held.state_dict["w"] == 3.0)   # v3 is deleted, still readable
    restarted = WeightCacheWriter(tmp_path / "shm", writer.prefix)           # a new learner incarnation adopts v4, v5
    restarted.publish_latest("main", _frame("main", 6, 6.0, generation=2), version=6, generation=2,
                             teacher_active=False)
    assert sorted(f for f in _files(tmp_path) if "-main-v" in f)[-2:] == ["colosseum-abababab-01234567-n1-main-v5",
                                                                          "colosseum-abababab-01234567-n1-main-v6"]
    assert len([f for f in _files(tmp_path) if "-main-v" in f]) == 2


def test_a_reader_re_reads_the_pointer_when_its_segment_vanished(tmp_path, monkeypatch):
    writer, reader = _pair(tmp_path)
    writer.publish_latest("main", _frame("main", 1, 1.0), version=1, generation=1, teacher_active=False)
    real_open = weights._open_segment
    calls = []

    def racing_open(path):
        calls.append(path.name)
        if len(calls) == 1:                                                   # the writer moves on in between
            writer.publish_latest("main", _frame("main", 2, 2.0), version=2, generation=1, teacher_active=False)
            writer.publish_latest("main", _frame("main", 3, 3.0), version=3, generation=1, teacher_active=False)
            raise FileNotFoundError(path)
        return real_open(path)

    monkeypatch.setattr(weights, "_open_segment", racing_open)
    cached = reader.read_latest("main")
    assert cached.meta.policy_version == 3 and calls[-1].endswith("-main-v3") and len(calls) == 2
    (tmp_path / "shm" / "colosseum-abababab-01234567-n1-main-v3").unlink()   # a pointer to nothing
    monkeypatch.setattr(weights, "_open_segment", real_open)
    assert reader.read_latest("main") is None                                 # bounded retries, no exception


def test_snapshots_and_blobs_by_id(tmp_path):
    writer, reader = _pair(tmp_path)
    snap, blob = CacheKey("snapshot", "main", "ckpt_v40"), CacheKey("blob", "", "0f" * 16)
    assert not reader.has(snap) and reader.get(snap) is None
    assert writer.put(snap, _frame("main", 40, 4.0, kind="snapshot")) == "colosseum-abababab-01234567-n1-main-ckpt_v40"
    inode = os.stat(tmp_path / "shm" / "colosseum-abababab-01234567-n1-main-ckpt_v40").st_ino
    writer.put(snap, _frame("main", 40, 9.0, kind="snapshot"))               # immutable: kept as is
    assert os.stat(tmp_path / "shm" / "colosseum-abababab-01234567-n1-main-ckpt_v40").st_ino == inode
    writer.put(blob, _frame("random_net", 0, 0.5, kind="fixed"))
    assert reader.has(snap) and reader.get(snap).meta.policy_version == 40
    assert np.all(reader.get(snap).state_dict["w"] == 4.0) and reader.get(blob).meta.kind == "fixed"
    writer.drop(snap)
    assert not reader.has(snap) and reader.has(blob)
    bad_keys = (CacheKey("snapshot", "main", "v40"), CacheKey("blob", "", "NOT-HEX"),
                CacheKey("snapshot", "a b", "ckpt_v1"))
    for bad in bad_keys:
        with pytest.raises(ValueError):
            writer.put(bad, b"")
    with pytest.raises(ValueError):
        writer.put(CacheKey("latest", "main"), b"")


def test_a_corrupt_or_mismatched_segment_is_ignored_with_one_warning(tmp_path, caplog):
    writer, reader = _pair(tmp_path)
    (tmp_path / "shm" / "colosseum-abababab-01234567-n1-main-ckpt_v1").write_bytes(b"garbage" * 10)
    assert reader.get(CacheKey("snapshot", "main", "ckpt_v1")) is None
    assert reader.get(CacheKey("snapshot", "main", "ckpt_v1")) is None
    writer.publish_latest("main", _frame("other", 5, 1.0), version=5, generation=1, teacher_active=False)
    assert reader.read_latest("main") is None
    assert len([r for r in caplog.records if "ignored" in r.getMessage()]) == 2


def test_the_free_space_check_refuses_a_segment_with_the_hint(tmp_path, monkeypatch):
    writer, _ = _pair(tmp_path)
    frame = _frame("main", 1, 1.0)
    usage = weights.shutil.disk_usage(tmp_path)
    monkeypatch.setattr(weights.shutil, "disk_usage",
                        lambda path: usage._replace(free=2 * len(frame) + weights.SPACE_MARGIN_BYTES - 1))
    with pytest.raises(CacheFullError, match="--shm-size"):
        writer.publish_latest("main", frame, version=1, generation=1, teacher_active=False)
    with pytest.raises(CacheFullError, match="emptyDir"):
        writer.put(CacheKey("blob", "", "aa"), frame)
    assert not any(f.endswith("-main-v1") for f in _files(tmp_path))


def test_two_hubs_and_two_runs_on_one_machine_never_share_files(tmp_path):
    a, ra = _pair(tmp_path)
    b, rb = _pair(tmp_path, ca="cd" * 32)
    c, rc = _pair(tmp_path, run="fedcba9876543210" * 2)
    d, rd = _pair(tmp_path, node=12)
    for i, writer in enumerate((a, b, c, d)):
        writer.publish_latest("main", _frame("main", i + 1, float(i)), version=i + 1, generation=1,
                              teacher_active=False)
    assert [r.latest_version("main")[0] for r in (ra, rb, rc, rd)] == [1, 2, 3, 4]
    assert len(_files(tmp_path)) == 8 and len({f.split("-main")[0] for f in _files(tmp_path)}) == 4


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_the_node_lock_and_cleanup_of_dead_owners_only(tmp_path):
    root = tmp_path / "shm"
    mine, theirs = cache_prefix(CA, RUN, 1), cache_prefix(CA, RUN, 2)
    lock = acquire_node_lock(root, mine)
    assert json.loads(lock.read_text()) == {"pid": os.getpid(), "proc_start": process_start_ticks(os.getpid())}
    assert acquire_node_lock(root, mine) == lock                               # re-acquiring our own lock is fine
    WeightCacheWriter(root, mine).publish_latest("main", _frame("main", 1, 1.0), version=1, generation=1,
                                                 teacher_active=False)
    dead = WeightCacheWriter(root, theirs)
    (root / f"{theirs}.lock").write_text(json.dumps({"pid": _dead_pid(), "proc_start": None}))
    dead.publish_latest("main", _frame("main", 1, 1.0), version=1, generation=1, teacher_active=False)
    (root / f".{theirs}-main-v2.tmp-999").write_bytes(b"half")                  # a dead writer's temp file
    stray = cache_prefix(CA, RUN, 3)
    (root / f"{stray}-main-v9").write_bytes(b"x")                              # no lock at all
    (root / "unrelated-file").write_bytes(b"keep")
    reused = cache_prefix(CA, RUN, 4)                                          # the pid lives, the start time differs
    (root / f"{reused}.lock").write_text(json.dumps({"pid": os.getpid(), "proc_start": -1}))
    removed = cleanup_dead(root)
    assert sorted(os.listdir(root)) == sorted([f"{mine}.lock", f"{mine}-main-v1", f"{mine}-main.latest",
                                               "unrelated-file"])
    assert f"{theirs}.lock" in removed and f"{stray}-main-v9" in removed and f"{reused}.lock" in removed
    assert cleanup_dead(root) == [] and cleanup_dead(tmp_path / "missing") == []
    parent = os.getppid()                                                     # a live foreign owner
    (root / f"{theirs}.lock").write_text(json.dumps({"pid": parent, "proc_start": process_start_ticks(parent)}))
    with pytest.raises(CacheLockedError, match=str(parent)):
        acquire_node_lock(root, theirs)
```

- [ ] **Step 2: Run it (RED)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_weight_cache.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.node'`.

- [ ] **Step 3: Write the package and the cache**

Create `src/colosseum/node/__init__.py`:
```python
"""The SP5 node agent (one per machine) and its children: the shared-memory weight cache (``weights``), the hub
session client, child supervision, the chunk router, child entry points, hardware probes and node logs."""
```

Create `src/colosseum/node/weights.py`:
```python
"""The per-machine shared-memory weight cache (SP5 spec blocks 4.6 and 7.8; ruling R10).

Workers and evaluators read weights only from here; the writer of an agent's latest weights is its learner when the
learner runs on this machine, else the node (versions fetched from another machine); snapshots, frozen, teacher and
``init`` weights (immutable, by id) are written by the node. One writer per file, any number of readers, no locks
between processes (spec 4.7 principle).

Files in :func:`shm_dir` (``$COLOSSEUM_SHM_DIR``, else ``/dev/shm`` on Linux, else ``<tempdir>/colosseum-shm``), all
named with the prefix ``P = colosseum-<ca8>-<run8>-n<node>`` (:func:`cache_prefix`), so two hubs, two runs and two
nodes on one machine never share a file:

=============================  ===========================================================  ====================
``{P}-{agent}-v{N}``           weights frame (``colosseum.net.chunks.encode_weights``)       learner or node
``{P}-{agent}.latest``         JSON ``{"segment", "version", "generation", "size",          same single writer
                               "digest", "teacher_active"}``, replaced atomically
``{P}-{agent}-{checkpoint}``   weights frame of a snapshot (``ckpt_v<N>``)                  node
``{P}-blob-{blob_id}``         weights frame of a frozen / teacher / init / recovery blob   node
``{P}.lock``                   JSON ``{"pid", "proc_start"}`` of the node process            node
=============================  ===========================================================  ====================

Segments are immutable: a writer writes ``.{name}.tmp-<pid>`` (mode ``0600``) and renames it into place, then
replaces the pointer; the writer of ``latest`` keeps the newest 2 versions and deletes older ones. A reader opens the
segment with ``os.open`` + ``mmap`` (read-only) and decodes the frame into ``np.frombuffer`` views (no copy; the
mapping lives while any view does; ``load_state_dict`` copies). Deleting a file a reader has mapped is safe on
POSIX; a reader that finds the pointer's segment gone re-reads the pointer (``READ_RETRIES``). Not
``multiprocessing.shared_memory`` (on Python 3.12 another process unlinks a segment it did not create) and not a
seqlock (readers starve): ``research/sp5_data_plane.md`` §2.6.

Before every write the free space must be ``size x 2 + 64 MiB``, else :class:`CacheFullError` (the node logs it as an
ERROR with the hint and takes no work). :func:`cleanup_dead` removes only the files of prefixes whose lock names a
dead process, and stray files of prefixes without a lock.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mmap
import os
import re
import shutil
import sys
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from colosseum.core.run_state import process_start_ticks
from colosseum.net.chunks import WeightsMeta, decode_weights
from colosseum.net.frame import FrameError

logger = logging.getLogger(__name__)

SHM_DIR_ENV = "COLOSSEUM_SHM_DIR"
KEEP_LATEST = 2                   # versions of an agent's latest weights kept (the pointer's and the previous one)
READ_RETRIES = 5
SPACE_MARGIN_BYTES = 64 * 2**20
_PREFIX_RE = re.compile(r"^\.?(colosseum-[0-9a-f]{8}-[0-9a-f]{8}-n\d+)[-.]")
_AGENT_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_-]*$")
_CKPT_RE = re.compile(r"^ckpt_v\d+$")
_BLOB_RE = re.compile(r"^[0-9a-f]{1,64}$")
FULL_HINT = ("the shared-memory weight cache is full: free space in {root} (Docker: `docker run --shm-size=1g`, "
             "K8s: an `emptyDir: {{medium: Memory}}` volume at /dev/shm), or set COLOSSEUM_SHM_DIR to a larger "
             "tmpfs")


class CacheFullError(RuntimeError):
    """Not enough space for a segment (message: the hint)."""


class CacheLockedError(RuntimeError):
    """Another live process owns this cache prefix (a second node of the same identity and run on this machine)."""


def shm_dir() -> Path:
    """Where the cache lives: ``$COLOSSEUM_SHM_DIR``, else ``/dev/shm`` (Linux), else ``<tempdir>/colosseum-shm``."""
    configured = os.environ.get(SHM_DIR_ENV)
    if configured:
        return Path(configured)
    if sys.platform.startswith("linux") and Path("/dev/shm").is_dir():
        return Path("/dev/shm")
    return Path(tempfile.gettempdir()) / "colosseum-shm"


def cache_prefix(ca_fingerprint: str, run_id: str, node: int) -> str:
    """``colosseum-<ca8>-<run8>-n<node>``."""
    if not re.fullmatch(r"[0-9a-f]{8,}", ca_fingerprint) or not re.fullmatch(r"[0-9a-f]{8,}", run_id):
        raise ValueError(f"cache_prefix needs hex ids, got {ca_fingerprint!r} / {run_id!r}")
    if int(node) < 0:
        raise ValueError(f"node number must be >= 0, got {node}")
    return f"colosseum-{ca_fingerprint[:8]}-{run_id[:8]}-n{int(node)}"


@dataclass(frozen=True)
class CacheKey:
    kind: Literal["latest", "snapshot", "blob"]
    agent_id: str = ""
    ident: str = ""                     # checkpoint id (snapshot) or blob id (blob); "" for latest


@dataclass(frozen=True)
class CachedWeights:
    key: CacheKey
    state_dict: dict[str, np.ndarray]    # read-only views over the mapping (load_state_dict copies)
    meta: WeightsMeta
    nbytes: int


def _check_agent(agent_id: str) -> str:
    if not _AGENT_RE.match(agent_id):
        raise ValueError(f"invalid agent id {agent_id!r}")
    return agent_id


def segment_name(prefix: str, key: CacheKey, version: int | None = None) -> str:
    """The file name of ``key`` (``latest`` needs ``version``)."""
    if key.kind == "latest":
        if version is None or version < 0:
            raise ValueError("a latest segment needs its version")
        return f"{prefix}-{_check_agent(key.agent_id)}-v{int(version)}"
    if key.kind == "snapshot":
        if not _CKPT_RE.match(key.ident):
            raise ValueError(f"invalid checkpoint id {key.ident!r}")
        return f"{prefix}-{_check_agent(key.agent_id)}-{key.ident}"
    if key.kind == "blob":
        if not _BLOB_RE.match(key.ident):
            raise ValueError(f"invalid blob id {key.ident!r}")
        return f"{prefix}-blob-{key.ident}"
    raise ValueError(f"unknown cache key kind {key.kind!r}")


def pointer_name(prefix: str, agent_id: str) -> str:
    return f"{prefix}-{_check_agent(agent_id)}.latest"


def frame_digest(frame: bytes | memoryview) -> str:
    return hashlib.blake2b(frame, digest_size=16).hexdigest()


def _write_file(path: Path, data: bytes | memoryview) -> None:
    """Write ``data`` to ``path`` through a ``0600`` temp file and a rename (readers never see half a file)."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class WeightCacheWriter:
    """Writes segments and pointers of one prefix (one process writes a given agent's latest)."""

    def __init__(self, root: Path, prefix: str) -> None:
        self.root = Path(root)
        self.prefix = prefix
        self.root.mkdir(parents=True, exist_ok=True)
        self._kept: dict[str, deque[str]] = {}

    def _check_space(self, size: int) -> None:
        free = shutil.disk_usage(self.root).free
        if free < 2 * size + SPACE_MARGIN_BYTES:
            raise CacheFullError(f"{size / 2**20:.1f} MiB segment, {free / 2**20:.1f} MiB free: "
                                 + FULL_HINT.format(root=self.root))

    def _kept_of(self, agent_id: str) -> deque[str]:
        kept = self._kept.get(agent_id)
        if kept is None:                  # a restarted writer adopts what an earlier incarnation left
            pattern = re.compile(re.escape(f"{self.prefix}-{agent_id}-v") + r"(\d+)$")
            found = sorted((int(m.group(1)), p.name) for p in self.root.iterdir() if (m := pattern.match(p.name)))
            kept = self._kept[agent_id] = deque(name for _, name in found)
        return kept

    def publish_latest(self, agent_id: str, frame: bytes, *, version: int, generation: int,
                       teacher_active: bool) -> str:
        """Write version ``version`` of ``agent_id``'s latest weights, point the pointer at it, keep the newest
        ``KEEP_LATEST`` versions; returns the segment name."""
        name = segment_name(self.prefix, CacheKey("latest", agent_id), version)
        self._check_space(len(frame))
        kept = self._kept_of(agent_id)
        _write_file(self.root / name, frame)
        pointer = {"segment": name, "version": int(version), "generation": int(generation), "size": len(frame),
                   "digest": frame_digest(frame), "teacher_active": bool(teacher_active)}
        _write_file(self.root / pointer_name(self.prefix, agent_id), json.dumps(pointer).encode("utf-8"))
        if name in kept:
            kept.remove(name)
        kept.append(name)
        while len(kept) > KEEP_LATEST:
            (self.root / kept.popleft()).unlink(missing_ok=True)
        return name

    def put(self, key: CacheKey, frame: bytes) -> str:
        """Write an immutable snapshot / blob segment (kept as is if it exists already); returns its name."""
        if key.kind == "latest":
            raise ValueError("latest weights go through publish_latest")
        name = segment_name(self.prefix, key)
        if not (self.root / name).exists():
            self._check_space(len(frame))
            _write_file(self.root / name, frame)
        return name

    def drop(self, key: CacheKey) -> None:
        """Remove a snapshot / blob segment (readers that mapped it keep their views)."""
        if key.kind == "latest":
            raise ValueError("latest segments are pruned by publish_latest")
        (self.root / segment_name(self.prefix, key)).unlink(missing_ok=True)


def _open_segment(path: Path) -> mmap.mmap:
    """A read-only mapping of ``path`` (FileNotFoundError when it is gone; ValueError when empty)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        return mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)


class WeightCacheReader:
    """Reads segments of one prefix. Never writes; never blocks on a writer."""

    def __init__(self, root: Path, prefix: str) -> None:
        self.root = Path(root)
        self.prefix = prefix
        self._warned: set[str] = set()

    def _pointer(self, agent_id: str) -> dict | None:
        try:
            data = json.loads((self.root / pointer_name(self.prefix, agent_id)).read_bytes())
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return None                   # mid-replace on a platform without atomic rename: next poll
        return data if isinstance(data, dict) and isinstance(data.get("segment"), str) else None

    def latest_version(self, agent_id: str) -> tuple[int, int] | None:
        """``(version, generation)`` the pointer names (no segment is opened), or None."""
        pointer = self._pointer(agent_id)
        return None if pointer is None else (int(pointer["version"]), int(pointer["generation"]))

    def _decode(self, key: CacheKey, mapping: mmap.mmap, name: str) -> CachedWeights | None:
        try:
            state_dict, meta = decode_weights(memoryview(mapping), cap=len(mapping))
        except FrameError as e:
            if name not in self._warned:
                self._warned.add(name)
                logger.warning(f"weight cache segment {name} is not a weights frame ({e}); ignored")
            return None
        return CachedWeights(key, state_dict, meta, len(mapping))

    def read_latest(self, agent_id: str, *, newer_than: int = -1) -> CachedWeights | None:
        """The pointer's version if it is newer than ``newer_than``, else None (also when nothing is published)."""
        for _ in range(READ_RETRIES):
            pointer = self._pointer(agent_id)
            if pointer is None or int(pointer["version"]) <= newer_than:
                return None
            name = pointer["segment"]
            try:
                mapping = _open_segment(self.root / name)
            except FileNotFoundError:
                continue                  # replaced and pruned since the pointer was read: read it again
            except (OSError, ValueError):
                return None
            cached = self._decode(CacheKey("latest", agent_id), mapping, name)
            if cached is not None and (cached.meta.agent_id != agent_id
                                       or cached.meta.policy_version != int(pointer["version"])):
                if name not in self._warned:
                    self._warned.add(name)
                    logger.warning(f"weight cache segment {name} holds {cached.meta.agent_id} "
                                   f"v{cached.meta.policy_version}, the pointer says {agent_id} "
                                   f"v{pointer['version']}; ignored")
                return None
            return cached
        return None

    def get(self, key: CacheKey) -> CachedWeights | None:
        """The weights of ``key`` (``latest``: the pointer's version), or None when absent."""
        if key.kind == "latest":
            return self.read_latest(key.agent_id)
        name = segment_name(self.prefix, key)
        try:
            mapping = _open_segment(self.root / name)
        except (OSError, ValueError):
            return None
        return self._decode(key, mapping, name)

    def has(self, key: CacheKey) -> bool:
        if key.kind == "latest":
            return (self.root / pointer_name(self.prefix, key.agent_id)).exists()
        return (self.root / segment_name(self.prefix, key)).exists()


# ---------------------------------------------------------------------------
# Ownership and cleanup
# ---------------------------------------------------------------------------


def _owner_alive(lock: Path) -> bool:
    """Whether the process a lock file names still runs (pid, and its start time where the OS tells it)."""
    try:
        data = json.loads(lock.read_text())
        pid, start = int(data["pid"]), data.get("proc_start")
    except FileNotFoundError:
        return False
    except (OSError, ValueError, KeyError, TypeError):
        return False                      # a broken lock owns nothing
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass                              # exists, owned by another user
    now = process_start_ticks(pid)
    return not (start is not None and now is not None and int(start) != now)


def acquire_node_lock(root: Path, prefix: str) -> Path:
    """Write ``{P}.lock`` with this process's pid and start time (before any segment of the prefix);
    CacheLockedError if a live process owns the prefix already."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f"{prefix}.lock"
    if lock.exists() and _owner_alive(lock):
        owner = json.loads(lock.read_text()).get("pid")
        if owner != os.getpid():
            raise CacheLockedError(f"weight cache {lock} belongs to the running process {owner} (another node of the "
                                   f"same identity and run on this machine?)")
    _write_file(lock, json.dumps({"pid": os.getpid(), "proc_start": process_start_ticks(os.getpid())}).encode())
    return lock


def release_node_lock(root: Path, prefix: str) -> None:
    """Remove every file of ``prefix`` and its lock (a node that stops cleanly)."""
    removed = _remove_prefix(Path(root), prefix, os.listdir(root))
    logger.debug(f"weight cache {prefix}: removed {len(removed)} files")


def _remove_prefix(root: Path, prefix: str, names: list[str]) -> list[str]:
    removed = []
    lock = f"{prefix}.lock"
    for name in names:
        if name != lock and _PREFIX_RE.match(name) and _PREFIX_RE.match(name).group(1) == prefix:
            (root / name).unlink(missing_ok=True)
            removed.append(name)
    if lock in names:
        (root / lock).unlink(missing_ok=True)
        removed.append(lock)
    return removed


def cleanup_dead(root: Path) -> list[str]:
    """Remove the files of every prefix whose lock names a dead process, and stray files of prefixes without a lock
    (a lock is always written before a prefix's first segment). Live owners' files are never touched; other files
    in ``root`` neither. Returns the removed names."""
    root = Path(root)
    if not root.is_dir():
        return []
    names = os.listdir(root)
    prefixes = {m.group(1) for name in names if (m := _PREFIX_RE.match(name))}
    removed: list[str] = []
    for prefix in sorted(prefixes):
        lock = root / f"{prefix}.lock"
        if f"{prefix}.lock" in names:
            if _owner_alive(lock):
                continue
        elif lock.exists():               # the owner started after the listing: its files are not stray
            continue
        removed += _remove_prefix(root, prefix, names)
    if removed:
        logger.info(f"weight cache {root}: removed {len(removed)} files of dead nodes")
    return removed
```

- [ ] **Step 4: Run the focused test (GREEN)**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_weight_cache.py -q -rw`
Expected: 9 passed, zero warnings; `ls /dev/shm | grep colosseum` shows nothing (the tests write under `tmp_path/shm`).

- [ ] **Step 5: Write the criterion-4 read-latency script and run it on a quiet machine**

Create `scripts/measure_weight_cache.py` (rule fixed in its docstring before measuring):
```python
"""SP5 criterion 4, second half (spec §3): a worker reads a 50 MB model from the shared-memory weight cache (``mmap`` +
``np.frombuffer`` of every leaf, without ``load_state_dict``) while the learner publishes twice a second; the p95 of
the read must be at most 100 ms.

    env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/measure_weight_cache.py [--seconds 30] [--json OUT]

A writer process (spawn) publishes a 50 MB float32 state dict (100 leaves of 0.5 MB) with ``WeightCacheWriter.
publish_latest`` every 0.5 s, each version with new values. The reader (this process) loops for ``--seconds``:
``WeightCacheReader.read_latest`` of the newest version, then one read of every 4 KiB page of every leaf (so the
mapping is really read, not only set up). Each read is timed with ``time.perf_counter`` (a duration on one machine).
The cache lives in ``shm_dir()`` (``/dev/shm``; ``--dir`` elsewhere) under a fresh prefix that is removed at the end.

Rule (fixed before measuring, T1.5): PASS iff p95 <= 100 ms over at least 50 reads, at least 20 versions were
published during the window, and every read saw a complete version (the leaf values equal the version's). Run alone
on a quiet machine, two runs, both must pass.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import secrets
import statistics
import time
from pathlib import Path

import numpy as np

from colosseum.net.chunks import WeightsMeta, encode_weights
from colosseum.node.weights import (
    WeightCacheReader,
    WeightCacheWriter,
    acquire_node_lock,
    cache_prefix,
    release_node_lock,
    shm_dir,
)

MODEL_BYTES = 50 * 2**20
LEAVES = 100
PUBLISH_SEC = 0.5
P95_MAX_SEC = 0.100
MIN_READS, MIN_VERSIONS = 50, 20
AGENT = "main"


def _state_dict(version: int) -> dict[str, np.ndarray]:
    per_leaf = MODEL_BYTES // LEAVES // 4
    return {f"layer{i}.weight": np.full(per_leaf, float(version), np.float32) for i in range(LEAVES)}


def _writer(root: str, prefix: str, stop, published) -> None:
    writer = WeightCacheWriter(Path(root), prefix)
    version = 0
    next_at = time.monotonic()
    while not stop.is_set():
        version += 1
        frame = encode_weights(_state_dict(version), WeightsMeta(AGENT, "latest", 1, version))
        writer.publish_latest(AGENT, frame, version=version, generation=1, teacher_active=False)
        published.value = version
        next_at += PUBLISH_SEC
        stop.wait(max(0.0, next_at - time.monotonic()))


def _touch(state_dict: dict[str, np.ndarray]) -> float:
    """Read one float of every 4 KiB page of every leaf; returns their sum."""
    return float(sum(leaf[::1024].sum(dtype=np.float64) for leaf in state_dict.values()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--dir", type=str, default=None, help="cache directory (default: shm_dir())")
    parser.add_argument("--json", type=str, default=None, help="write the numbers to this file")
    args = parser.parse_args()
    root = Path(args.dir) if args.dir else shm_dir()
    prefix = cache_prefix(secrets.token_hex(32), secrets.token_hex(16), 1)
    acquire_node_lock(root, prefix)
    ctx = mp.get_context("spawn")
    stop, published = ctx.Event(), ctx.Value("q", 0)
    proc = ctx.Process(target=_writer, args=(str(root), prefix, stop, published), daemon=True)
    proc.start()
    reader = WeightCacheReader(root, prefix)
    durations: list[float] = []
    incomplete = 0
    try:
        while reader.latest_version(AGENT) is None:
            time.sleep(0.01)
        first_version = published.value
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            start = time.perf_counter()
            cached = reader.read_latest(AGENT)
            total = _touch(cached.state_dict) if cached is not None else None
            durations.append(time.perf_counter() - start)
            pages_per_leaf = -(-(MODEL_BYTES // LEAVES // 4) // 1024)
            if cached is None or total != cached.meta.policy_version * pages_per_leaf * LEAVES:
                incomplete += 1
            del cached
        versions = published.value - first_version
    finally:
        stop.set()
        proc.join(timeout=10)
        release_node_lock(root, prefix)
    p95 = statistics.quantiles(durations, n=20)[-1] if len(durations) >= 20 else max(durations)
    ok = p95 <= P95_MAX_SEC and len(durations) >= MIN_READS and versions >= MIN_VERSIONS and incomplete == 0
    verdict = "PASS" if ok else "FAIL"
    print(f"{len(durations)} reads of {MODEL_BYTES / 2**20:.0f} MB while {versions} versions were published: "
          f"median {statistics.median(durations) * 1e3:.2f} ms, p95 {p95 * 1e3:.2f} ms, max {max(durations) * 1e3:.2f} "
          f"ms, incomplete {incomplete} (p95 <= {P95_MAX_SEC * 1e3:.0f} ms: {verdict})")
    if args.json:
        Path(args.json).write_text(json.dumps({"reads": len(durations), "versions": versions, "p95_sec": p95,
                                               "median_sec": statistics.median(durations), "max_sec": max(durations),
                                               "incomplete": incomplete, "verdict": verdict, "dir": str(root)},
                                              indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
```
Run alone on a quiet machine, twice (it uses the real `/dev/shm`, about 100 MB, and removes its files at the end):
```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/measure_weight_cache.py --json /tmp/cache-1.json
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/measure_weight_cache.py --json /tmp/cache-2.json
```
Expected: `... p95 ... ms, incomplete 0 (p95 <= 100 ms: PASS)` both times, exit 0 (scratch: median 1.4 ms, p95 1.7 ms over 6 269 reads, 24 versions); afterwards `ls /dev/shm | grep colosseum` is empty. Put both outputs into the task report (T7.4 copies them into `docs/benchmarks.md`). A FAIL is reported with its numbers; the controller decides.

- [ ] **Step 6: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings; ruff clean.

- [ ] **Step 7: Commit and push**

```bash
git add src/colosseum/node/__init__.py src/colosseum/node/weights.py scripts/measure_weight_cache.py \
        tests/unit/test_sp5_weight_cache.py
git commit -m "feat: SP5 shared-memory weight cache (segments, pointers, node lock, dead-owner cleanup)"
git push
```

---

## Contract notes

### Additions and deviations versus `00-overview.md`

1. **`colosseum.net.frame`:** `decode_frame(buf, *, cap, copy=False, kind=None)` — `kind` refuses a frame of another kind before the leaves are read (`FrameError("kind")`); `copy=True` makes **one** private copy of the data region and returns writable views of it (not a copy per leaf), so torch shares the memory. Added: `frame_sizes(buf) -> (raw data bytes, stored data bytes)` (what `Lz4Probe.observe` needs; the sender cannot know the compressed size otherwise), `lookup_class(module, qualname)` (the never-importing namedtuple lookup, moved from `transport/serialization.py`), `Lz4Probe.enabled`, `MAX_NDIM = 32`, `FLAG_LZ4 = 1`, `FRAME_KINDS`, `DTYPES` (the whitelist as a frozenset). Encoding refuses NaN / Infinity in the header and in scalars (`TypeError`); decoding refuses them (`header_json`) and requires every leaf to be referenced exactly once (`leaf_ref`). `FrameError(reason, message)` is constructed with both.
2. **`colosseum.net.chunks`:** the learner checks a chunk on its **numpy payload, inside `decode_chunk`**: `decode_chunk(buf, *, cap, validator=None)` (the intake path: frame → decode → check → chunk), `ChunkValidator.check(payload, agent_id, policy_version=None)`; `ChunkValidator.validate(chunk)` stays for already-built chunks (the SP4 queue path, tests). Reason: criterion 5 (validating torch tensors made the new path only 6× faster; ruling P1-1). `ChunkValidator` gains `chunk_length` (S must be `rollout.chunk_length`: chunks of another S would break the learner's batch stacking) and `state_template` (the model's `initial_state(1)` as numpy: structure, shapes, dtypes); it also checks finite rewards everywhere and discrete action ranges / finite box values for `actions` (not only `teacher_action`), because an out-of-range index or a NaN would stop or poison the learner. It does **not** check mask legality (ruling P1-2). `ChunkError(reason, detail)` has a `detail` string; `decode_chunk` raises `ChunkError("invalid", "frame:<reason>: ...")` for frame errors too (one `except` for the intake). Added helpers `chunk_meta(meta) -> ChunkMeta`, `check_payload_fields(payload) -> int`, constants `WEIGHTS_KINDS`, `CHECKPOINT_KINDS`.
3. **`colosseum.core.types.check_slot_arrays(kind, terminal, reset_after, has_teacher, *, agent_id, policy_version)`** — the numpy core of `validate_slot_structure` (same rules and messages), created by **T1.1**, which therefore edits `core/types.py` too (only this function; T1.2 edits only the three dataclasses — disjoint hunks, cherry-pick friendly). Deviation from the overview's file split.
4. **`VersionInfo` lives in `colosseum.net.messages` (T1.2)**, not in `colosseum.hub.state` (T2.1): T1.2 runs before T2.1 and its converters need the class. T2.1 imports it from `colosseum.net.messages` and re-exports it from `colosseum.hub.state` (`from colosseum.net.messages import VersionInfo  # re-export`), so the contract's name keeps working. Added in `messages`: `protocol_compatible(major, minor)` (same major), `results_to_batch(seq, results)`, `seat_result_to_proto` / `seat_result_from_proto`.
5. **Proto:** an `Empty` message (`Bootstrap/Info(Empty)`); `/colosseum.v1.Bootstrap/Script` is deliberately not declared (raw bytes, R2; T6.2 registers it as a generic handler). Field numbers are T1.2's (in the `.proto` above); comments name the enum-like string values. Runtime floors: `grpcio>=1.84`, `protobuf>=7.35.1` (gencode of grpcio-tools 1.84; protobuf 7.x checks major equality at import).
6. **`colosseum.net.security`:** `NodeRecord` dataclass (`node_id, number, name, machine_id, tags, secret_sha256, cert_serial, persistent, registered_run`); `NodeBook.register(..., persistent=True, run_id="")` re-registers a known `node_id` with the same number and a new secret (old secret invalid at once) and raises `SecurityError` for a banned node id or machine id; `NodeBook.{records, get, reload, set_cert_serial}`; `HubIdentity.rotate()` keeps the **same `NodeBook` object** (reloaded empty), so interceptors holding it see the revocation at once; `HubIdentity.key_pem`; `NodeIdentity.{key_pem, cert_pem, registered, hub_addresses, forget()}`, `save_registration(reply, *, hub_addresses=())` (any object with the `RegisterReply` fields), and the node dir also keeps `ca.crt` (the pinned CA for later connections); helpers `private_dir`, `write_private`, `new_secret`; regexes `TOKEN_RE`, `NODE_ID_RE`, `NODE_NAME_RE` (names: letters, digits, `_ . -`, ≤ 63). `Token.__repr__` hides the secret. Certificates are back-dated 7 days (clock skew between machines), CA 20 years, node certificates 10 years. `fetch_hub_ca` raises `SecurityError` with a hint for an unreachable / plain hub as well as for a fingerprint mismatch.
7. **`colosseum.net.grpc_util`:** the data-port check is a separate **`DataPortInterceptor(allowed: Callable[[str], bool])`** — a node has no `NodeBook`; its allow-list (the run's registered, not banned nodes) comes from the hub (T3.2). `AuthInterceptor(token_secret, nodes)` serves the hub port and **refuses `NodeData/*` there** (PERMISSION_DENIED); a data port refuses everything but `NodeData`. Added: `KEEPALIVE_MS`, `KEEPALIVE_TIMEOUT_MS`, `OPEN_METHODS`, `TOKEN_METHODS`, `token_metadata(token)`, `node_metadata(node_id, secret)`, `metadata_value(md, key)`, `node_id_of(context)`, `peer_common_name(context)`. `channel_options` / `server_options` keep the contract signatures (keepalive 10 s; servers accept pings every 5 s without calls).
8. **Config:** `transport` is translated by `translate_transport(data)` (module function) with **one** WARNING per load naming everything it did (the overview allowed one per key); `TRANSPORT_IGNORED_KEYS`, `TRANSPORT_RENAMED_KEYS`; `--set transport.*` stays accepted (`LEGACY_OVERRIDE_KEYS`); an unknown key inside `transport` is an error. `DistributedConfig` also checks `node_lost_after_sec > heartbeat_sec` and `learner_relocate_after_sec >= node_lost_after_sec` (as the overview's comments say). `learner.device` is validated against `auto|cpu|cuda|cuda:N|mps`.
9. **`colosseum.node.weights`:** added `CacheLockedError` (a live foreign owner of the prefix: a second node of the same identity and run), `release_node_lock(root, prefix)` (a clean stop removes the prefix's files), `segment_name`, `pointer_name`, `frame_digest`, `KEEP_LATEST`, `READ_RETRIES`, `SPACE_MARGIN_BYTES`. `put` keeps an existing immutable segment as is; a corrupt or mismatched segment reads as `None` with one WARNING per file; a restarted writer adopts the segments its predecessor left (so at most `KEEP_LATEST` stay). `cache_prefix` requires hex ids (run ids are uuid4 hex). Prefix files are recognised by `colosseum-<8 hex>-<8 hex>-n<N>` followed by `-` or `.`.
10. **Test support:** `tests/net_kit.py` provides `loopback_server(credentials=None, *, interceptors=(), handlers=None) -> (server, port)` and `channel(port, credentials=None, server_name=HUB_SERVER_NAME)` instead of the overview's `loopback_server(servicers, credentials)` / `tls_channel(...)`, plus `register(hub, node, ...)` (registration without a server), `echo_handlers()`, `call`, `call_stream`. T2.3's `hub_kit` builds on them.
11. **`scripts/bench_throughput.py::_make_config`** pins every `distributed` key at its default (`heartbeat_sec: 1.0`); T2.5 changes it to `0.25` as the overview says.
12. **T0.1 adds no eviction-reaching-workers test** (already pinned by `tests/contract/test_sp3_snapshot_eviction.py`; cross-task notes). The rotation baseline lives in `tests/contract/test_sp5_rotation_baseline.py` with `owner_sequence(...)` as the reusable expectation.

### Proposed rulings (what — why — cost if wrong)

- **P1-1 — criterion 5 is measured from received bytes to a checked numpy payload, against today's lz4 wire.** Old = `deserialize_chunk_payload` (lz4 npz + JSON, as `GRPCTransport.send_chunk` sends it) + `validate_chunk_payload`; new = `decode_frame` + `chunk_meta` + `check_payload_fields` + `ChunkValidator.check` on a plain frame (what a worker sends to a learner on its machine); PASS iff the ratio of medians ≥ 10. Building torch tensors (`from_payload`, the same work on both sides, not parsing) is reported but not part of the rule; old-plain / new and old-lz4 / new-lz4 are reported too. — Why: the spec's wording is "разбор ... вместе с проверками враждебного ввода и проверкой чанка" against "нынешнего npz+JSON с validate_chunk_payload"; today's wire is lz4; the new learner checks the payload before building the chunk. — Cost if wrong: under the stricter reading "to a built `TrajectoryChunk`" the scratch ratio was 9.1–9.6 (below 10); passing that would need a faster `validate_slot_structure` / `from_payload` (both shared with the old path) — a perf task, no design change.
- **P1-2 — `ChunkValidator` does not check mask legality of recorded actions.** — Why: the worker's legality gate (and the masked sampling of neural policies) already guarantees it, `ActionSpec.first_illegal_action` is the one home of the mask rules, and spec 4.9's list does not ask for it; range checks (which prevent index errors) are kept. — Cost if wrong: a corrupted chunk with an illegal-but-in-range action reaches the loss as a −inf log-probability (NaN); adding the check later costs ≈ 12 µs per 256-slot chunk.
- **P1-3 — the SP4 fixture holds pulse and tournament matches, a pinned snapshot and no `logs/`.** — Why: T4.3 must migrate every SP4 row kind and resume a pinned pool in place; logs hold machine paths (SP3 fixture precedent). — Cost if wrong: an in-place resume test needs `logs/` → T4.3 creates it (the code must anyway: a user may delete logs).
- **P1-4 — the hub's TLS certificate validity is back-dated 7 days; node certificates live 10 years, the CA 20.** — Why: TLS verification compares certificate dates with each verifier's clock, and machine clocks are never compared elsewhere (spec 7.6); revocation is by ban / rotate, not by expiry. — Cost if wrong: a machine whose clock is more than a week behind cannot connect (clear TLS error).
- **P1-5 — one WARNING per config load for the whole `transport` section** (see note 8). — Why: one message says everything that happened; the spec asks for "одним WARNING". — Cost if wrong: none.

### Assumptions about other parts

- **T2.1** re-exports `VersionInfo` from `colosseum.net.messages` (note 4) and reuses `test_sp5_rotation_baseline.owner_sequence` as the expectation of `OwnerRotation` on one node with simultaneous expiry.
- **T2.2** builds the learner's `ChunkValidator(agent_role_spec(spec, roles), agent_id=..., teacher=<scripted teacher>, chunk_length=rollout.chunk_length, state_template=state_to_numpy(model.initial_state(1)))` and decodes with `decode_chunk(buf, cap=payload_byte_cap(distributed.max_message_mb), validator=...)`; a `ChunkError` is counted under its `reason` (`invalid`), never raised out of the intake. Workers encode with `encode_chunk(..., compress=probe.compress())` only on the route connection (overview deviation "lz4 only for frames that leave the machine") and call `probe.observe(*frame_sizes(frame))` when compressed. The learner publishes `encode_weights(..., WeightsMeta(kind="latest", ...))` through `WeightCacheWriter.publish_latest` (R10).
- **T2.3** implements Register with `HubIdentity.nodes.register(...)` → `issue_node_certificate(...)` → `RegisterReply` (`ca_pem` = `HubIdentity.ca_pem`); `SecurityError` from the book becomes `PERMISSION_DENIED` with its message; servers install `AuthInterceptor(lambda: identity.token.secret, identity.nodes)`; blob parts are added to `grpc_util` (`PART_BYTES`, `send_parts`, `receive_parts`).
- **T2.4** acquires the cache lock with `acquire_node_lock(shm_dir(), cache_prefix(ca_fp, run_id, number))` before any segment, runs `cleanup_dead(shm_dir())` at start, reports `shm_free_bytes` from `shutil.disk_usage(shm_dir())`, maps `CacheFullError` to an ERROR with its message and no work, and calls `release_node_lock` on a clean stop.
- **T3.2** serves data ports with `data_server_credentials(node, ca_pem)` + `DataPortInterceptor(allowed)`, where `allowed(cn)` is the node agent's current set of the run's node ids (from `Welcome` / `Announce`), and connects with `data_channel_credentials` + `channel_options(..., server_name=NODE_SERVER_NAME)`.
- **T4.3**: the SP4 fixture's `config.resolved.yaml` has `transport: {mode, grpc_port, grpc_max_message_mb}` — loading it logs the T1.4 WARNING; an in-place resume without `-c` that also passes `--set distributed.max_message_mb=...` would hit "old + new = ConfigError" — T4.3 drops the `transport` key from the saved config before applying `--set` (or documents the conflict). The fixture has no `logs/`.
- **T6.3** builds `join` on `resolve_token`, `fetch_hub_ca`, `NodeIdentity.load_or_create(fp)` / `save_registration` and `hub_channel_credentials`, and `hub token [--rotate]` on `HubIdentity.load_or_create()` / `rotate()` (refused while a hub of this identity runs: overview deviation); `nodes remove --ban` calls `NodeBook.ban`.
- **T7.2** deletes `colosseum.transport.serialization` together with `test_sp5_chunks.py::test_frame_and_npz_formats_decode_to_the_same_content` (listed in its report) and removes the "old" columns of `scripts/measure_frame_decode.py` (the T1.1 result stays in `docs/benchmarks.md`); it also drops the old `transport/colosseum_pb2*.py` ruff excludes.
- **T7.4** copies the T1.1 and T1.5 measurement outputs (both runs) into `docs/benchmarks.md` «SP5».

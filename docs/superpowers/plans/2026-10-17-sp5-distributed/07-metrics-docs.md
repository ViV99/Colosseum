# SP5 Plan — Part 7: Metrics, old roles, `validate`, Docker, measurements, documentation and the acceptance runbook

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Scope.** This part owns the following parts of `docs/superpowers/specs/2026-10-17-sp5-distributed-design.md` (Russian; read the spec first, it is the binding authority):
- **Block 8 (metrics, report, dashboard):** worker statistics keyed by (node, worker) with monotonic counters (the keying itself is T2.5's; T7.1 pins it with tests), the SP5 keys of the `system` record and the extended `REQUIRED_KEYS`, the `train` keys of other tasks (T3.3, T4.1) checked end to end, `runs/<name>/nodes.json`, the console nodes line, the «Nodes» section of `colosseum report` / `colosseum dashboard`, policy lag in the training section, `colosseum hub --dashboard [PORT]` (T7.1).
- **Block 4.10 (what goes):** `serve-weight-store` / `run-learner` / `run-workers` end with exit 2 and the hint; `distributed.py`, `transport/`, `weight_store/`, `proto/colosseum.proto` and their tests are removed (the `transport.*` translation itself is T1.4's); the hub and the node log before their checks (T7.2).
- **Block 9.2 (`validate`):** the placement plan (learner slots by device, preferences, workers, evaluators) and the `distributed:` line (T7.2).
- **Block 10 (Docker; K8s is T8.4):** `deployment/Dockerfile.node` (`cpu` / `cuda`), `deployment/compose.yaml` (hub + 2 nodes), `deployment/compose.node.yaml` (one node for a VM), the `docker` marker and its tests, the old deployment files removed (T7.3).
- **Spec §3 criteria 2–5 and 9 (measurements) and §6 «Slow»:** the hub load generator (criterion 3), the scaling driver with the workload rule (criterion 9), the re-run of the criterion 2 / 4 / 5 scripts on the final tree, the localhost distributed pipeline and the hub load slow tests, `docs/benchmarks.md` section «SP5» (T7.4).
- **Block 11 (documentation):** `docs/DISTRIBUTED_GUIDE.md` (Russian), README, CLAUDE.md, `docs/ARENA_GUIDE.md`, `docs/LEAGUE_GUIDE.md`, and a test that every flag the guides use exists in `--help` (T7.5).
- **Spec §3, every criterion:** the acceptance runbook — exact commands, the thresholds fixed here before any measurement, the owner actions; criteria 8–15 on the owner's two Windows 10 + WSL2 machines, where WSL mirrored networking and the Hyper-V firewall are unavailable, so the hub port is opened with `netsh interface portproxy` + a Windows firewall rule (`research/sp5_node_environment.md` §5) (T7.6).

Not in this part: the `transport.*` translation and the `distributed` config section (T1.4), worker-stat keys by (node, worker, incarnation) (T2.5), `policy_lag_*` / `dropped_stale_chunks` (T3.3), `learner_node` / `generation` (T4.1), the `hub_state.json` counters' save and load (T2.1, T4.3), `join` / `nodes …` / `hub token` (T6.3), SSH mode, agents on the fly, `update-code`, K8s manifests (part 08).

**What the part delivers.** After T7.6:
- every `system` record carries the SP5 cluster keys (`nodes`, `nodes_lost`, `by_node`, `relay_bytes`, `dropped_chunks`, `dropped_match_results_by_reason`, `lineup_fetch_failed`, `worker_restarts`, `learner_restarts`, `pending_agents`, `live_ratings`), `nodes.json` is rewritten on every metrics tick, the console prints one nodes line per tick, the report and the dashboard have a «Nodes» section and a policy-lag chart, and `colosseum hub --dashboard` serves the live page;
- the three SP2 roles answer exit 2 with the hint, and nothing of the old distribution layer is left in the tree; `colosseum validate` prints the placement plan and the `distributed:` line; warnings of the config checks reach `logs/main.log` (hub) and the node's log;
- the node image builds, the compose file trains tic-tac-toe to the budget, the `-m docker` tests skip without Docker;
- `docs/benchmarks.md` «SP5» holds criteria 2–5 measured on the final tree; two slow tests (localhost distributed pipeline, hub load) are in the slow suite;
- `docs/DISTRIBUTED_GUIDE.md` and the updated README / CLAUDE.md / ARENA_GUIDE / LEAGUE_GUIDE describe the real state; a fast test checks the guides' flags against the CLI;
- the runbook below drives the one-machine and the two-machine acceptance.

**Read `00-overview.md` first** (global constraints, file map, the binding interface contract). Names used here are the contract's; additions and deviations are listed in `## Contract notes` at the end.

**Conventions used in every task.**
- Run every command from the repository root as `env -u PYTHONPATH .venv/bin/python ...` / `env -u PYTHONPATH .venv/bin/ruff ...` (owner's machine: ROS sets `PYTHONPATH`; a parallel worktree uses `env -u PYTHONPATH PYTHONPATH=<worktree>/src` with the main checkout's venv).
- "Full fast suite" means `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (zero failures, zero warnings), followed by `env -u PYTHONPATH .venv/bin/ruff check .`.
- New test files start with `test_sp5_` (unique basenames across `tests/`); support modules are imported by bare name; every file a test writes is under `tmp_path` (identity dirs too: the autouse `_isolate_colosseum_dirs` of T1.3 redirects `XDG_DATA_HOME`, `XDG_CACHE_HOME`, `COLOSSEUM_SHM_DIR`); processes start through `tests/cli_runner.py` (`run_in_session` for `hub` / `join` / `tournament --hub`); at most 2 loaded CPU processes per fast test (workers, evaluators, learners; hub and node processes are not counted), at most 3 per slow test; ports come from the code under test binding port 0.
- Data are ordered by counters (`env_steps`, `train_step`, versions, journal row ids), never by `ts`; durations use `time.monotonic()`; no test asserts on wall-clock values or rates. Messages go through `logging` (never `warnings.warn`).
- Commit messages use conventional prefixes and carry no attribution lines; push after every task (`git push`, branch `sp5-distributed`).
- Integration edits name functions and handlers, not lines: everything after T2.5 runs on the hub / node architecture, and earlier tasks of parts 02–06 shape `hub/hub.py`, `node/agent.py` and `cli.py`. Adapt the edit to the code as it is; keep the behaviour and the tests of this part.

**Cross-task notes.**
- **T7.1 ‖ T7.2** run in parallel worktrees after T6.3 (overview, execution order). Shared files and their split: `cli.py` — T7.1 adds only the `--dashboard` option of the `hub` group and its plumbing; T7.2 replaces the three old commands and extends `validate`; `hub/hub.py` — T7.1 edits the metrics tick (`_open_metrics`, the tick in the main loop, `_finish_metrics`), the heartbeat / session bookkeeping it needs and the dashboard child; T7.2 edits only `run_training` / `run_hub` start-up (logging order); `node/agent.py` — T7.1 the heartbeat counters and the final heartbeat, T7.2 the start-up logging; `pyproject.toml` — only T7.2 (ruff excludes) and T7.3 (marker). The controller integrates by cherry-pick with the full fast suite after each.
- **T7.3** after both; **T7.4** after T7.3 (it measures the final code; its slow tests need T7.1's `nodes.json` and the report's «Nodes» section); **T7.5** after T7.4 (docs quote the measurements); **T7.6** is the runbook the controller follows after the final review and fix wave; it changes no code.
- **Quiet-machine work** (one at a time, nothing else CPU-heavy; overview): T7.4's measurements and its slow-suite run. The slow suite is run once at the end of T7.4 (the task adds slow tests and its scripts drive training), in the background, alone.
- **Counters stay monotonic across joins, restarts and in-place resume.** The hub's counters (`hub_state.json` `counters`, T2.1 / T4.3) are the only source of the run-level cluster keys; T7.1 only reads and formats them, and adds per-node counters (`counters["node_counters"]`) that T4.3's save / load carries as an opaque dict.
- **The thresholds of criteria 2, 3, 4, 5, 9, 10 and 11 are fixed in T7.4 and T7.6 below**, before any measurement; a measurement never changes them. A criterion that cannot be checked is reported «не проверено» with the reason; a part 08 criterion that moved to SP5b is «перенесено в SP5b».

**Verified while writing this part.** The pure code of this part was run in a scratch package (`/tmp/claude-1000/-home-viv-dev-repos-Colosseum/b41cd787-f9ea-471b-9664-8578a7f7b4b1/scratchpad/part07/`, a copy of `src/colosseum` at `3022e6a` with this part's edits applied) against the real repo code, with the repo venv (`env -u PYTHONPATH .venv/bin/python`) and the repo's ruff config:
- `colosseum.metrics.nodes` (T7.1) with the `SystemStats.snapshot(cluster=)` / `MetricsHub.maybe_tick(cluster=)` / `REQUIRED_KEYS` edits: rates, rows, `by_node`, the console line and the `dropped_match_results` sum checked by a script; `tests/unit/test_sp2_metrics.py`, `test_sp4_live_ratings.py`, `test_sp4_metric_changes.py`, `test_sp3_share_metrics.py` pass unchanged on the edited copy;
- the report's `_nodes` section, `hardware_text` / `roles_text`, the training section's `lag`, and the new `R.nodes` / lag chart of `page.js`: `node --check` passes; a synthetic run dir (`tests/report_kit.write_run_dir` + a `nodes.json` + `system` records with `by_node` + `train` records with lag) renders in headless Chrome with every section drawn (`data-rendered="1"` × 10) and the expected texts; `tests/unit/test_sp4_report_data.py`, `test_sp4_report_render.py` and `tests/integration/test_sp4_dashboard.py` pass (62 tests) once `report_kit.sample_report_data` carries the `nodes` section and `lag` given in T7.1;
- `dashboard_child` (T7.1): a spawned child binds port 0, reports it over a pipe, answers `/api/nodes`; on a held port it reports the SP4 "port is in use" error;
- `utils.logging` `keep_until_file` / `flush_pending` / `discard_pending` (T7.2): kept records reach the log file first, in order, with original timestamps, INFO and above only; overflow is reported by one line; discarded records never reach a file;
- `describe_distributed` / `describe_plan` (T7.2) on stand-in configs (the `distributed` section exists only after T1.4);
- the doc-flag checker of T7.5 against today's README / ARENA_GUIDE / LEAGUE_GUIDE (43 command lines, no unknown flag) and on a synthetic document (unknown command, unknown flags found);
- the pure rules of `scripts/measure_hub_load.py` and `scripts/measure_scaling.py` (T7.4): percentiles, refit windows, the criterion 3 verdict, the criterion 9 workload rule and verdict, the `system`-record tail: T7.4's `tests/unit/test_sp5_measure_scripts.py` ran on the scratch scripts (17 tests pass; the 4 config-validation tests fail only on the `distributed` section, which exists after T1.4); the hub load config (without `distributed`) validates against today's `ColosseumConfig`;
- T7.3's static deployment tests (6) against the files of T7.3 written to a scratch tree; `sh -n` and the usage / unknown-command exits of the entrypoint;
- click 8.5 behaviour the CLI edits rely on: an option with `is_flag=False, flag_value=8765` parses `--dashboard`, `--dashboard 9000`, `--dashboard=0` and leaves the `hub token` subcommand alone; hidden commands with `ignore_unknown_options` / `allow_extra_args` and `ctx.exit(2)` exit 2 for any arguments (`--help` included) and are absent from `colosseum --help`.

Not verifiable before the earlier parts exist (so given as behaviour, interface and test lists): everything that drives `Hub`, `NodeAgent`, the protobuf session, the cluster kits, the bootstrap loader and Docker. Docker itself is not usable on this machine (Docker Desktop's WSL integration is off: `docker info` fails), so T7.3's image and compose run only where Docker works.

**Traceability (spec item → task → test).**

| Spec | Item | Task | Test(s) |
|---|---|---|---|
| 8 | worker stats by (node, worker), monotonic across joins / restarts | T7.1 (T2.5 keys) | `test_sp5_node_metrics.py::test_worker_counters_stay_monotonic_across_incarnations_and_nodes`; `test_sp5_metrics_runs.py::test_worker_restart_keeps_dropped_counters_monotonic` |
| 8 | `system` keys, `REQUIRED_KEYS` extended | T7.1 | `test_sp5_node_metrics.py::test_required_system_keys_include_every_cluster_key`, `::test_system_snapshot_merges_cluster_keys_and_sums_every_drop`; `test_sp5_metrics_runs.py::test_train_run_writes_cluster_keys_nodes_json_and_the_nodes_line` |
| 8 | `train` keys `policy_lag_*`, `dropped_stale_chunks`, `learner_node`, `generation` | T3.3, T4.1 (checked in T7.1) | `test_sp5_metrics_runs.py::test_train_records_carry_lag_learner_node_and_generation` |
| 8 | `nodes.json` each metrics tick | T7.1 | `test_sp5_node_metrics.py::test_node_rows_*`; `test_sp5_metrics_runs.py::test_two_node_cluster_reports_both_nodes_and_a_lost_one` |
| 8 | «Nodes» section; policy lag in «Training» | T7.1 | `test_sp5_report_nodes.py` (all) |
| 8 | `hub --dashboard [PORT]` (8765, 127.0.0.1) | T7.1 | `test_sp5_hub_dashboard.py` (all) |
| 8 | console nodes line | T7.1 | `test_sp5_node_metrics.py::test_metrics_hub_prints_the_nodes_line_only_with_a_cluster`; `test_sp5_metrics_runs.py::test_train_run_writes_cluster_keys_nodes_json_and_the_nodes_line` |
| 4.10 | old roles exit 2 with the hint | T7.2 | `test_sp5_old_roles.py::test_removed_command_exits_2_with_the_hint` |
| 4.10 | old modules removed | T7.2 | `test_sp5_old_roles.py::test_old_distribution_modules_are_gone`, `::test_no_source_imports_a_removed_module` |
| 4.10 | `transport.*` translation | T1.4 | (T1.4's tests) |
| 4.10 | logging before config checks (hub, node) | T7.2 | `test_sp5_logging_before_checks.py` (all) |
| 9.2 | `validate` placement plan, `distributed:` line | T7.2 | `test_sp5_validate_plan.py` (all) |
| 10 | node image `cpu` / `cuda`, uv, loader | T7.3 | `test_sp5_deployment_files.py::test_node_dockerfile_*`; `test_sp5_docker_slow.py::test_node_image_builds_and_runs_the_loader` |
| 10 | compose hub + 2 nodes to the budget; compose.node.yaml | T7.3 | `test_sp5_deployment_files.py::test_compose_*`; `test_sp5_docker_slow.py::test_compose_trains_to_the_budget` |
| 10 | old deployment files removed | T7.3 | `test_sp5_deployment_files.py::test_old_deployment_files_are_gone` |
| 11 | DISTRIBUTED_GUIDE and the other docs | T7.5 | `test_sp5_docs.py` (all) |
| §6 Slow | localhost distributed pipeline (league, anchors, pulse on an evaluator, `tournament --hub`, report «Nodes») | T7.4 | `test_sp5_distributed_pipeline_slow.py::test_two_node_league_pulse_tournament_and_report` |
| §6 Slow | hub load 20 synthetic nodes × 16 workers | T7.4 | `test_sp5_hub_load_slow.py::test_hub_takes_the_load_of_twenty_synthetic_nodes` |
| §3.2 | one-machine speed ≥ 95 % | T7.4 (measure), T7.6 | `scripts/bench_throughput.py` |
| §3.3 | hub load thresholds | T7.4, T7.6 | `scripts/measure_hub_load.py`; `test_sp5_measure_scripts.py` (rules) |
| §3.4 | one fetch per (machine, agent, version); 50 MB read p95 ≤ 100 ms | T7.4 (T1.5's script, T3.1's test), T7.6 | `scripts/measure_weight_cache.py`; `nodes.json` `weight_fetches` |
| §3.5 | frame decode ≥ 10× | T7.4 (T1.1's script), T7.6 | `scripts/measure_frame_decode.py` |
| §3.9 | two-machine scaling rule and ≥ 0.85 | T7.4 (driver), T7.6 | `scripts/measure_scaling.py`; `test_sp5_measure_scripts.py` (rules) |
| §3.1–§3.15 | the acceptance procedure | T7.6 | runbook |

---

### Task T7.1: Cluster metrics, `nodes.json`, the console nodes line, the «Nodes» section, `hub --dashboard`

Spec block 8 (every bullet) and 9.2 (`hub --dashboard [PORT]`). The hub turns what it already knows about its nodes (registry entries, sessions' cumulative env steps, results accepted per node, the last heartbeat of each node, the learner placement, its run-level counters) into one row per node at every metrics tick. The rows go into `runs/<name>/nodes.json` (rewritten atomically each tick and at the end) and, condensed, into the `system` record's `by_node`; the `system` record gains every SP5 key of the spec (the run-level counters come from the hub's `hub_state.json` counters, so they stay monotonic across joins, restarts and `--resume`); `REQUIRED_KEYS["system"]` is extended; the console prints one nodes line per tick. Nodes add counters to their heartbeats (results sent, results dropped by the buffer, network fetches of latest weights and the distinct versions they brought), so the acceptance can check "sent = journaled + counted losses" per node and "one fetch per (machine, agent, version)" (criteria 4 and 10). The report and the dashboard get a «Nodes» section (rows, totals, per-node series) and a policy-lag chart in «Training». `colosseum hub --dashboard [PORT]` runs the SP4 dashboard server in a child process of the hub (127.0.0.1, default port 8765).

**Split point (if one session is not enough):** dispatch (a) Steps 1–7 (metrics library, `system` keys, heartbeat counters, the hub tick, `nodes.json`, console line; their tests), (b) Steps 8–13 (report section, page, dashboard child, `hub --dashboard`; their tests).

**Files:**
- Create: `src/colosseum/metrics/nodes.py` (pure: `NodeView`, `NodeRates`, `node_rows`, `by_node`, `nodes_payload`, `cluster_record`, `cluster_defaults`, `CLUSTER_KEYS`, `NODES_FILE`, `nodes_line`, `weight_fetches`, `DROPPED_CHUNK_REASONS`, `DROPPED_RESULT_REASONS`)
- Modify: `src/colosseum/metrics/aggregator.py` (`SystemStats.snapshot(..., cluster=)`, the `dropped_match_results` sum)
- Modify: `src/colosseum/metrics/jsonl.py` (`REQUIRED_KEYS["system"]`)
- Modify: `src/colosseum/metrics/console.py` (`ConsoleReporter.nodes_line`)
- Modify: `src/colosseum/metrics/hub.py` (`MetricsHub.maybe_tick(..., cluster=)`, `close(..., cluster=)`, the nodes line)
- Modify: `src/colosseum/net/proto/colosseum_v1.proto` (`Heartbeat.counters`) and the generated `colosseum_v1_pb2.py` / `colosseum_v1_pb2_grpc.py` (regenerated with `scripts/gen_proto.sh`)
- Modify: `src/colosseum/node/agent.py` (heartbeat counters; the final heartbeat after the final result flush; the fetch counters where T3.1 fetches latest weights)
- Modify: `src/colosseum/hub/hub.py` (heartbeat / session bookkeeping per node, results accepted per node, `_node_views`, `_cluster_snapshot`, the metrics tick and `_finish_metrics` pass `cluster=`; `nodes.json`; the dashboard child: `_start_dashboard`, `_stop_dashboard`)
- Modify: `src/colosseum/arena/report/data.py` (`SECTIONS`, `LAG_KEYS`, `NODE_SERIES`, `_nodes`, `_training` `lag`, `hardware_text`, `roles_text`; module docstring's section list)
- Modify: `src/colosseum/arena/report/assets/page.html`, `src/colosseum/arena/report/assets/page.js` (section, `R.nodes`, lag chart)
- Modify: `src/colosseum/arena/report/server.py` (`dashboard_child`)
- Modify: `src/colosseum/cli.py` (the `--dashboard` option of the `hub` group only)
- Modify (test support): `tests/report_kit.py` (`sample_report_data`: a `nodes` section and `training.lag`)
- Test: `tests/unit/test_sp5_node_metrics.py`, `tests/unit/test_sp5_report_nodes.py`, `tests/unit/test_sp5_node_counters.py`, `tests/integration/test_sp5_metrics_runs.py`, `tests/integration/test_sp5_hub_dashboard.py`

**Interfaces:**
- Consumes: `SystemStats` keyed by (node, worker, incarnation) (T2.5); `MetricsHub`, `MetricsWriter`, `write_json_atomic` (SP4); `Hub` and its main loop, `_open_metrics`, `_finish_metrics`, the metrics tick, `_on_session_opened`, the heartbeat handler, R8 commit accounting (T2.5); `NodeRegistry` / `NodeEntry` (T2.1); `BudgetCounter.state()["sessions"]` (T2.1); the hub's run-level counters (`HubStateData.counters`, `nodes_lost`; T2.1 / T2.5 / T3.x / T4.x); `PlacementPlan.assigned` / `pending` (T4.1); `LiveRatings.latest()` (SP4); `pb.Heartbeat`, `pb.ProcState` (T1.2); `NodeAgent` heartbeat and stop paths (T2.4, T4.4); the node's network fetch of latest weights (T3.1); `make_server` (SP4); `watch_parent` (T4.4); `ReportBuilder`, `MetricsTail`, `thin`, `_sorted_by`, `_finite` (SP4); `cluster_kit`, `fault_proxy`, `node_kit.FakeHub` (T3.1, T3.2, T2.4); `cli_runner.run_train`, `start_hub` (T2.5).
- Produces: `colosseum.metrics.nodes` (above); `SystemStats.snapshot(env_steps, queue_depths, *, match_log=None, cluster=None)`; `MetricsHub.maybe_tick(..., cluster: Callable[[], tuple[dict, list[dict]]] | None = None)` and `close(..., cluster=None)`; `ConsoleReporter.nodes_line(rows, nodes_lost) -> str`; `pb.Heartbeat.counters` (`map<string, int64>`; node counter names below); `runs/<name>/nodes.json`; the `system` keys of `CLUSTER_KEYS` (spec block 8 plus `live_ratings`); `arena.report.data.SECTIONS` with `"nodes"` after `"system"`, `hardware_text`, `roles_text`, `LAG_KEYS`, `NODE_SERIES`; the training section's `lag`; `arena.report.server.dashboard_child(run_dir, port, conn)`; `Hub._start_dashboard()`, `Hub._stop_dashboard()`; CLI `colosseum hub ... --dashboard [PORT]`.

**Node counters** (`Heartbeat.counters`, cumulative over the node process, which is one session; names fixed here):

| Name | Meaning | Incremented where |
|---|---|---|
| `results_sent` | results the node put into a `ResultBatch` for the first time (a resend does not count again) | the node's result batcher (T2.4) |
| `results_dropped_buffer` | results dropped because the unacked buffer was full (`node_result_buffer_mb`) | the same overflow branch that counts them today (T2.4) |
| `weight_fetches/<agent>` | network downloads of the agent's latest weights (from a data port or through the hub), retries included | the node's fetch of latest versions (T3.1) |
| `weight_versions/<agent>` | distinct versions of the agent the node published into its cache from such downloads | right after the publish of a fetched version (T3.1) |

The hub keeps the last `counters` per `(node_id, session_id)` and sums them per node; it saves them in `hub_state.json` under `counters["node_counters"]` (`{"<node_id>/<session_id>": {name: value}}`) and the results it accepted per node under `counters["node_results"]` (`{"<node number>": n}`), so the per-node sums survive `--resume`.

- [ ] **Step 1: Write the failing tests of the metrics library**

Create `tests/unit/test_sp5_node_metrics.py`:
```python
"""Cluster metrics of the hub (SP5 T7.1, spec block 8): per-node rates and rows, nodes.json, the SP5 keys of the
system record, the sum of every dropped result, the console nodes line, and monotonic worker counters keyed by
(node, worker, incarnation). Pure: no processes."""
from __future__ import annotations

import json
import logging
from dataclasses import replace

import pytest

from colosseum.metrics.aggregator import SystemStats
from colosseum.metrics.hub import MetricsHub
from colosseum.metrics.jsonl import REQUIRED_KEYS, MetricsWriter
from colosseum.metrics.nodes import (
    CLUSTER_KEYS,
    DROPPED_CHUNK_REASONS,
    DROPPED_RESULT_REASONS,
    NodeRates,
    NodeView,
    by_node,
    cluster_defaults,
    cluster_record,
    node_rows,
    nodes_line,
    nodes_payload,
    weight_fetches,
)


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _view(number: int, *, state: str = "online", env_steps: int = 0, results: int = 0, **kw) -> NodeView:
    base = dict(number=number, name=f"n{number}", node_id=f"{number:032x}", state=state,
                hardware={"logical_cores": 8, "physical_cores": 4}, tags=(), data_path="direct", code_hash="c0de",
                last_heartbeat_age_sec=0.5, workers=3, evaluators=1, learners=(), load=0.5, env_steps=env_steps,
                results=results, counters={})
    base.update(kw)
    return NodeView(**base)


def _record(rows, **kw):
    args = dict(nodes_lost=0, relay_bytes=0, dropped_chunks={}, dropped_results={}, lineup_fetch_failed=0,
                worker_restarts=0, learner_restarts=0, pending_agents=[])
    args.update(kw)
    return cluster_record(rows, **args)


def test_rates_are_zero_on_first_sight_then_per_second_on_the_hub_clock():
    clock = Clock()
    rates = NodeRates(clock=clock)
    assert rates.update([_view(1, env_steps=100, results=10)]) == {1: (0.0, 0.0)}
    clock.t = 2.0
    assert rates.update([_view(1, env_steps=4100, results=210)]) == {1: (2000.0, 100.0)}


def test_rates_of_a_lost_node_a_shrinking_counter_and_a_zero_interval_are_zero():
    clock = Clock()
    rates = NodeRates(clock=clock)
    rates.update([_view(1, env_steps=1000), _view(2, env_steps=1000)])
    clock.t = 1.0
    out = rates.update([_view(1, env_steps=500), _view(2, state="lost", env_steps=5000)])
    assert out == {1: (0.0, 0.0), 2: (0.0, 0.0)}
    assert rates.update([_view(1, env_steps=900)]) == {1: (0.0, 0.0)}       # same instant


def test_node_rows_are_sorted_and_carry_roles_results_and_weight_fetches():
    views = [_view(2, learners=("b", "a"), counters={"results_sent": 7, "results_dropped_buffer": 1,
                                                       "weight_fetches/main": 3, "weight_versions/main": 3}),
             _view(1, data_path="relay")]
    rows = node_rows(views, {1: (10.04, 1.0), 2: (0.0, 0.0)})
    assert [r["number"] for r in rows] == [1, 2]
    assert rows[0]["env_steps_per_sec"] == 10.0 and rows[0]["data_path"] == "relay"
    assert rows[1]["roles"] == {"workers": 3, "evaluators": 1, "learners": ["a", "b"]}
    assert rows[1]["results_sent"] == 7 and rows[1]["results_dropped"] == 1
    assert rows[1]["weight_fetches"] == {"main": {"fetches": 3, "versions": 3}}
    assert json.loads(json.dumps(nodes_payload(42, rows))) == {"env_steps": 42, "nodes": rows}


def test_weight_fetches_pairs_downloads_with_distinct_versions():
    assert weight_fetches({"weight_fetches/a": 4, "weight_versions/a": 3, "weight_versions/b": 1, "x": 9}) == \
        {"a": {"fetches": 4, "versions": 3}, "b": {"fetches": 0, "versions": 1}}


def test_by_node_keys_are_node_numbers_as_strings():
    rows = node_rows([_view(10), _view(2, state="lost")], {})
    cells = by_node(rows)
    assert list(cells) == ["2", "10"]
    assert cells["2"] == {"name": "n2", "env_steps_per_sec": 0.0, "results_per_sec": 0.0, "load": 0.5,
                          "last_heartbeat_age_sec": 0.5, "data_path": "direct", "workers": 3, "state": "lost"}


def test_cluster_record_counts_online_nodes_and_keeps_every_reason():
    rows = node_rows([_view(1), _view(2, state="lost")], {})
    record = _record(rows, nodes_lost=3, dropped_chunks={"disconnect": 2, "future_reason": 1},
                     dropped_results={"invalid": 4}, pending_agents=["z", "a"])
    assert record["nodes"] == 1 and record["nodes_lost"] == 3 and record["pending_agents"] == ["a", "z"]
    assert record["dropped_chunks"] == {**{r: 0 for r in DROPPED_CHUNK_REASONS}, "disconnect": 2, "future_reason": 1}
    assert record["dropped_match_results_by_reason"] == {**{r: 0 for r in DROPPED_RESULT_REASONS}, "invalid": 4}
    assert set(record) == set(CLUSTER_KEYS)


def test_cluster_defaults_are_fresh_copies():
    first = cluster_defaults()
    first["dropped_chunks"]["stale"] = 5
    assert cluster_defaults()["dropped_chunks"]["stale"] == 0


def test_required_system_keys_include_every_cluster_key():
    assert set(CLUSTER_KEYS) <= REQUIRED_KEYS["system"]


def test_system_snapshot_merges_cluster_keys_and_sums_every_drop():
    stats = SystemStats(clock=Clock())
    stats.on_worker_stats({"worker_id": 0, "node": 1, "incarnation": 0, "dropped_results": 3})
    plain = stats.snapshot(10, {}, match_log={"dropped": 2})
    assert plain["dropped_match_results"] == 5 and plain["nodes"] == 0 and plain["by_node"] == {}
    record = _record([], nodes_lost=2, dropped_results={"invalid": 4, "node_buffer": 1})
    snap = stats.snapshot(10, {}, match_log={"dropped": 2}, cluster=record)
    assert snap["dropped_match_results"] == 3 + 2 + 4 + 1 and snap["nodes_lost"] == 2
    assert not (REQUIRED_KEYS["system"] - {"ts", "kind"} - set(snap))


def test_worker_counters_stay_monotonic_across_incarnations_and_nodes():
    """T2.5 keys worker stats by (node, worker, incarnation): a restarted worker (new incarnation, counters from 0)
    and the same worker number on another node add up; nothing goes down."""
    stats = SystemStats(clock=Clock())
    stats.on_worker_stats({"worker_id": 0, "node": 1, "incarnation": 0, "dropped_reward_episodes": 5})
    first = stats.snapshot(1, {})["dropped_reward_episodes"]
    stats.on_worker_stats({"worker_id": 0, "node": 1, "incarnation": 1, "dropped_reward_episodes": 1})
    stats.on_worker_stats({"worker_id": 0, "node": 2, "incarnation": 0, "dropped_reward_episodes": 2})
    stats.on_worker_stats({"worker_id": 0, "node": 1, "incarnation": 0, "dropped_reward_episodes": 4})   # stale
    assert first == 5 and stats.snapshot(2, {})["dropped_reward_episodes"] == 5 + 1 + 2


def test_nodes_line_counts_online_nodes_the_run_losses_and_online_rates():
    rows = node_rows([_view(1), _view(2), _view(3, state="lost")], {1: (1200.4, 1.0), 2: (34.0, 0.0), 3: (9.0, 0.0)})
    assert nodes_line(rows, nodes_lost=4) == "nodes: 2 online, 4 lost, 1,234 env steps/s"
    assert nodes_line([], nodes_lost=0) == "nodes: 0 online, 0 lost, 0 env steps/s"


@pytest.fixture
def progress_lines():
    lines: list[str] = []

    class Grab(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    log = logging.getLogger("colosseum.progress")
    handler, level = Grab(), log.level
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    yield lines
    log.removeHandler(handler)
    log.setLevel(level)


def _hub(tmp_path, clock):
    return MetricsHub(writer=MetricsWriter(tmp_path / "metrics.jsonl"), ratings_path=tmp_path / "ratings.json",
                      agent_ids=["main"], total_timesteps=100, log_interval=1, console_interval_sec=1.0, clock=clock)


def test_metrics_hub_prints_the_nodes_line_only_with_a_cluster(tmp_path, progress_lines):
    clock = Clock()
    hub = _hub(tmp_path, clock)
    clock.t = 5.0
    assert hub.maybe_tick(env_steps=5, ratings={}, queue_depths={})
    assert not any(line.startswith("nodes:") for line in progress_lines)
    rows = node_rows([_view(1)], {1: (99.0, 1.0)})
    clock.t = 10.0
    assert hub.maybe_tick(env_steps=9, ratings={}, queue_depths={},
                          cluster=lambda: (_record(rows, nodes_lost=1), rows))
    assert progress_lines[-1] == "nodes: 1 online, 1 lost, 99 env steps/s"
    systems = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
               if json.loads(line)["kind"] == "system"]
    assert systems[0]["nodes"] == 0 and systems[1]["nodes"] == 1 and systems[1]["by_node"]["1"]["state"] == "online"


def test_the_cluster_callable_runs_only_when_the_tick_is_due_and_at_close(tmp_path):
    clock = Clock()
    hub = _hub(tmp_path, clock)
    calls = []

    def cluster():
        calls.append(clock.t)
        return _record([]), []

    clock.t = 0.5
    assert not hub.maybe_tick(env_steps=1, ratings={}, queue_depths={}, cluster=cluster)
    clock.t = 1.5
    assert hub.maybe_tick(env_steps=2, ratings={}, queue_depths={}, cluster=cluster)
    hub.close(env_steps=3, ratings={}, queue_depths={}, cluster=cluster)
    assert calls == [1.5, 1.5]


def test_replace_keeps_node_views_immutable():
    view = _view(1)
    assert replace(view, env_steps=5).env_steps == 5 and view.env_steps == 0
```

- [ ] **Step 2: Run them to verify they fail**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_node_metrics.py -q -rw`
Expected: collection error `ModuleNotFoundError: No module named 'colosseum.metrics.nodes'`.

- [ ] **Step 3: Write `src/colosseum/metrics/nodes.py`**

```python
"""Cluster metrics of the hub (SP5 spec block 8): per-node rows, rates, ``nodes.json``, the SP5 keys of the
``system`` record and the console nodes line. Pure: the hub builds one ``NodeView`` per known node at a metrics
tick (``Hub._node_views``) and passes counters it owns; nothing here reads the network, the clock of another
machine or the run dir.

Rules:
- rates are computed on the hub's monotonic clock from cumulative counters (``NodeRates``): a node's env steps
  (the sum over its sessions, as the hub's budget counts them) and its results accepted at intake; a counter that
  goes down (a new session that the hub has not summed yet) gives rate 0 for that interval, never a negative;
- a node that is not online keeps its row (state ``lost`` / ``removed``) with rates 0, so the report shows it;
- ``by_node`` keys are node numbers as strings (JSON object keys), sorted numerically;
- the console line counts online nodes, the run's cumulative ``nodes_lost`` and the online nodes' env steps/s.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

NODES_FILE = "nodes.json"
DROPPED_CHUNK_REASONS = ("disconnect", "no_learner", "stale", "stale_generation", "invalid", "overflow")
DROPPED_RESULT_REASONS = ("node_buffer", "invalid", "stale_generation", "writer")
# The SP5 keys of every ``system`` record (spec block 8) and their values when the hub has nothing to report
# (a MetricsHub without a hub, tests): ``cluster_defaults()`` returns a fresh copy.
_CLUSTER_DEFAULTS: dict[str, Any] = {
    "nodes": 0,
    "nodes_lost": 0,
    "by_node": {},
    "relay_bytes": 0,
    "dropped_chunks": {r: 0 for r in DROPPED_CHUNK_REASONS},
    "dropped_match_results_by_reason": {r: 0 for r in DROPPED_RESULT_REASONS},
    "lineup_fetch_failed": 0,
    "worker_restarts": 0,
    "learner_restarts": 0,
    "pending_agents": [],
    "live_ratings": {"version": None, "refit_sec": None, "entities": 0},
}
CLUSTER_KEYS = tuple(_CLUSTER_DEFAULTS)


def cluster_defaults() -> dict[str, Any]:
    return copy.deepcopy(_CLUSTER_DEFAULTS)


@dataclass(frozen=True)
class NodeView:
    """What the hub knows about one node at a metrics tick."""

    number: int
    name: str
    node_id: str
    state: str                                   # online | lost | removed
    hardware: Mapping[str, Any]
    tags: tuple[str, ...]
    data_path: str                               # direct | relay | local
    code_hash: str
    last_heartbeat_age_sec: float | None         # hub monotonic now - last heartbeat; None: never
    workers: int                                 # worker processes running on the node
    evaluators: int                              # evaluator processes running on the node
    learners: tuple[str, ...]                    # agents whose learner the hub placed on the node
    load: float | None                           # Heartbeat.load
    env_steps: int                               # cumulative env steps of the node (all its sessions)
    results: int                                 # cumulative results the hub accepted from the node
    counters: Mapping[str, int] = field(default_factory=dict)   # Heartbeat.counters (node-side cumulative)


class NodeRates:
    """Per-node env steps/s and results/s between two ``update`` calls, on the hub's monotonic clock."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last: dict[int, tuple[float, int, int]] = {}

    def update(self, views: Sequence[NodeView]) -> dict[int, tuple[float, float]]:
        """``{number: (env_steps_per_sec, results_per_sec)}``; (0, 0) for a node seen for the first time, for a
        node that is not online and for an interval shorter than 1 ms."""
        now = self._clock()
        out: dict[int, tuple[float, float]] = {}
        for view in views:
            last = self._last.get(view.number)
            self._last[view.number] = (now, int(view.env_steps), int(view.results))
            if last is None or view.state != "online":
                out[view.number] = (0.0, 0.0)
                continue
            dt = now - last[0]
            if dt < 1e-3:
                out[view.number] = (0.0, 0.0)
                continue
            out[view.number] = (max(0, int(view.env_steps) - last[1]) / dt, max(0, int(view.results) - last[2]) / dt)
        return out


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def weight_fetches(counters: Mapping[str, int]) -> dict[str, dict[str, int]]:
    """``{agent: {"fetches": n, "versions": m}}`` from the node counters ``weight_fetches/<agent>`` (network
    downloads of latest weights) and ``weight_versions/<agent>`` (distinct versions the node published into its cache
    from such downloads); criterion 4 is ``fetches == versions`` on every machine whatever its worker count."""
    out: dict[str, dict[str, int]] = {}
    for key, value in sorted(counters.items()):
        kind, _, agent = key.partition("/")
        if agent and kind in ("weight_fetches", "weight_versions"):
            out.setdefault(agent, {"fetches": 0, "versions": 0})["fetches" if kind == "weight_fetches" else
                                                                "versions"] = int(value)
    return out


def node_rows(views: Sequence[NodeView], rates: Mapping[int, tuple[float, float]]) -> list[dict[str, Any]]:
    """The ``nodes`` rows of ``nodes.json``, by node number."""
    rows = []
    for view in sorted(views, key=lambda v: v.number):
        env_rate, result_rate = rates.get(view.number, (0.0, 0.0))
        counters = dict(view.counters)
        rows.append({
            "number": int(view.number),
            "name": view.name,
            "node_id": view.node_id,
            "state": view.state,
            "hardware": dict(view.hardware),
            "roles": {"workers": int(view.workers), "evaluators": int(view.evaluators),
                      "learners": sorted(view.learners)},
            "env_steps_per_sec": _round(env_rate, 1),
            "results_per_sec": _round(result_rate, 1),
            "load": _round(view.load),
            "last_heartbeat_age_sec": _round(view.last_heartbeat_age_sec),
            "data_path": view.data_path,
            "tags": list(view.tags),
            "code_hash": view.code_hash,
            "env_steps": int(view.env_steps),
            "results_journaled": int(view.results),
            "results_sent": int(counters.get("results_sent", 0)),
            "results_dropped": int(counters.get("results_dropped_buffer", 0)),
            "weight_fetches": weight_fetches(counters),
        })
    return rows


def by_node(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """The ``system`` record's ``by_node`` (spec block 8)."""
    keys = ("name", "env_steps_per_sec", "results_per_sec", "load", "last_heartbeat_age_sec", "data_path")
    out = {}
    for row in sorted(rows, key=lambda r: r["number"]):
        cell = {k: row[k] for k in keys}
        cell["workers"] = row["roles"]["workers"]
        cell["state"] = row["state"]
        out[str(row["number"])] = cell
    return out


def nodes_payload(env_steps: int, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``runs/<name>/nodes.json``."""
    return {"env_steps": int(env_steps), "nodes": [dict(r) for r in rows]}


def cluster_record(rows: Sequence[Mapping[str, Any]], *, nodes_lost: int, relay_bytes: int,
                   dropped_chunks: Mapping[str, int], dropped_results: Mapping[str, int], lineup_fetch_failed: int,
                   worker_restarts: int, learner_restarts: int, pending_agents: Sequence[str],
                   live_ratings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The SP5 keys of a ``system`` record. Unknown reasons are kept (a newer node may count more); missing ones
    are 0."""
    record = cluster_defaults()
    record.update({
        "nodes": sum(1 for r in rows if r["state"] == "online"),
        "nodes_lost": int(nodes_lost),
        "by_node": by_node(rows),
        "relay_bytes": int(relay_bytes),
        "lineup_fetch_failed": int(lineup_fetch_failed),
        "worker_restarts": int(worker_restarts),
        "learner_restarts": int(learner_restarts),
        "pending_agents": sorted(pending_agents),
    })
    record["dropped_chunks"].update({k: int(v) for k, v in dropped_chunks.items()})
    record["dropped_match_results_by_reason"].update({k: int(v) for k, v in dropped_results.items()})
    if live_ratings is not None:
        record["live_ratings"].update(dict(live_ratings))
    return record


def nodes_line(rows: Sequence[Mapping[str, Any]], nodes_lost: int) -> str:
    """``nodes: 2 online, 1 lost, 12,345 env steps/s`` (the online nodes' env steps/s; ``lost``: cumulative)."""
    online = [r for r in rows if r["state"] == "online"]
    rate = sum(float(r["env_steps_per_sec"] or 0.0) for r in online)
    return f"nodes: {len(online)} online, {int(nodes_lost)} lost, {rate:,.0f} env steps/s"
```

- [ ] **Step 4: `system` keys, the drop sum, the console line**

`src/colosseum/metrics/jsonl.py`, `REQUIRED_KEYS["system"]` gains the cluster keys (keep the old ones):
```python
    "system": {"ts", "kind", "env_steps", "env_steps_per_sec", "train_steps_per_sec", "queue_depths",
               "parked_buffers", "dropped_reward_episodes", "dropped_match_results", "workers_reporting",
               # SP5 (spec block 8; ``metrics.nodes.CLUSTER_KEYS``)
               "nodes", "nodes_lost", "by_node", "relay_bytes", "dropped_chunks", "dropped_match_results_by_reason",
               "lineup_fetch_failed", "worker_restarts", "learner_restarts", "pending_agents", "live_ratings"},
```

`src/colosseum/metrics/aggregator.py`, `SystemStats.snapshot` (import `Mapping` from `collections.abc`): new keyword `cluster`, merged after the SP4 keys; `dropped_match_results` becomes the sum of every counted loss (the workers' own drops, the hub's by reason, the writer's queue drops; under R8 the last is normally 0):
```python
    def snapshot(self, env_steps: int, queue_depths: dict[str, int], *,
                 match_log: dict[str, Any] | None = None, cluster: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """``match_log``: the journal writer's ``WriterStats.as_dict()`` (None: no journal); ``cluster``: the hub's SP5
        keys (``metrics.nodes.cluster_record``; None: their neutral values, ``cluster_defaults``)."""
        from colosseum.metrics.nodes import cluster_defaults
        ...                                       # SP4 body unchanged up to the dict, without "dropped_match_results"
            "match_log": None if match_log is None else dict(match_log),
            "workers_reporting": len(self._workers),
        }
        extra = cluster_defaults()
        if cluster is not None:
            extra.update({k: v for k, v in cluster.items() if k in extra})
        snap.update(extra)
        # every lost result once: the workers' own drops, the hub's by reason, and the journal writer's queue drops
        snap["dropped_match_results"] = (self.cumulative("dropped_results")
                                         + sum(int(v) for v in extra["dropped_match_results_by_reason"].values())
                                         + int((match_log or {}).get("dropped", 0)))
        ...                                       # the baseline updates as before
```

`src/colosseum/metrics/console.py`, `ConsoleReporter`:
```python
    def nodes_line(self, rows: list[dict], nodes_lost: int) -> str:
        """The hub's line of nodes (SP5 spec block 8): ``metrics.nodes.nodes_line``."""
        from colosseum.metrics.nodes import nodes_line

        return nodes_line(rows, nodes_lost)
```

`src/colosseum/metrics/hub.py`, `MetricsHub.maybe_tick` and `close` gain `cluster: Callable[[], tuple[dict, list[dict]]] | None = None` (docstring: "the hub's cluster snapshot, ``(cluster_record, node rows)``, called only when the tick is due, like ``ratings``"); in `maybe_tick`:
```python
        record, rows = cluster() if cluster is not None else (None, [])
        system = self._system.snapshot(env_steps, queue_depths, match_log=match_log, cluster=record)
        ...
        self._report_console(int(env_steps), system, ratings, summary)
        if cluster is not None:
            self._console.emit([self._console.nodes_line(rows, int(system["nodes_lost"]))])
        return True
```
and `close` forwards `cluster=cluster` to its forced tick. WandB keeps flattening the `system` record (numbers only: `system/by_node/<n>/env_steps_per_sec` etc., bounded by the node count).

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_node_metrics.py tests/unit/test_sp2_metrics.py tests/unit/test_sp4_live_ratings.py tests/unit/test_sp4_metric_changes.py -q -rw`
Expected: all pass (the SP2 / SP4 metrics tests pass unchanged: `REQUIRED_KEYS` is a lower bound of a record's keys, and every `SystemStats` snapshot now carries the cluster keys with neutral values).

- [ ] **Step 5: Heartbeat counters (proto and node)**

`src/colosseum/net/proto/colosseum_v1.proto`: add to `message Heartbeat` the next free field number, `map<string, int64> counters = <N>;` (comment: "node-side cumulative counters of this session: results_sent, results_dropped_buffer, weight_fetches/<agent>, weight_versions/<agent>"); regenerate with `scripts/gen_proto.sh` and commit the generated files (T1.2's rule). A peer without the field sends an empty map (protobuf compatibility); the hub treats a missing name as 0.

`src/colosseum/node/agent.py` (edit points by function; adapt to the code as T2.4–T6.3 left it):
- the node keeps one `collections.Counter` of the names above; the result batcher adds a batch's size to `results_sent` when the batch is built the first time (not on a resend after a reconnect); the buffer-overflow branch adds the dropped count to `results_dropped_buffer` (it already counts them for `Heartbeat.dropped_results`); the latest-weights fetch (T3.1) adds 1 to `weight_fetches/<agent>` per download attempt that transferred a frame, and 1 to `weight_versions/<agent>` after it published a version not yet in the cache;
- every `Heartbeat` carries `counters` (a copy of the counter);
- at stop (budget, `stop_run`, Ctrl+C of the node), after the final result flush and before the session closes, the node sends one more `Heartbeat` (the final one), so the hub's last view of `results_sent` covers every result it was sent (T4.4's stop order: stop children → flush results → wait for acks within the grace → **final heartbeat** → close).

Create `tests/unit/test_sp5_node_counters.py` (node with `node_kit.FakeHub`, one real worker child at most):
- `test_heartbeat_counters_survive_the_wire`: a `pb.Heartbeat(counters={"results_sent": 3, "weight_fetches/main": 1})` serialized and parsed back keeps both entries; a `Heartbeat` built without `counters` parses with an empty map.
- `test_node_heartbeat_counts_sent_results_once_and_ends_with_a_final_heartbeat`: FakeHub scripts a Welcome with one agent and a non-collecting lineup for a 1-worker node (tic-tac-toe TINY settings from `node_kit.toy_node_options`); FakeHub drops the session once after the first acked batch (the node reconnects and resends the unacked batches); the node is stopped by a `Command(kind="stop_run")` after FakeHub has received ≥ 3 result batches. Assert: the last message FakeHub recorded from the node is a `Heartbeat`; its `counters["results_sent"]` equals the number of distinct results (by uid) FakeHub received; `results_dropped_buffer` is 0.
- `test_buffer_overflow_counts_results_dropped_buffer`: the node with `distributed.node_result_buffer_mb` set so small that two batches overflow it and FakeHub never acks: `counters["results_dropped_buffer"]` equals `Heartbeat.dropped_results` and is > 0.
- `test_one_remote_fetch_per_version_is_counted`: FakeHub announces versions 1, 2, 3 of agent `main` from another node and serves them (`GetLatest` on the FakeHub relay path or a FakeNode data port, as T3.1's tests do) to a node with 2 workers that both play `main`: `weight_fetches/main == weight_versions/main == 3` (the criterion 4 counter; 2 loaded processes).

- [ ] **Step 6: The hub's cluster snapshot and `nodes.json`**

`src/colosseum/hub/hub.py` (edit points by function; the names in parentheses are the contract's or this task's):
- **Bookkeeping per node.** In `_on_session_opened` store the `Hello.code_hash` per node id (`self._node_code_hash`). In the handler of `Heartbeat` messages store, per node id, the last `procs`, `load` and `counters` (`self._node_heartbeat[node_id] = (procs, load, dict(counters))`) and the counters also per `(node_id, session_id)` in `self._counters["node_counters"]["<node_id>/<session_id>"]`. In the R8 commit accounting (where each newly journaled result is accounted; and at intake when the journal is off), add 1 to `self._counters["node_results"][str(result.node)]` for every result of a node's `ResultBatch` (training results; evaluator block results are not counted here, as `results_sent` does not count them either).
- **`_node_views(now) -> list[NodeView]`**: one view per `NodeRegistry` entry the hub knows this run (online, lost or removed). `workers` / `evaluators` = the number of `procs` of the node's last heartbeat named `worker-*` / `evaluator-*` with state `running` (0 before the first heartbeat); `learners` = the agents whose current `LearnerSlot.node` is this node (T4.1's plan); `env_steps` = the sum over the node's sessions of `BudgetCounter.state()["sessions"]` values (keys `<node_id>/<session_id>`); `results` = `node_results[str(number)]`; `counters` = the sum over the node's sessions of `node_counters`; `last_heartbeat_age_sec` = `now - entry.last_heartbeat`; `code_hash` from `_node_code_hash` (empty string if unknown).
- **`_cluster_snapshot() -> tuple[dict, list[dict]]`**: `views = self._node_views(time.monotonic())`; `rows = node_rows(views, self._node_rates.update(views))` (`self._node_rates = NodeRates()` built in `_open_metrics`); `write_json_atomic(run_dir.root / NODES_FILE, nodes_payload(self.env_steps_done, rows))` (an `OSError` is one WARNING per run and never stops the run); returns `(cluster_record(rows, nodes_lost=..., relay_bytes=..., dropped_chunks=..., dropped_results=..., lineup_fetch_failed=..., worker_restarts=..., learner_restarts=..., pending_agents=..., live_ratings=self._live_ratings_info()), rows)` with every counter taken from the hub's run-level counters (`hub_state.json` `counters` / `nodes_lost`), `pending_agents` from the placement plan's `pending`, and `_live_ratings_info()` = `{"version": s.version, "refit_sec": s.refit_sec, "entities": sum(len(f.rating) for f in s.fits.values()) + sum(len(x.entities) for x in s.scores.values())}` of `LiveRatings.latest()` (all None / 0 without live ratings).
- **The metrics tick** (main loop deadline) and `_finish_metrics` pass `cluster=self._cluster_snapshot` to `MetricsHub.maybe_tick` / `close`; so `nodes.json` is written on every tick and once more at the end, after the nodes' final heartbeats (the final `close` runs after the stop sequence of T4.4).
- `train`'s local node is a node like any other: one row, `data_path` `local`.

- [ ] **Step 7: Integration tests of the cluster metrics**

Create `tests/integration/test_sp5_metrics_runs.py`:
- `test_train_run_writes_cluster_keys_nodes_json_and_the_nodes_line`: `cli_runner.run_train(TTT_CONFIG, tmp_path)` (TINY, 1 worker). Every `system` record has `REQUIRED_KEYS["system"]`; the last one has `nodes == 1`, `by_node["1"]["data_path"] == "local"`, `nodes_lost == 0`, `dropped_match_results == 0`; `nodes.json` exists with one row: `roles == {"workers": 1, "evaluators": 0, "learners": ["agent_0"]}`, `env_steps > 0`, `results_sent == results_journaled + results_dropped`, and `results_journaled` equals the journal's train rows (`MatchLog.open_readonly`, `source="train"`); `logs/main.log` contains a line matching `nodes: 1 online, 0 lost, [\d,]+ env steps/s`.
- `test_train_records_carry_lag_learner_node_and_generation`: the same run's `train` records carry `policy_lag_p50`, `policy_lag_p95`, `policy_lag_max`, `dropped_stale_chunks` (T3.3) and `learner_node == 1`, `generation >= 1` (T4.1).
- `test_two_node_cluster_reports_both_nodes_and_a_lost_one`: `cluster_kit` hub (TLS, `--listen 127.0.0.1 --port 0`, `distributed.node_lost_after_sec=2`, `heartbeat_sec=0.5`, `learner_relocate_after_sec=600`) with node A (`--workers 0 --cpu-learners 1 --evaluators 0`) and node B (`--workers 1 --no-learners --evaluators 0`, connected through `fault_proxy.TcpFaultProxy`); 2 loaded processes. Wait (by counters) until `nodes.json` has two online rows with `env_steps_per_sec > 0` for B; `proxy.cut(8)`; wait until a `system` record has `nodes == 1`, `nodes_lost == 1` and `by_node[B]["state"] == "lost"`; `main.log` has a line `nodes: 1 online, 1 lost`; after the cut ends, wait until B is online again with the same number (`nodes.json`). Stop the hub (SIGINT) → exit 130.
- `test_worker_restart_keeps_dropped_counters_monotonic`: the same cluster without the proxy; `SIGKILL` B's worker (pid from `worker-0.log`'s "started (pid N)" line in B's log dir); wait for `worker_restarts == 1` in a `system` record; over all `system` records ordered by `env_steps`, `dropped_reward_episodes`, `dropped_match_results`, `worker_restarts` and every value of `dropped_chunks` never decrease.

Mark a test `slow` if it runs longer than ~20 s on the owner's machine (global rule); the first two share one module-scoped run.

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_node_counters.py tests/integration/test_sp5_metrics_runs.py -q -rw`
Expected: all pass.

- [ ] **Step 8: Write the failing report tests**

Create `tests/unit/test_sp5_report_nodes.py`:
```python
"""The «Nodes» section and the policy-lag chart of the run report and the dashboard (SP5 T7.1, spec block 8):
rows from nodes.json with hardware and roles texts, totals from the newest system record, per-node series from the
records' by_node, a note for runs without nodes.json (SP4 and earlier) or with a broken one, and the page drawing
every section in a browser."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import urllib.request

import pytest

from colosseum.arena.report.data import SECTIONS, build_report_data, hardware_text, roles_text
from colosseum.arena.report.render import render_html
from colosseum.arena.report.server import make_server
from colosseum.metrics.nodes import NODES_FILE, NodeView, cluster_record, node_rows, nodes_payload
from report_kit import sample_report_data, write_run_dir

CHROME = next((shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser") if shutil.which(n)), None)
BIG = 10**9          # counters above the synthetic run's own records (data are ordered by counters)


def _views():
    pc = NodeView(1, "pc", "a" * 32, "online",
                  {"logical_cores": 20, "physical_cores": 10, "memory_bytes": 26 * 2**30, "os": "Linux",
                   "platform": "WSL2", "gpus": [{"index": 0, "name": "RTX 4070 Ti", "memory_bytes": 12 * 2**30,
                                                 "kind": "cuda"}]},
                  ("gpu",), "local", "c0de", 0.4, 3, 0, ("main",), 1.5, 5000, 300, {"results_sent": 300})
    laptop = NodeView(2, "laptop", "b" * 32, "lost", {"logical_cores": 8}, (), "relay", "c0de", 14.0, 4, 1, (), None,
                      500, 40, {"results_sent": 41, "results_dropped_buffer": 1})
    return [pc, laptop]


def _add_cluster(root, *, lag: bool = True) -> list[dict]:
    rows = node_rows(_views(), {1: (2000.0, 100.0), 2: (0.0, 0.0)})
    (root / NODES_FILE).write_text(json.dumps(nodes_payload(BIG, rows)))
    record = cluster_record(rows, nodes_lost=1, relay_bytes=3 * 2**20, dropped_chunks={"disconnect": 2},
                            dropped_results={}, lineup_fetch_failed=0, worker_restarts=1, learner_restarts=0,
                            pending_agents=["second"])
    lines = [json.dumps({"ts": 0, "kind": "system", "env_steps": BIG + 1000 * i, "env_steps_per_sec": 100.0, **record})
             for i in range(1, 6)]
    if lag:
        lines += [json.dumps({"ts": 0, "kind": "train", "agent": "main", "train_step": BIG + i, "total_loss": 1.0,
                              "policy_lag_p50": 1.0, "policy_lag_p95": 2.0 + i, "policy_lag_max": 5.0})
                  for i in range(5)]
    with (root / "metrics.jsonl").open("a") as fh:
        fh.write("\n".join(lines) + "\n")
    return rows


def test_nodes_is_a_section_after_system():
    assert SECTIONS.index("nodes") == SECTIONS.index("system") + 1


def test_a_run_without_nodes_json_has_no_rows_and_a_note(tmp_path):
    run = write_run_dir(tmp_path, "wdl")
    data = build_report_data(run.root)
    nodes = data["sections"]["nodes"]
    assert nodes["nodes"] == [] and nodes["series"] == {} and "No nodes.json" in nodes["notes"][0]
    assert data["sections"]["training"]["lag"] == {}


def test_a_broken_nodes_json_is_a_note_not_a_failure(tmp_path):
    run = write_run_dir(tmp_path, "wdl")
    (run.root / NODES_FILE).write_text("{not json")
    nodes = build_report_data(run.root)["sections"]["nodes"]
    assert nodes["nodes"] == [] and "cannot be read" in nodes["notes"][0]


def test_nodes_section_rows_totals_and_series(tmp_path):
    run = write_run_dir(tmp_path, "wdl")
    _add_cluster(run.root)
    data = build_report_data(run.root)
    nodes = data["sections"]["nodes"]
    assert [n["name"] for n in nodes["nodes"]] == ["pc", "laptop"] and nodes["env_steps"] == BIG
    pc, laptop = nodes["nodes"]
    assert pc["hardware_text"] == "20 cores (10 physical), 26 GB, RTX 4070 Ti 12 GB, Linux / WSL2"
    assert pc["roles_text"] == "3 workers, learner main" and laptop["roles_text"] == "4 workers, 1 evaluator"
    assert laptop["state"] == "lost" and laptop["data_path"] == "relay" and laptop["results_dropped"] == 1
    totals = nodes["totals"]
    assert (totals["online"], totals["known"], totals["nodes_lost"], totals["env_steps_per_sec"]) == (1, 2, 1, 2000.0)
    assert totals["relay_bytes"] == 3 * 2**20 and totals["dropped_chunks"]["disconnect"] == 2
    assert totals["pending_agents"] == ["second"]
    assert set(nodes["series"]) == {"1", "2"} and nodes["series"]["1"]["env_steps_per_sec"][-1] == 2000.0
    lag = data["sections"]["training"]["lag"]["main"]
    assert lag["p95"][-1] == 6.0 and lag["max"][-1] == 5.0 and len(lag["train_step"]) == len(lag["p50"])


@pytest.mark.parametrize("hw, text", [
    ({}, "-"),
    ({"logical_cores": 4, "physical_cores": 4}, "4 cores"),
    ({"memory_bytes": 512 * 2**20, "gpus": [{"kind": "mps"}]}, "512 MB, mps"),
])
def test_hardware_text(hw, text):
    assert hardware_text(hw) == text


@pytest.mark.parametrize("roles, text", [
    ({}, "idle"),
    ({"workers": 1, "evaluators": 2, "learners": ["a", "b"]}, "1 worker, 2 evaluators, learners a, b"),
])
def test_roles_text(roles, text):
    assert roles_text(roles) == text


def test_the_dashboard_serves_the_nodes_section(tmp_path):
    run = write_run_dir(tmp_path, "wdl")
    _add_cluster(run.root)
    server = make_server(run.root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{server.port}/api/nodes", timeout=30).read())
    finally:
        server.shutdown()
        server.server_close()
    assert [n["name"] for n in body["nodes"]] == ["pc", "laptop"]


def _dom(tmp_path, data: dict) -> str:
    page = tmp_path / "report.html"
    page.write_text(render_html(data))
    proc = subprocess.run([CHROME, "--headless=new", "--no-sandbox", "--disable-gpu", "--dump-dom",
                           "--virtual-time-budget=3000", page.as_uri()], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0 or not proc.stdout.strip():
        pytest.fail(f"headless Chrome failed (exit code {proc.returncode}): {proc.stderr[-2000:]}")
    return proc.stdout


def _visible(dom: str) -> str:
    return re.sub(r"<[^>]+>", " ", dom.split('<script type="application/json"')[0])


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome")
def test_the_nodes_section_and_the_lag_chart_are_drawn(tmp_path):
    run = write_run_dir(tmp_path, "wdl")
    _add_cluster(run.root)
    dom = _dom(tmp_path, build_report_data(run.root))
    assert dom.count('data-rendered="1"') == len(SECTIONS) and "could not be drawn:" not in dom.split("<script")[0]
    text = _visible(dom)
    for needle in ("RTX 4070 Ti", "learner main", "relay", "14.0 s ago", "Policy lag", "pending", "disconnect 2",
                   "env steps/s by node"):
        assert needle in text, needle


@pytest.mark.skipif(CHROME is None, reason="no headless Chrome")
def test_the_sample_report_draws_every_section(tmp_path):
    dom = _dom(tmp_path, sample_report_data())
    assert dom.count('data-rendered="1"') == len(SECTIONS)
    assert "laptop" in _visible(dom)
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_report_nodes.py -q -rw`
Expected: FAIL at import (`cannot import name 'hardware_text'`).

- [ ] **Step 9: The report data (`src/colosseum/arena/report/data.py`)**

Module docstring: the section list gains `nodes` after `system`. Constants:
```python
SECTIONS = ("summary", "ratings", "matrix", "anchors", "opponents", "training", "system", "nodes", "matches",
            "tournaments")
...
LAG_KEYS = {"p50": "policy_lag_p50", "p95": "policy_lag_p95", "max": "policy_lag_max"}   # train record keys (SP5)
NODE_SERIES = ("env_steps_per_sec", "results_per_sec")          # per-node series of the system records' by_node
```

`ReportBuilder._training`: before its `return`, the per-agent lag series (same thinned records as the metrics):
```python
        lag = {}
        for agent, a in agents.items():
            if any(key in a["metrics"] for key in LAG_KEYS.values()):
                lag[agent] = {"train_step": a["train_step"],
                              **{name: a["metrics"].get(key, [None] * len(a["train_step"]))
                                 for name, key in LAG_KEYS.items()}}
        return {"agents": agents, "default": default, "lag": lag, "notes": notes}
```

`ReportBuilder._nodes` (new, after `_system`):
```python
    # ----- nodes -----
    def _nodes(self) -> dict:
        """The machines of the run (SP5 spec block 8): the rows of ``nodes.json`` (the hub's last metrics tick) with
        a hardware and a roles text, totals from the newest ``system`` record and each node's env steps/s and
        results/s over env steps from the records' ``by_node``. A run without ``nodes.json`` (SP4 and earlier, one
        machine) has no rows and a note."""
        from colosseum.metrics.nodes import NODES_FILE

        notes: list[str] = []
        path = self.run_dir / NODES_FILE
        payload = None
        if path.exists():
            try:
                payload = json.loads(path.read_text())
                if not isinstance(payload, Mapping) or not isinstance(payload.get("nodes"), list):
                    raise ValueError("not a nodes.json object")
            except (OSError, ValueError) as e:
                notes.append(f"{NODES_FILE} cannot be read ({type(e).__name__}: {e}).")
                payload = None
        else:
            notes.append(f"No {NODES_FILE}: a run of SP4 or earlier (one machine), or a hub that has not written its "
                         f"first metrics tick yet.")
        rows = []
        for raw in (payload or {}).get("nodes", []):
            if not isinstance(raw, Mapping):
                continue
            row = dict(raw)
            row["hardware_text"] = hardware_text(row.get("hardware") or {})
            row["roles_text"] = roles_text(row.get("roles") or {})
            rows.append(row)
        rows.sort(key=lambda r: (r.get("number") is None, r.get("number")))
        recs = _sorted_by(self.records("system"), "env_steps")
        latest = recs[-1] if recs else {}
        idx = thin(len(recs), self.limits.series_points)
        series: dict[str, dict[str, Any]] = {}
        for i in idx:
            rec = recs[i]
            for number, cell in (rec.get("by_node") or {}).items():
                if not isinstance(cell, Mapping):
                    continue
                entry = series.setdefault(str(number), {"name": cell.get("name"), "env_steps": [],
                                                        **{k: [] for k in NODE_SERIES}})
                entry["env_steps"].append(rec.get("env_steps"))
                for k in NODE_SERIES:
                    entry[k].append(_finite(cell.get(k)))
        online = [r for r in rows if r.get("state") == "online"]
        totals = {"online": len(online), "known": len(rows),
                  "nodes_lost": latest.get("nodes_lost"),
                  "env_steps_per_sec": sum(_finite(r.get("env_steps_per_sec")) or 0.0 for r in online),
                  "relay_bytes": latest.get("relay_bytes"),
                  "dropped_chunks": latest.get("dropped_chunks") or {},
                  "pending_agents": latest.get("pending_agents") or []}
        return {"env_steps": (payload or {}).get("env_steps"), "nodes": rows, "series": dict(sorted(series.items(),
                key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)), "totals": totals, "notes": notes}
```

Module-level helpers (before `_add`):
```python
def _gb(n: Any) -> str | None:
    f = _finite(n)
    return None if f is None else f"{f / 2**30:.0f} GB" if f >= 2**30 else f"{f / 2**20:.0f} MB"


def hardware_text(hw: Mapping[str, Any]) -> str:
    """One line of a node's hardware report: cores, memory, GPUs, OS (missing parts are left out)."""
    parts = []
    logical, physical = hw.get("logical_cores"), hw.get("physical_cores")
    if logical is not None:
        parts.append(f"{logical} cores" + (f" ({physical} physical)" if physical not in (None, logical) else ""))
    if (mem := _gb(hw.get("memory_bytes"))) is not None:
        parts.append(mem)
    for gpu in hw.get("gpus") or []:
        if isinstance(gpu, Mapping):
            mem = _gb(gpu.get("memory_bytes"))
            parts.append(f"{gpu.get('name') or gpu.get('kind') or 'GPU'}" + (f" {mem}" if mem else ""))
    system = " / ".join(str(hw[k]) for k in ("os", "platform") if hw.get(k))
    if system:
        parts.append(system)
    return ", ".join(parts) or "-"


def roles_text(roles: Mapping[str, Any]) -> str:
    """``2 workers, 1 evaluator, learner main`` (what the node runs)."""
    parts = []
    for key, word in (("workers", "worker"), ("evaluators", "evaluator")):
        n = int(roles.get(key) or 0)
        if n:
            parts.append(f"{n} {word}{'' if n == 1 else 's'}")
    learners = list(roles.get("learners") or [])
    if learners:
        parts.append(("learner " if len(learners) == 1 else "learners ") + ", ".join(map(str, learners)))
    return ", ".join(parts) or "idle"
```

The dashboard needs no change for the section itself: `server.py` serves every name of `SECTIONS` at `/api/<section>` (no filters for `nodes`).

- [ ] **Step 10: The page (`assets/page.html`, `assets/page.js`)**

`page.html`, after the system section:
```html
  <section id="sec-nodes" class="sec"><h2>Nodes</h2><div class="body"></div></section>
```

`page.js`: `ORDER` gains `"nodes"` after `"system"` and `TITLES` gains `nodes: "Nodes"`. In `R.training`, right after the controls line:
```js
    const lag = (s.lag || {})[A];
    if (lag) {
      out.push(h("h3", {}, "Policy lag (versions behind the learner, per batch)"));
      out.push(lineChart({ series: ["p50", "p95", "max"].map((k) => ({ name: "lag " + k, pts: lag.train_step.map((x, i) => [x, lag[k][i]]).filter((p) => p[1] !== null && p[1] !== undefined) })), xlabel: "train step", height: 180, label: "policy lag", digits: 1 }));
    }
```
After `R.system`:
```js
  R.nodes = function (s) {
    const rows = s.nodes || [], t = s.totals || {}, out = [];
    if (!rows.length) return [h("p", { class: "note" }, "No machines recorded for this run.")].concat(notes(s.notes));
    const mb = (b) => (typeof b === "number" ? num(b / 1048576, 1) + " MB" : "-");
    const drops = Object.entries(t.dropped_chunks || {}).filter(([, v]) => v);
    out.push(h("div", { class: "tiles" },
      tile("Online", num(t.online, 0) + " of " + num(t.known, 0)),
      tile("Lost (run)", num(t.nodes_lost, 0)),
      tile("Env steps/s (online)", num(t.env_steps_per_sec, 0)),
      tile("Relayed by the hub", mb(t.relay_bytes)),
      tile("Dropped chunks", drops.length ? drops.map(([k, v]) => k + " " + num(v, 0)).join(", ") : "0")));
    if ((t.pending_agents || []).length) out.push(h("div", { class: "flag" }, "Agents waiting for a learner slot (pending): " + t.pending_agents.join(", ")));
    const DOT = { online: "good", lost: "critical", removed: "warning" };
    out.push(table("nodes", [
      { key: "number", label: "#", num: true },
      { key: "name", label: "node" },
      { key: "state", label: "state", cell: (r) => h("span", { class: "status" }, h("i", { class: "dot " + (DOT[r.state] || ""), "aria-hidden": "true" }), r.state || "-") },
      { key: "hardware_text", label: "hardware" },
      { key: "roles_text", label: "roles" },
      { key: "load", label: "load", num: true, fmt: (v) => num(v, 2) },
      { key: "env_steps_per_sec", label: "env steps/s", num: true, fmt: (v) => num(v, 0) },
      { key: "results_per_sec", label: "results/s", num: true, fmt: (v) => num(v, 0) },
      { key: "last_heartbeat_age_sec", label: "last heartbeat", num: true, fmt: (v) => (typeof v === "number" ? num(v, 1) + " s ago" : "-") },
      { key: "data_path", label: "data path", title: "how chunks reach this node's learners: direct (its data port) or relay (through the hub port)" },
      { key: "tags", label: "tags", text: (r) => (r.tags || []).join(", ") || "-" },
    ], rows, { key: "nodes", filter: rows.length > 8, sort: { key: "number", dir: "asc" } }));
    const numbers = Object.keys(s.series || {});
    if (numbers.length) {
      for (const [key, label] of [["env_steps_per_sec", "env steps/s"], ["results_per_sec", "results/s"]]) {
        out.push(h("h3", {}, label + " by node"));
        out.push(lineChart({ series: numbers.map((n) => { const d = s.series[n]; return { name: "#" + n + " " + (d.name || ""), pts: d.env_steps.map((x, i) => [x, d[key][i]]).filter((p) => p[1] !== null && p[1] !== undefined) }; }), xlabel: "env steps", height: 180, label: label + " by node", digits: 0 }));
      }
    }
    return out.concat(notes(s.notes));
  };
```
"last heartbeat … s ago" is the hub's own monotonic age at its last tick (never a comparison of clocks of two machines).

`tests/report_kit.py`, `sample_report_data()`: add, before `"matches"`, a `nodes` section and a `lag` entry in `training` (the render tests count `data-rendered="1"` against `len(SECTIONS)`):
```python
            "nodes": {"notes": [], "env_steps": steps[-1], "nodes": [
                {"number": 1, "name": "pc", "node_id": "a" * 32, "state": "online",
                 "hardware": {"logical_cores": 20, "physical_cores": 10, "memory_bytes": 26 * 2**30,
                              "gpus": [{"index": 0, "name": "RTX 4070 Ti", "memory_bytes": 12 * 2**30, "kind": "cuda"}],
                              "os": "Linux", "platform": "WSL2"},
                 "hardware_text": "20 cores (10 physical), 26 GB, RTX 4070 Ti 12 GB, Linux / WSL2",
                 "roles": {"workers": 9, "evaluators": 0, "learners": ["main"]}, "roles_text": "9 workers, learner main",
                 "env_steps_per_sec": 1200.0, "results_per_sec": 150.0, "load": 0.8, "last_heartbeat_age_sec": 0.4,
                 "data_path": "local", "tags": ["gpu"], "code_hash": "c0de", "env_steps": 150000,
                 "results_journaled": 20000, "results_sent": 20000, "results_dropped": 0, "weight_fetches": {}},
                {"number": 2, "name": "laptop", "node_id": "b" * 32, "state": "lost",
                 "hardware": {"logical_cores": 8, "physical_cores": 4}, "hardware_text": "8 cores (4 physical)",
                 "roles": {"workers": 3, "evaluators": 1, "learners": []}, "roles_text": "3 workers, 1 evaluator",
                 "env_steps_per_sec": 0.0, "results_per_sec": 0.0, "load": None, "last_heartbeat_age_sec": 14.0,
                 "data_path": "relay", "tags": [], "code_hash": "c0de", "env_steps": 50000,
                 "results_journaled": 7000, "results_sent": 7000, "results_dropped": 0,
                 "weight_fetches": {"main": {"fetches": 12, "versions": 12}}}],
                "series": {"1": {"name": "pc", "env_steps": steps, "env_steps_per_sec": [1200.0] * len(steps),
                                 "results_per_sec": [150.0] * len(steps)},
                           "2": {"name": "laptop", "env_steps": steps, "env_steps_per_sec": [400.0] * len(steps),
                                 "results_per_sec": [50.0] * len(steps)}},
                "totals": {"online": 1, "known": 2, "nodes_lost": 1, "env_steps_per_sec": 1200.0,
                           "relay_bytes": 3 * 2**20, "dropped_chunks": {"disconnect": 2}, "pending_agents": []}},
```
and in `"training"`, next to `"agents"`:
```python
                "lag": {"main": {"train_step": list(range(0, 1000, 10)), "p50": [1.0] * 100, "p95": [3.0] * 100,
                                 "max": [6.0] * 100}}},
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_report_nodes.py tests/unit/test_sp4_report_data.py tests/unit/test_sp4_report_render.py tests/integration/test_sp4_dashboard.py tests/integration/test_sp4_report_cli.py -q -rw`
Expected: all pass (`node --check` of `page.js` runs in `test_page_js_is_valid_javascript`).

- [ ] **Step 11: `hub --dashboard [PORT]`**

Create `tests/integration/test_sp5_hub_dashboard.py` (no loaded process: a hub without nodes):
- `test_hub_dashboard_serves_the_run_and_dies_with_the_hub`: `cli_runner.start_hub(TTT_CONFIG, tmp_path, "dash", args=["--listen", "127.0.0.1", "--port", "0", "--dashboard", "0"])`; wait (≤ 30 s) for a stdout line `Dashboard at http://127.0.0.1:<port>/`; `GET /api/nodes` answers 200 JSON with a `nodes` list; `GET /` contains `id="sec-nodes"`; SIGINT the hub → exit code 130; within 10 s a connect to `<port>` is refused (the child is gone).
- `test_hub_dashboard_without_a_port_uses_8765`: `colosseum hub --help` lists `--dashboard` with "8765" and "127.0.0.1" in its help; the click option's `flag_value` is 8765 (in-process `click` introspection of `colosseum.cli.main.commands["hub"]`). (No bind of 8765 in tests.)
- `test_hub_dashboard_on_a_taken_port_stops_the_hub_with_exit_1`: hold a listening socket on `127.0.0.1:<p>` (bound to port 0 by the test), `colosseum hub ... --dashboard <p>` → exit 1, stderr names the port and "in use" (SP4's bind message), and `run_state.json` says `failed`.
- `test_dashboard_child_reports_its_port_or_the_bind_error`: in-process `multiprocessing` (spawn) `dashboard_child` on a synthetic run dir (`report_kit.write_run_dir`): port 0 → `("ok", port)` and `/api/nodes` answers; a held port → `("error", "... in use ...")`.

`src/colosseum/arena/report/server.py` (append):
```python
def dashboard_child(run_dir: str, port: int, conn) -> None:
    """Target of the dashboard process ``colosseum hub --dashboard`` spawns (SP5 spec block 8): binds 127.0.0.1,
    sends ``("ok", port)`` or ``("error", message)`` over ``conn``, then serves until killed. It only reads the run
    dir, so its builds never hold the hub's GIL; it dies with the hub (``watch_parent``)."""
    import logging as _logging

    from colosseum.node.procs import watch_parent
    from colosseum.utils.logging import setup_process_logging

    setup_process_logging(Path(run_dir) / "logs", "dashboard")
    watch_parent()
    try:
        server = make_server(run_dir, host="127.0.0.1", port=int(port))
    except ConfigError as e:
        conn.send(("error", str(e)))
        conn.close()
        return
    conn.send(("ok", server.port))
    conn.close()
    _logging.getLogger(__name__).info(f"dashboard of {run_dir} at http://127.0.0.1:{server.port}/")
    try:
        server.serve_forever()
    finally:
        server.server_close()
```

`src/colosseum/hub/hub.py`:
- `_start_dashboard()` — when `HubOptions.dashboard_port is not None`, after the run dir and `config.resolved.yaml` exist and before nodes are accepted: `ctx = multiprocessing.get_context("spawn")`; `parent, child = ctx.Pipe(duplex=False)`; `proc = ctx.Process(target=dashboard_child, args=(str(run_dir.root), port, child), name="dashboard", daemon=True)`; `proc.start()`; `child.close()`; wait for the message with `parent.poll(30)`. `("ok", p)`: log INFO and print `Dashboard at http://127.0.0.1:<p>/ (no authentication; from another machine use an SSH tunnel: docs/ARENA_GUIDE.md)`. `("error", msg)` or no answer in 30 s: raise `ConfigError(f"--dashboard: {msg}")` (exit 1 through the normal start-up error path; nothing else of the run started yet).
- the main loop checks the child each metrics tick: a dead dashboard child is one WARNING (`the dashboard process exited with code N; the run continues without it (colosseum dashboard RUN serves it again)`), never a stop.
- `_stop_dashboard()` — called at the very end of `_shutdown` (after the final metrics): `terminate()`, `join(2)`, `kill()` if alive. The child also dies with the hub (`watch_parent`).

`src/colosseum/cli.py`, the `hub` group (T2.5) gains (only this option; T7.2 edits other parts of `cli.py`):
```python
@click.option("--dashboard", "dashboard", type=click.IntRange(0, 65535), is_flag=False, flag_value=8765, default=None,
              help="Also serve the live report (colosseum dashboard) on 127.0.0.1:PORT (default 8765; 0: any free "
                   "port); no authentication: use an SSH tunnel from other machines")
```
passed to `run_hub(..., dashboard=dashboard)` → `HubOptions.dashboard_port`. Help text names the flag as `--dashboard [PORT]` in the group's docstring.

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp5_hub_dashboard.py -q -rw`
Expected: all pass.

- [ ] **Step 12: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings. Tests that compare console output line by line (none known; grep `"\[agent_0\] step"` in `tests/`) must tolerate the extra nodes line.

- [ ] **Step 13: Commit and push**

```bash
git add src/colosseum/metrics src/colosseum/net/proto src/colosseum/node/agent.py src/colosseum/hub/hub.py \
        src/colosseum/arena/report src/colosseum/cli.py tests/report_kit.py \
        tests/unit/test_sp5_node_metrics.py tests/unit/test_sp5_report_nodes.py tests/unit/test_sp5_node_counters.py \
        tests/integration/test_sp5_metrics_runs.py tests/integration/test_sp5_hub_dashboard.py
git commit -m "feat: cluster metrics, nodes.json, nodes console line, Nodes report section and hub --dashboard"
git push
```

---

### Task T7.2: Old roles removed, old modules and tests removed, `validate` placement plan and `distributed:` line, logging before checks

Spec block 4.10 and 9.2 (`validate`). The SP2 roles answer exit 2 with the hint "use `colosseum hub` / `join` / `nodes`" (hidden commands that accept any arguments, so a script that still calls them gets the hint, not a click usage error). `distributed.py`, `transport/` (with its SP1 npz codec: T1.1 moved `_check_state_payload` to the frame codec, and its hostile-input tests are ported here where T1.1 has no equivalent), `weight_store/` and `proto/colosseum.proto` are deleted with their tests. `colosseum validate` (train scope) prints the `distributed:` line and the `placement:` lines. The hub and the node start logging before their checks: records of the config checks (translation warnings of `transport.*` and of the SP2 knobs, validation lines) are kept in memory until the log file exists and then written at its top; a node that stops before its first Welcome writes them to `join.log`.

**Files:**
- Delete: `src/colosseum/distributed.py`, `src/colosseum/transport/` (whole package), `src/colosseum/weight_store/` (whole package), `proto/colosseum.proto` (and the empty `proto/` dir)
- Delete (tests whose subject is removed): `tests/integration/test_sp2_distributed_e2e.py`, `tests/integration/test_sp2_grpc.py`, `tests/unit/test_sp2_distributed_roles.py`, `tests/unit/test_sp3_distributed_guards.py`, `tests/unit/test_sp4_distributed_guards.py`, `tests/unit/test_sp2_serialization.py` (after Step 3's port)
- Modify (remove the distributed cases only): `tests/unit/test_sp2_payloads.py` (`test_in_memory_weight_store_keeps_numpy`), `tests/unit/test_sp2_learner_entry.py` (`test_run_learner_seeds_before_building_the_model`), `tests/unit/test_sp3_learner_factory.py` (`test_the_distributed_learner_builds_through_the_factory`), `tests/unit/test_sp4_snapshot_store.py` (`test_the_distributed_learner_caps_its_pool_fifo_and_refuses_top_k` and the `colosseum.distributed` import), `tests/contract/test_sp4_first_weights.py` (`test_the_distributed_worker_waits_for_the_first_weights`), `tests/unit/test_sp2_entry_points.py` (`test_distributed_roles_reject_the_config_before_serving_or_spawning`), `tests/unit/test_sp2_launcher_lifecycle.py` or its T2.5 port (the `run-learner` / `run-workers` / `serve-weight-store` parametrizations and the commands-set assertion), `tests/unit/test_sp3_dagger_chunks.py` (ported to the frame codec and `ChunkValidator`)
- Modify: `src/colosseum/cli.py` (the three commands → `_removed_command`; `validate` unchanged except that the report now has the new lines), `src/colosseum/core/validation.py` (`validate_config` train scope appends the lines; docstring: "the distributed roles" leave the `scope="train"` list), `src/colosseum/hub/placement.py` (`describe_plan`, `describe_distributed`), `src/colosseum/utils/logging.py` (`keep_until_file`, `PendingRecords`, `flush_pending`, `discard_pending`, `PENDING_LIMIT`), `src/colosseum/hub/hub.py` (`run_training`, `run_hub`: start-up order only), `src/colosseum/node/agent.py` (`NodeAgent.run` start-up logging, `join.log`), `pyproject.toml` (`extend-exclude`: drop the two `transport/colosseum_pb2*.py` lines)
- Possibly delete: `worker.rollout_worker.rollout_worker_process` and `core.ipc` helpers with no remaining reference (Step 9)
- Test: `tests/unit/test_sp5_old_roles.py`, `tests/unit/test_sp5_validate_plan.py`, `tests/unit/test_sp5_pending_log.py`, `tests/integration/test_sp5_logging_before_checks.py`, `tests/unit/test_sp5_ported_serialization.py`

**Interfaces:**
- Consumes: `colosseum.cli.main`; `validate_config` / `ValidationReport` (SP4); `DistributedConfig`, `LearnerConfig.placement` (T1.4); `hub.placement` (T4.1); `run_training` / `run_hub` (T2.5); `NodeAgent.run`, `node.logs.node_log_dir` (T2.4); `security.parse_token`, `Token.ca_fingerprint` (T1.3); `encode_frame` / `decode_frame` / `FrameError` / `payload_byte_cap`, `encode_chunk` / `decode_chunk` / `ChunkValidator` / `ChunkError`, `encode_weights` / `decode_weights` / `WeightsMeta` (T1.1); `node_kit.FakeHub` (T2.4).
- Produces: hidden CLI commands `serve-weight-store`, `run-learner`, `run-workers` (exit 2); `cli.REMOVED_COMMANDS: dict[str, str]`; `hub.placement.describe_plan(config) -> list[str]`, `hub.placement.describe_distributed(config) -> str`; `ValidationReport.lines` with one `distributed:` line and the `placement:` lines (train scope only); `utils.logging.setup_process_logging(..., keep_until_file=False)`, `PendingRecords`, `flush_pending(path, process_name) -> Path | None`, `discard_pending()`, `PENDING_LIMIT = 10_000`; the node's pre-Welcome log `<cache_home>/colosseum/logs/<ca8>/join.log`.

- [ ] **Step 1: Write the failing tests of the removal**

Create `tests/unit/test_sp5_old_roles.py`:
```python
"""SP5 spec block 4.10: the SP2 distributed roles are gone. Their commands exit 2 with the hint to use the hub,
join and nodes; the modules of the old distribution layer do not exist and nothing imports them."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest
from click.testing import CliRunner

from colosseum.cli import REMOVED_COMMANDS, main

REPO = Path(__file__).resolve().parents[2]
REMOVED_MODULES = ("colosseum.distributed", "colosseum.transport", "colosseum.weight_store")


@pytest.mark.parametrize("args", [
    ["serve-weight-store"], ["serve-weight-store", "--port", "50051"], ["serve-weight-store", "--help"],
    ["run-learner", "-c", "cfg.yaml", "--agent", "a", "--traj-port", "1", "--weight-store", "h:1"],
    ["run-workers", "-c", "cfg.yaml", "--weight-store", "h:1", "-l", "a=h:2"], ["run-workers"],
])
def test_removed_command_exits_2_with_the_hint(args):
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 2, result.output
    text = result.output + (result.stderr if result.stderr_bytes is not None else "")
    assert f"'colosseum {args[0]}' was removed in SP5" in text
    for word in ("colosseum hub", "colosseum join", "colosseum nodes", "DISTRIBUTED_GUIDE"):
        assert word in text


def test_removed_commands_are_hidden_from_help():
    text = CliRunner().invoke(main, ["--help"]).output
    assert set(REMOVED_COMMANDS) == {"serve-weight-store", "run-learner", "run-workers"}
    assert not any(name in text for name in REMOVED_COMMANDS) and "train" in text


@pytest.mark.parametrize("module", REMOVED_MODULES)
def test_old_distribution_modules_are_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_the_old_proto_is_gone():
    assert not (REPO / "proto" / "colosseum.proto").exists()


def test_no_source_imports_a_removed_module():
    offenders = []
    for root in ("src", "scripts", "tests", "examples"):
        for path in (REPO / root).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if any(name == m or name.startswith(m + ".") for m in REMOVED_MODULES):
                        offenders.append(f"{path.relative_to(REPO)}:{node.lineno} {name}")
    assert offenders == []
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_old_roles.py -q -rw`
Expected: FAIL (`cannot import name 'REMOVED_COMMANDS'`).

- [ ] **Step 2: Replace the three commands (`src/colosseum/cli.py`)**

Delete `run_learner_cmd`, `run_workers_cmd`, `serve_weight_store_cmd` and add (after the last command, before `if __name__ == "__main__"`):
```python
# SP5 spec block 4.10: the SP2 roles are removed; a script that still calls them gets the hint (exit 2).
REMOVED_COMMANDS = {
    "serve-weight-store": "the weight store is a per-machine cache inside every node now",
    "run-learner": "the hub places every learner on a node",
    "run-workers": "workers run on the nodes that join the hub",
}
_REMOVED_HINT = ("use 'colosseum hub -c cfg.yaml' on the main machine, 'colosseum join ADDR --token-file F' on every "
                 "other machine and 'colosseum nodes list' to see them (docs/DISTRIBUTED_GUIDE.md)")


def _removed_command(name: str, why: str) -> click.Command:
    @click.command(name, hidden=True, add_help_option=False,
                   context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
    @click.pass_context
    def removed(ctx: click.Context) -> None:
        click.echo(f"Error: 'colosseum {name}' was removed in SP5 ({why}); {_REMOVED_HINT}", err=True)
        ctx.exit(2)

    removed.__doc__ = f"Removed in SP5: {why}."
    return removed


for _name, _why in REMOVED_COMMANDS.items():
    main.add_command(_removed_command(_name, _why))
```
(Verified with click 8.5: exit code 2 for every argument list, `--help` included; the commands are absent from `colosseum --help`.) The README's "Process lifecycle" list of commands that a Ctrl-C ends loses the three names in T7.5.

- [ ] **Step 3: Port the hostile-input and DAgger tests, then delete the old ones**

For every test of `tests/unit/test_sp2_serialization.py`, find the equivalent in T1.1's frame tests (`tests/unit/test_sp5_frame*.py`, `test_sp5_chunks*.py`); where there is none, port it into `tests/unit/test_sp5_ported_serialization.py` against the frame codec. Mapping (the task report lists, per row, "covered by <T1.1 test>" or "ported as <name>"):

| SP2 test | What it must still prove on the frame codec | Ported name (when T1.1 has none) |
|---|---|---|
| `test_pack_unpack_nested_structures` | nested dict (str / int keys) / list / tuple / py scalars / numpy leaves round-trip with dtypes, compressed and not | `test_frame_round_trips_nested_trees[lz4 / plain]` |
| `test_pack_rejects_tensors_object_arrays_and_container_keys` | `encode_frame` refuses a `torch.Tensor`, an object array and a tuple key (TypeError or ValueError, never a frame) | `test_encode_refuses_tensors_object_arrays_and_tuple_keys` |
| `test_chunk_payload_bytes_roundtrip_with_state`, `test_serialize_chunk_convenience_roundtrip` | `decode_chunk(encode_chunk(chunk))` equals the chunk (initial state, mask dtype bool, kind int8), meta carries agent / version / generation / node / worker | `test_chunk_frame_round_trip_with_state_and_meta` |
| `test_state_dict_roundtrip` | `decode_weights(encode_weights(sd, meta))` equals `sd` and `meta` | `test_weights_frame_round_trip` |
| `test_namedtuple_state_survives_the_wire` | a module-level namedtuple state (`HC`) comes back as `HC` | `test_namedtuple_state_survives_a_frame` |
| `test_unresolvable_namedtuple_is_rejected_with_a_clear_error` | a namedtuple class that is not module-level is refused at encode | `test_a_local_namedtuple_is_refused_at_encode` |
| `test_unpack_does_not_import_unknown_namedtuple_modules`, `test_namedtuple_lookup_never_triggers_module_getattr` | a header naming a module that is not in `sys.modules` (or whose class is reachable only through a PEP 562 `__getattr__`) is a `FrameError`, nothing is imported or called | `test_unknown_namedtuple_modules_are_never_imported` |
| `test_oversized_decompressed_payload_is_rejected`, `test_truncated_lz4_frame_is_rejected` | an lz4 data region over the cap, or truncated, is a `FrameError` before the full allocation | `test_lz4_bomb_and_truncated_lz4_are_frame_errors` |
| `test_array_header_declaring_a_huge_shape_is_rejected_before_allocation`, `test_negative_dimension_cannot_cancel_a_huge_array_out_of_the_cap`, `test_dimension_too_large_for_numpy_raises_value_error` | declared sizes over the cap, negative dims, dims above `sys.maxsize`, `nbytes != prod(shape) * itemsize` are `FrameError`s | `test_hostile_leaf_shapes_are_frame_errors[...]` |
| `test_malformed_skeletons_raise_value_error` | malformed tree nodes (`{"t": "nd", "i": -1}`, unknown `t`, `dict` with mismatched `k` / `v`, `nt` without fields, a leaf index out of range) are `FrameError`s | `test_malformed_trees_are_frame_errors[...]` |
| `test_weights_payload_rejects_non_numeric_dtypes` | the dtype whitelist (contract: bool, int8–int64, uint8–uint64, float16/32/64; `object`, str, bytes, void, datetime and complex refused) | `test_non_whitelisted_dtypes_are_frame_errors[...]` (complex is now refused: the SP1 codec accepted it; no model of the repo has complex weights) |
| `test_validate_chunk_payload_accepts_trees_and_rejects_bad_fields`, `test_validate_chunk_payload_rejects_a_broken_slot_structure` | `ChunkValidator.validate` accepts tree observations of the role's shapes and raises `ChunkError("invalid")` for a wrong slot dim, a wrong obs shape, a missing field and an ACT in the last slot | `test_chunk_validator_accepts_trees_and_rejects_bad_chunks` |

Raw hostile frames are built from the contract's byte layout (no codec internals):
```python
import json
import struct

FRAME_HEADER = struct.Struct("<4sHHQ")       # magic, version, flags, header length (contract §net.frame)


def raw_frame(header: dict, data: bytes = b"", *, flags: int = 0) -> bytes:
    text = json.dumps(header).encode()
    head = FRAME_HEADER.pack(b"CSTF", 1, flags, len(text)) + text
    pad = (-len(head)) % 64
    return head + b"\0" * pad + data
```

`tests/unit/test_sp3_dagger_chunks.py`: replace `from colosseum.transport.serialization import validate_chunk_payload` and its three uses:
- `test_payload_round_trip_carries_the_labels`: after the pickle round trip of the payload (kept: same-machine control messages are still pickled dataclasses, R3), build the chunk, `decode_chunk(encode_chunk(chunk, generation=1, node=-1, worker=0), cap=payload_byte_cap(64))`, and `ChunkValidator(RoleSpec(OBS, ACT), agent_id="a", teacher=True).validate(back)`; the label assertions stay.
- `test_payloads_without_the_fields_still_load`: the frame round trip of `TrajectoryChunk.from_payload(chunk_v2_payload(pattern="AAB"))` keeps `teacher_action is None and has_teacher is None`.
- `test_a_label_on_a_non_act_slot_breaks_the_slot_rules`: `validate_slot_structure` keeps raising `ValueError("teacher label on a non-ACT slot")`; the validator's equivalent is `ChunkError` with reason `"invalid"` for both the label on a BOOT slot and `has_teacher` without `teacher_action` ("come together").
- `test_weight_payload_carries_the_teacher_flag` gains: `decode_weights(encode_weights(state, WeightsMeta("a", "latest", 1, 1, teacher_active=True)), cap=...)[1].teacher_active is True`.

Then delete `tests/unit/test_sp2_serialization.py`.

- [ ] **Step 4: Delete the old modules and their tests**

```bash
git rm -r src/colosseum/distributed.py src/colosseum/transport src/colosseum/weight_store proto
git rm tests/integration/test_sp2_distributed_e2e.py tests/integration/test_sp2_grpc.py \
       tests/unit/test_sp2_distributed_roles.py tests/unit/test_sp3_distributed_guards.py \
       tests/unit/test_sp4_distributed_guards.py tests/unit/test_sp2_serialization.py
```
Remove the distributed test cases listed under **Files** (each named test function, its now-unused imports and helpers). In `pyproject.toml`, drop `"src/colosseum/transport/colosseum_pb2.py"` and `"src/colosseum/transport/colosseum_pb2_grpc.py"` from `[tool.ruff] extend-exclude` (T1.2 added the new generated files). `grep -rn "check_distributed_scope\|distributed_setup\|run_distributed\|serve_weight_store\|GRPCTransport\|SharedMemoryWeightStore\|LocalTransport\|transport.mode\|weight-store" src scripts tests examples configs` lists what is left; every hit is either removed or is the `transport.*` translation of T1.4 (and its tests).

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_old_roles.py tests/unit/test_sp5_ported_serialization.py tests/unit/test_sp3_dagger_chunks.py -q -rw`
Expected: all pass.

- [ ] **Step 5: Write the failing `validate` tests**

Create `tests/unit/test_sp5_validate_plan.py`:
```python
"""colosseum validate prints the placement plan and the distributed line (SP5 spec 9.2): learner slots by device,
each learner's preference, workers and evaluators, then ports, the TLS mode, the relay and the timeouts. Config only:
no hardware is probed."""
from __future__ import annotations

from click.testing import CliRunner

from cli_runner import TTT_CONFIG, TTT_MULTI_CONFIG
from colosseum.cli import main
from colosseum.core.config import load_config
from colosseum.core.registry import validate_config
from colosseum.hub.placement import describe_distributed, describe_plan
from game_helpers import make_test_config, scripted_agent


def test_the_distributed_line_names_ports_tls_relay_and_timeouts():
    line = describe_distributed(load_config(TTT_CONFIG))
    assert line == ("distributed: hub port 47100, node data port 47102, TLS on (token required), relay through the hub "
                    "on; heartbeat 1 s, node lost after 10 s, learner relocation after 120 s, hub lost timeout 1800 s "
                    "(relay-only collection 300 s), shutdown grace 20 s, block lease 600 s, max message 64 MB")


def test_insecure_and_no_relay_are_named():
    cfg = load_config(TTT_CONFIG, {"distributed.insecure": True, "distributed.relay": False,
                                   "distributed.port": 47200})
    line = describe_distributed(cfg)
    assert "hub port 47200" in line
    assert "TLS off (insecure: trusted networks only; the token is still required)" in line
    assert "relay off (unreachable nodes get no incoming data)" in line


def test_the_plan_counts_learner_slots_by_device_and_names_preferences():
    cfg = make_test_config("turns", agents={
        "main": {}, "cpuer": {"learner": {"device": "cpu", "placement": {"node": "laptop"}}},
        "gpu1": {"learner": {"device": "cuda:1", "placement": {"tags": ["gpu"], "strict": True}}},
        "greedy": scripted_agent()}, learner={"device": "auto"}, evaluation={"enabled": True})
    plan = describe_plan(cfg)
    assert plan[0] == ("placement: 3 learner slot(s) needed (1 gpu, 1 cpu, 1 auto); 'colosseum train' places every "
                       "learner on this machine")
    assert "learner of 'main' (device auto) needs a GPU slot if some node has a GPU, else a CPU slot" in plan[1]
    assert plan[2].endswith("needs a CPU learner slot; prefers node 'laptop' (may move elsewhere when its node is lost)")
    assert plan[3].endswith("needs the GPU slot of CUDA device 1; prefers tags gpu (strict: never elsewhere)")
    assert plan[-2].startswith("placement: workers: rollout.num_workers")
    assert plan[-1].startswith("placement: evaluators: pulse needs at least one")


def test_a_config_without_trainable_agents_needs_no_learner_slot():
    cfg = make_test_config("turns", agents={"a": scripted_agent(), "b": scripted_agent()})
    assert describe_plan(cfg)[0] == ("placement: no trainable agent: no learner slots (evaluation only, e.g. "
                                     "hub --idle)")


def test_validate_prints_the_plan_and_the_distributed_line():
    for config in (TTT_CONFIG, TTT_MULTI_CONFIG):
        result = CliRunner().invoke(main, ["validate", "-c", str(config)])
        assert result.exit_code == 0, result.output
        assert "  distributed: hub port 47100" in result.output
        assert "  placement: " in result.output and "learner slot(s) needed" in result.output


def test_the_play_scope_prints_neither():
    report = validate_config(load_config(TTT_CONFIG), scope="play")
    assert not any(line.startswith(("distributed:", "placement:")) for line in report.lines)
```
(`make_test_config` / `scripted_agent` are `tests/game_helpers.py` helpers; adapt the keyword names to their signatures, and the `turns` toy game to one that plays the agents' roles.)

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_validate_plan.py -q -rw`
Expected: FAIL (`cannot import name 'describe_distributed'`).

- [ ] **Step 6: `describe_plan` and `describe_distributed` (`src/colosseum/hub/placement.py`)**

If T4.1 left `describe_plan` as a stub, replace it; add `describe_distributed`. Both read the config only:
```python
def _sec(value: float) -> str:
    return f"{value:g} s"


def describe_distributed(config: ColosseumConfig) -> str:
    """The ``distributed:`` line of ``colosseum validate`` (SP5 spec 9.2): ports, TLS mode, relay, timeouts."""
    d = config.distributed
    tls = ("TLS off (insecure: trusted networks only; the token is still required)" if d.insecure
           else "TLS on (token required)")
    relay = "relay through the hub on" if d.relay else "relay off (unreachable nodes get no incoming data)"
    return (f"distributed: hub port {d.port}, node data port {d.data_port}, {tls}, {relay}; heartbeat "
            f"{_sec(d.heartbeat_sec)}, node lost after {_sec(d.node_lost_after_sec)}, learner relocation after "
            f"{_sec(d.learner_relocate_after_sec)}, hub lost timeout {_sec(d.hub_lost_timeout_sec)} (relay-only "
            f"collection {_sec(d.hub_lost_collect_sec)}), shutdown grace {_sec(d.shutdown_grace_sec)}, block lease "
            f"{_sec(d.block_lease_sec)}, max message {d.max_message_mb} MB")


_DEVICE_RULE = {
    "auto": "a GPU slot if some node has a GPU, else a CPU slot",
    "cpu": "a CPU learner slot",
    "mps": "a GPU slot (MPS)",
    "cuda": "a GPU slot",
}


def _device_rule(device: str) -> str:
    if device.startswith("cuda:"):
        return f"the GPU slot of CUDA device {device.split(':', 1)[1]}"
    return _DEVICE_RULE.get(device, f"device {device!r}")


def _slot_kind(device: str) -> str:
    return "cpu" if device == "cpu" else "auto" if device == "auto" else "gpu"


def describe_plan(config: ColosseumConfig) -> list[str]:
    """``placement:`` lines of ``colosseum validate`` (SP5 spec 9.2): the learner slots the config needs by device,
    each learner's preference, the workers and evaluators. Config only: no hardware is probed."""
    fixed = set(config.fixed_agent_ids())
    trainable = [aid for aid in config.agent_ids() if aid not in fixed]
    lines = []
    counts = {"gpu": 0, "cpu": 0, "auto": 0}
    for aid in trainable:
        learner = config.get_agent_config(aid).learner
        counts[_slot_kind(learner.device)] += 1
        pref = []
        if learner.placement.node:
            pref.append(f"node {learner.placement.node!r}")
        if learner.placement.tags:
            pref.append("tags " + ",".join(learner.placement.tags))
        where = ""
        if pref:
            where = f"; prefers {' and '.join(pref)}" + (" (strict: never elsewhere)" if learner.placement.strict
                                                           else " (may move elsewhere when its node is lost)")
        lines.append(f"placement: learner of '{aid}' (device {learner.device}) needs {_device_rule(learner.device)}"
                     f"{where}")
    if trainable:
        need = ", ".join(f"{n} {k}" for k, n in counts.items() if n)
        lines.insert(0, f"placement: {len(trainable)} learner slot(s) needed ({need}); 'colosseum train' places "
                        f"every learner on this machine")
    else:
        lines.insert(0, "placement: no trainable agent: no learner slots (evaluation only, e.g. hub --idle)")
    lines.append(f"placement: workers: rollout.num_workers {config.rollout.num_workers} on train's local node; a "
                 f"joined node runs physical cores - 1 (join --workers N)")
    if config.evaluation.enabled:
        lines.append("placement: evaluators: pulse needs at least one: train's local node runs 1; a joined node "
                     "runs 1 with >= 4 physical cores (join --evaluators N)")
    else:
        lines.append("placement: evaluators: only for tournaments (evaluation.enabled is false)")
    return lines
```
(Verified on stand-in configs; the wording of each line is pinned by Step 5's tests.)

`src/colosseum/core/validation.py`, `validate_config`, train scope, right after `_describe_retention(config, report)`:
```python
        from colosseum.hub.placement import describe_distributed, describe_plan  # lazy: core never imports hub at load

        report.lines.append(describe_distributed(config))
        report.lines.extend(describe_plan(config))
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_validate_plan.py -q -rw` → all pass.

- [ ] **Step 7: Write the failing logging tests**

Create `tests/unit/test_sp5_pending_log.py`:
```python
"""Logging before the log file exists (SP5 spec 4.10): with keep_until_file the records of INFO and above are kept
and written at the top of the log file by the next setup that names a log dir, in order and with their own times;
flush_pending writes them to a file of the caller's choice; discard_pending forgets them; overflow is one line."""
from __future__ import annotations

import collections
import logging

import pytest

from colosseum.utils import logging as plog


@pytest.fixture(autouse=True)
def _restore(restore_root_logging):
    yield
    plog.discard_pending()


def test_kept_records_open_the_log_file(tmp_path):
    plog.setup_process_logging(None, "main", logging.INFO, keep_until_file=True)
    logging.getLogger("colosseum.core.config").warning("transport.mode is unused since SP5; ignored")
    logging.getLogger("x").info("info before")
    logging.getLogger("x").debug("debug never")
    plog.setup_process_logging(tmp_path / "logs", "main", logging.INFO)
    logging.getLogger("x").info("after")
    lines = (tmp_path / "logs" / "main.log").read_text().splitlines()
    assert "[WARNING] main colosseum.core.config: transport.mode is unused" in lines[0]
    assert "info before" in lines[1] and "main started" in lines[2] and "after" in lines[3]
    assert not any("debug never" in line for line in lines)


def test_flush_pending_writes_the_kept_records_and_notes_an_overflow(tmp_path):
    plog.setup_process_logging(None, "node", keep_until_file=True)
    plog._pending.records = collections.deque(maxlen=3)
    for i in range(5):
        logging.getLogger("n").warning(f"w{i}")
    path = plog.flush_pending(tmp_path / "join.log", "node")
    lines = path.read_text().splitlines()
    assert "2 earlier log records of this process were not kept" in lines[0]
    assert [line[-2:] for line in lines[1:]] == ["w2", "w3", "w4"]
    assert plog.flush_pending(tmp_path / "join.log", "node") is None


def test_discarded_records_never_reach_a_file(tmp_path):
    plog.setup_process_logging(None, "main", keep_until_file=True)
    logging.getLogger("y").warning("invalid config")
    plog.discard_pending()
    plog.setup_process_logging(tmp_path / "logs", "main")
    assert "invalid config" not in (tmp_path / "logs" / "main.log").read_text()


def test_without_keep_nothing_is_kept(tmp_path):
    plog.setup_process_logging(None, "main")
    logging.getLogger("z").warning("console only")
    plog.setup_process_logging(tmp_path / "logs", "main")
    assert "console only" not in (tmp_path / "logs" / "main.log").read_text()
```
(`restore_root_logging` is the existing fixture of `tests/conftest.py`, used by `test_sp2_entry_points.py`; if its name differs, use that fixture.)

Create `tests/integration/test_sp5_logging_before_checks.py`:
- `test_config_translation_warnings_reach_main_log`: a config file under `tmp_path` = `configs/examples/tic_tac_toe.yaml` loaded as YAML with `transport: {mode: grpc, grpc_port: 50051}` added and `checkpoint.keep_last` replaced by the SP2 knob `checkpoint.pool_size: 5`; `cli_runner.run_train(cfg, tmp_path)` exits 0; `logs/main.log` contains the `transport.mode` / `transport.grpc_port` WARNING of T1.4 and the `pool_size` translation WARNING, both before the line `main started (pid`.
- `test_an_invalid_config_leaves_no_run_dir_and_no_log_file`: a config with an unknown key under `rollout` → exit 1, no `tmp_path/runs/<name>` dir, no `*.log` file anywhere under `tmp_path` (the kept records were discarded; the console has the error).
- `test_a_refused_node_writes_its_records_to_join_log`: `NodeAgent` (in a thread, `node_kit.toy_node_options`) against a `FakeHub` whose `Register` answers `PERMISSION_DENIED` "bad token" → `run()` returns 1; `<cache_home>/colosseum/logs/<ca8>/join.log` (ca8 = the token's fingerprint[:8]) contains the node's hardware INFO line and the refusal ERROR.
- `test_a_node_log_starts_with_what_the_node_logged_before_welcome`: the same with a `FakeHub` that welcomes the node with run id `r1` and then sends `stop_run`: `node_log_dir(ca8, "r1") / "node.log"` begins with the hardware INFO line, followed by `node started (pid`.

- [ ] **Step 8: Logging before checks**

`src/colosseum/utils/logging.py` (whole new module text; the SP4 behaviour of `setup_process_logging(log_dir, name, console_level)` is unchanged):
```python
"""Per-process logging: every process writes its own file under ``<run>/logs/``.

A process whose log file is known only after some checks (the hub: the run dir exists after the config checks; a
node: the run id comes with the hub's Welcome) calls ``setup_process_logging(None, name, keep_until_file=True)``
first: records of INFO and above are then kept in memory (at most ``PENDING_LIMIT``, the oldest dropped) and
written at the top of the log file by the next call that names a log dir, so warnings of the checks reach the log,
not only the console (SP5 spec 4.10). ``flush_pending`` writes them to a file of the caller's choice (a node that
stops before any Welcome); ``discard_pending`` forgets them (a config error: no run dir is created).
"""

from __future__ import annotations

import collections
import logging
import os
import sys
from pathlib import Path

ENV_LOG_DIR = "COLOSSEUM_LOG_DIR"            # inherited by grandchildren (subprocess env workers)
ENV_PROCESS_NAME = "COLOSSEUM_PROCESS_NAME"
PENDING_LIMIT = 10_000                       # records kept before the log file exists


class PendingRecords(logging.Handler):
    """Keeps records (INFO and above) until a file handler takes them; the oldest go first beyond ``limit``."""

    def __init__(self, limit: int = PENDING_LIMIT) -> None:
        super().__init__(logging.INFO)
        self.records: collections.deque[logging.LogRecord] = collections.deque(maxlen=limit)
        self.dropped = 0

    def emit(self, record: logging.LogRecord) -> None:
        if len(self.records) == self.records.maxlen:
            self.dropped += 1
        self.records.append(record)

    def replay(self, handler: logging.Handler) -> None:
        if self.dropped:
            note = logging.LogRecord(__name__, logging.WARNING, __file__, 0,
                                     f"{self.dropped} earlier log records of this process were not kept "
                                     f"(more than {self.records.maxlen} before the log file existed)", None, None)
            if self.records:
                note.created = self.records[0].created
            handler.handle(note)
        for record in self.records:
            if record.levelno >= handler.level:
                handler.handle(record)
        self.records.clear()
        self.dropped = 0


_pending: PendingRecords | None = None


def _formatter(process_name: str) -> logging.Formatter:
    return logging.Formatter(f"%(asctime)s [%(levelname)s] {process_name} %(name)s: %(message)s")


def setup_process_logging(log_dir: str | Path | None, process_name: str,
                          console_level: int = logging.WARNING, *, keep_until_file: bool = False) -> None:
    """Configure the root logger of this process. Call it first in every process target.

    - ``<log_dir>/<process_name>.log`` receives INFO and above (when ``log_dir`` is set), starting with the records
      kept by an earlier ``keep_until_file=True`` call of this process;
    - without ``log_dir`` and with ``keep_until_file``, INFO and above are kept in memory for that file;
    - stderr receives ``console_level`` and above: WARNING for children, INFO for main.

    Calling it again replaces the previous handlers.
    """
    global _pending
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        if handler is not _pending:
            handler.close()
    root.setLevel(logging.INFO)
    fmt = _formatter(process_name)
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
        if _pending is not None:
            _pending.replay(file_handler)
            _pending = None
        root.addHandler(file_handler)
        logging.getLogger(__name__).info(f"{process_name} started (pid {os.getpid()})")
    elif keep_until_file:
        if _pending is None:
            _pending = PendingRecords()
        root.addHandler(_pending)
    else:
        _pending = None
    logging.captureWarnings(True)


def flush_pending(path: str | Path, process_name: str) -> Path | None:
    """Append the kept records to ``path`` (a node that stops before its log dir is known); None: nothing kept."""
    global _pending
    if _pending is None or not (_pending.records or _pending.dropped):
        return None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(target, encoding="utf-8")
    handler.setFormatter(_formatter(process_name))
    try:
        _pending.replay(handler)
    finally:
        handler.close()
    return target


def discard_pending() -> None:
    """Forget the kept records (they reached the console already)."""
    global _pending
    if _pending is not None:
        logging.getLogger().removeHandler(_pending)
        _pending = None


def inherited_log_dir() -> str | None:
    """Log dir exported by the parent process (see ``utils.process.run_child``)."""
    return os.environ.get(ENV_LOG_DIR)
```

`src/colosseum/hub/hub.py`, `run_training` and `run_hub` (start-up order only; T7.1 does not touch these functions): the first call becomes `setup_process_logging(None, "main", console_level=logging.INFO, keep_until_file=True)`, before `load_config` (so its translation warnings are kept), the config checks (`validate_run_config`, resume checks) and the run-dir creation; the existing `setup_process_logging(run_dir.logs, "main", console_level=logging.INFO)` after `RunDir.create` / the in-place resume replays them into `main.log`; on any exception before the run dir exists, `discard_pending()` before re-raising (the SP2 rule "an invalid config leaves nothing behind" stays: `test_sp2_entry_points.py::test_train_rejects_an_invalid_config_without_creating_a_run_dir` keeps passing).

`src/colosseum/node/agent.py`, `NodeAgent.run` (a `join` node; `NodeOptions.local` nodes know their log dir and log to it directly): first `setup_process_logging(None, "node", console_level=logging.INFO, keep_until_file=True)`, then an INFO line `hardware: <hardware_text(probe())>` (one line: cores, memory, GPUs, OS; `arena.report.data.hardware_text` of T7.1 or an equivalent local formatter if T7.1 is not merged yet), then the registration and the session; after the first `Welcome`, `setup_process_logging(node_log_dir(ca8, run_id), "node", console_level=logging.INFO)` (replays the kept records); when `run()` returns before any Welcome (refusal, Ctrl+C, a fatal error), `flush_pending(cache_home() / "colosseum" / "logs" / ca8 / "join.log", "node")` and print its path in the final error line. A later Welcome of another run switches the log file to that run's dir (the SP5 rule of T2.4 / T6.3; no replay needed).

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_pending_log.py tests/integration/test_sp5_logging_before_checks.py tests/unit/test_sp2_entry_points.py -q -rw`
Expected: all pass.

- [ ] **Step 9: Leftovers of the queue era**

`grep -rn "rollout_worker_process\|put_latest\|SharedCounter\|BatchedCounter\|QueueReader\|flush_queue" src scripts tests`: delete each helper that has no remaining caller in `src/` or `scripts/` (and the tests that test only it); keep `ResultBatcher`, `assert_no_tensors` and anything still used (the task report lists what was removed and what stayed, with the reason).

- [ ] **Step 10: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings; `ruff` no longer needs the old excludes.

- [ ] **Step 11: Commit and push**

```bash
git add -A src/colosseum tests pyproject.toml
git commit -m "feat: old distributed roles removed (exit 2 with a hint), validate placement plan and distributed line, logging before checks"
git push
```

---

### Task T7.3: Node image, compose files, `-m docker` tests; old deployment files removed

Spec block 10 (Docker; K8s manifests are T8.4). One `deployment/Dockerfile.node` with the variants `cpu` and `cuda` (build argument `VARIANT`): Python, uv and the thin join loader (T6.2) only. The project's code and environment come from the hub, so the image fits any project; `/root/.cache/colosseum` is a volume (otherwise every new container downloads torch again). The CUDA variant differs only in the NVIDIA runtime variables: torch's CUDA wheels bring the CUDA runtime, and the host's driver comes in through `--gpus` (NVIDIA Container Toolkit). A second target `hub` (the project installed with CPU torch) serves `deployment/compose.yaml`, which runs a hub and two nodes on one machine and trains tic-tac-toe to a small budget; `deployment/compose.node.yaml` runs one node on a VM. The old `Dockerfile.base` / `Dockerfile.weight-store` / `Dockerfile.worker` / `docker-compose.yaml` / `k8s/*` are deleted. Tests: static checks of the files (fast, always), and image + compose runs under the new `docker` marker (also `slow`), skipped when `docker info` fails — on the owner's machine Docker Desktop's WSL integration is off, so they skip there unless the owner turns it on (spec §3 criterion 14).

**Files:**
- Create: `deployment/Dockerfile.node`, `deployment/node-entrypoint.sh` (mode 0755), `deployment/compose.yaml`, `deployment/compose.node.yaml`
- Delete: `deployment/Dockerfile.base`, `deployment/Dockerfile.weight-store`, `deployment/Dockerfile.worker`, `deployment/docker-compose.yaml`, `deployment/k8s/{namespace,weight-store,learner,worker}.yaml` (T8.4 adds the new manifests)
- Modify: `pyproject.toml` (marker `docker`), `tests/conftest.py` (skip `docker` tests without a working Docker), `tests/unit/test_example_configs.py` (delete `test_deployment_learner_keeps_run_dir_on_the_mounted_volume` — its subject is removed; adjust the module docstring)
- Test: `tests/unit/test_sp5_deployment_files.py`, `tests/integration/test_sp5_docker_slow.py`

**Interfaces:**
- Consumes: `src/colosseum/bootstrap/loader.py` (T6.2: runnable as `python3 loader.py ADDR [join flags]` with stdlib + grpcio), `colosseum hub token` (T6.3: prints the token alone on stdout, creating the hub identity if missing), `join --once`, `--workers`, `--cpu-learners`, `--no-learners`, `--evaluators`, `--name`, `--token-file` (T6.3), `colosseum hub -c ...` (T2.5).
- Produces: the image `colosseum-node:<cpu|cuda>` (entrypoint `colosseum-node`: `join ...` runs the loader; `--help` prints the usage), the build target `hub`; the compose project files; the `docker` marker.

- [ ] **Step 1: Write the failing static tests**

Create `tests/unit/test_sp5_deployment_files.py`:
```python
"""Deployment files of SP5 (spec block 10), checked without Docker: the node image (variants, uv, the loader, the
entrypoint), the compose files (a hub and two nodes to the budget; one node for a VM) and the removal of the SP2
deployment files."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deployment"


def _dockerfile() -> str:
    return (DEPLOY / "Dockerfile.node").read_text()


def test_node_dockerfile_has_cpu_and_cuda_variants_and_a_hub_target():
    text = _dockerfile()
    assert re.search(r"^ARG VARIANT=cpu$", text, re.M)
    stages = re.findall(r"^FROM \S+ AS (\S+)$", text, re.M)
    assert {"node-base", "node-cpu", "node-cuda", "hub", "node"} <= set(stages)
    assert stages[-1] == "node"                       # the default build target is the node image
    assert re.search(r"^FROM node-\$\{VARIANT\} AS node$", text, re.M)
    cuda = text.split("AS node-cuda", 1)[1].split("FROM ", 1)[0]
    assert "NVIDIA_VISIBLE_DEVICES=all" in cuda and "NVIDIA_DRIVER_CAPABILITIES=compute,utility" in cuda


def test_node_dockerfile_carries_python_uv_and_the_loader_only():
    text = _dockerfile()
    node_base = text.split("AS node-base", 1)[1].split("\nFROM ", 1)[0]
    assert re.search(r"^ARG UV_VERSION=0\.13\.1$", text, re.M)
    assert "COPY --from=uv /uv /uvx /usr/local/bin/" in node_base
    assert "COPY src/colosseum/bootstrap/loader.py /opt/colosseum/loader.py" in node_base
    assert "grpcio==" in node_base and "torch" not in node_base     # torch comes from the hub's env, per node
    assert 'ENTRYPOINT ["colosseum-node"]' in node_base
    assert 'VOLUME ["/root/.cache/colosseum"]' in node_base


def test_the_entrypoint_is_executable_valid_sh_and_runs_the_loader():
    script = DEPLOY / "node-entrypoint.sh"
    assert os.access(script, os.X_OK)
    assert subprocess.run(["sh", "-n", str(script)], capture_output=True).returncode == 0
    text = script.read_text()
    assert 'join) shift; exec python3 /opt/colosseum/loader.py "$@"' in text
    usage = subprocess.run(["sh", str(script), "--help"], capture_output=True, text=True)
    assert usage.returncode == 0 and "join HUB:PORT" in usage.stdout
    bad = subprocess.run(["sh", str(script), "train"], capture_output=True, text=True)
    assert bad.returncode == 2 and "unknown command" in bad.stderr


def _services(name: str) -> dict:
    return yaml.safe_load((DEPLOY / name).read_text())["services"]


def test_compose_runs_a_hub_and_two_once_nodes_with_shm_and_their_own_caches():
    services = _services("compose.yaml")
    assert set(services) == {"hub", "node-1", "node-2"}
    hub = services["hub"]
    assert hub["build"]["target"] == "hub"
    script = " ".join(hub["command"])
    assert "colosseum hub token > /shared/token" in script and "exec colosseum hub -c configs/examples/" in script
    assert "training.total_timesteps=${BUDGET:-20000}" in script
    caches = set()
    for name in ("node-1", "node-2"):
        node = services[name]
        cmd = node["command"]
        assert cmd[:2] == ["join", "hub:47100"] and "--once" in cmd and "--token-file" in cmd
        assert node["depends_on"]["hub"]["condition"] == "service_healthy" and node["shm_size"]
        caches |= {v.split(":")[0] for v in node["volumes"] if v.endswith(":/root/.cache/colosseum")}
    assert len(caches) == 2                           # two nodes on one machine: two cache volumes
    assert "--cpu-learners" in services["node-1"]["command"] and "--no-learners" in services["node-2"]["command"]


def test_compose_node_runs_one_restartable_node_with_volumes_and_a_gpu_profile():
    services = _services("compose.node.yaml")
    node, gpu = services["node"], services["node-gpu"]
    for svc in (node, gpu):
        assert svc["command"][0] == "join" and "--token-file" in svc["command"]
        assert svc["restart"] == "unless-stopped" and svc["shm_size"]
        mounts = {next(p for p in v.split(":") if p.startswith("/")) for v in svc["volumes"]}
        assert {"/root/.cache/colosseum", "/root/.local/share/colosseum", "/run/colosseum/token"} <= mounts
    assert gpu["profiles"] == ["gpu"] and gpu["build"]["args"]["VARIANT"] == "cuda"
    assert gpu["deploy"]["resources"]["reservations"]["devices"][0]["capabilities"] == ["gpu"]


def test_old_deployment_files_are_gone():
    for name in ("Dockerfile.base", "Dockerfile.weight-store", "Dockerfile.worker", "docker-compose.yaml",
                 "k8s/namespace.yaml", "k8s/weight-store.yaml", "k8s/learner.yaml", "k8s/worker.yaml"):
        assert not (DEPLOY / name).exists(), name
    texts = [p.read_text() for p in DEPLOY.rglob("*") if p.is_file()]
    assert not any(word in t for t in texts for word in ("serve-weight-store", "run-learner", "run-workers"))
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_deployment_files.py -q -rw`
Expected: FAIL (`FileNotFoundError: .../deployment/Dockerfile.node`).

- [ ] **Step 2: Write the files**

`deployment/Dockerfile.node`:
```dockerfile
# syntax=docker/dockerfile:1
# Colosseum node image (SP5 spec block 10): Python, uv and the thin join loader only. The project's code and its
# environment come from the hub (a bundle by code hash, a uv environment by requirements hash) and are cached in
# /root/.cache/colosseum: mount a volume there, or every new container downloads torch again.
#
#   docker build -f deployment/Dockerfile.node --build-arg VARIANT=cpu  -t colosseum-node:cpu  .
#   docker build -f deployment/Dockerfile.node --build-arg VARIANT=cuda -t colosseum-node:cuda .
#   docker run --rm --shm-size=1g -v colosseum-cache:/root/.cache/colosseum [--gpus all] \
#       -v "$PWD/token:/run/colosseum/token:ro" colosseum-node:cpu join HUB:47100 --token-file /run/colosseum/token
#
# The target "hub" (deployment/compose.yaml only) installs the project with CPU torch to run `colosseum hub` in a
# container next to two nodes on one machine. The default target (the last stage) is the node image.
ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.13.1
ARG VARIANT=cpu

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim-bookworm AS node-base
# the loader speaks the hub's Bootstrap service with grpcio only (spec 2.3); pin = the hub's floor in pyproject.toml
ARG GRPCIO_VERSION=1.84.0
COPY --from=uv /uv /uvx /usr/local/bin/
RUN pip install --no-cache-dir "grpcio==${GRPCIO_VERSION}"
COPY src/colosseum/bootstrap/loader.py /opt/colosseum/loader.py
COPY deployment/node-entrypoint.sh /usr/local/bin/colosseum-node
RUN chmod 0755 /usr/local/bin/colosseum-node
ENV PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
VOLUME ["/root/.cache/colosseum"]
ENTRYPOINT ["colosseum-node"]
CMD ["--help"]

FROM node-base AS node-cpu

FROM node-base AS node-cuda
# torch's CUDA wheels bring the CUDA runtime; the host's driver comes in with --gpus (NVIDIA Container Toolkit)
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility

FROM node-cpu AS hub
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
COPY configs ./configs
RUN uv pip install --system --torch-backend cpu -e .
ENTRYPOINT []
CMD ["colosseum", "--help"]

FROM node-${VARIANT} AS node
```
If T6.2's loader also imports `protobuf`, add `"protobuf==${PROTOBUF_VERSION}"` with the same pin rule (the floor of `pyproject.toml`), and the static test's `grpcio==` assertion stays.

`deployment/node-entrypoint.sh`:
```sh
#!/bin/sh
# Entrypoint of the colosseum-node image (SP5 spec block 10): "join ADDR [join flags]" runs the thin join loader
# (stdlib + grpcio; spec 2.3), which fetches the hub's code and environment and execs the node agent.
set -eu
usage() {
  echo "usage: docker run [--shm-size=1g] [-v colosseum-cache:/root/.cache/colosseum] [--gpus all] IMAGE \\"
  echo "         join HUB:PORT (--token-file FILE | --token TOKEN) [--workers N] [--evaluators N] [--name NAME] ..."
  echo "The token may also come from COLOSSEUM_TOKEN (docs/DISTRIBUTED_GUIDE.md)."
}
case "${1:-}" in
  join) shift; exec python3 /opt/colosseum/loader.py "$@" ;;
  ""|-h|--help|help) usage; exit 0 ;;
  *) echo "colosseum-node: unknown command '$1'" >&2; usage >&2; exit 2 ;;
esac
```
(`git add --chmod=+x deployment/node-entrypoint.sh`.)

`deployment/compose.yaml`:
```yaml
# Colosseum on one machine in containers (SP5 spec block 10): a hub and two nodes, for a smoke run and the docker
# tests. The hub trains tic-tac-toe to a small budget and exits 0; the nodes (--once) leave after the run.
#   BUDGET=20000 docker compose -f deployment/compose.yaml up --build --abort-on-container-exit --exit-code-from hub
#   docker compose -f deployment/compose.yaml down -v
name: colosseum
services:
  hub:
    build: {context: .., dockerfile: deployment/Dockerfile.node, target: hub}
    image: colosseum-hub:compose
    command:
      - sh
      - -c
      - >-
        colosseum hub token > /shared/token &&
        exec colosseum hub -c configs/examples/tic_tac_toe.yaml --set run.dir=/runs --set run.name=compose
        --set training.total_timesteps=${BUDGET:-20000}
    volumes: ["shared:/shared", "runs:/runs"]
    healthcheck:
      test: ["CMD", "python3", "-c", "import os, socket; assert os.path.getsize('/shared/token') > 0; socket.create_connection(('127.0.0.1', 47100), 2)"]
      interval: 2s
      timeout: 5s
      retries: 90
  node-1:
    build: {context: .., dockerfile: deployment/Dockerfile.node, args: {VARIANT: cpu}}
    image: colosseum-node:cpu
    command: ["join", "hub:47100", "--token-file", "/shared/token", "--name", "node-1", "--workers", "1",
              "--cpu-learners", "1", "--evaluators", "0", "--once"]
    shm_size: "1gb"
    volumes: ["shared:/shared:ro", "cache-1:/root/.cache/colosseum"]
    depends_on: {hub: {condition: service_healthy}}
  node-2:
    build: {context: .., dockerfile: deployment/Dockerfile.node, args: {VARIANT: cpu}}
    image: colosseum-node:cpu
    command: ["join", "hub:47100", "--token-file", "/shared/token", "--name", "node-2", "--workers", "1",
              "--no-learners", "--evaluators", "0", "--once"]
    shm_size: "1gb"
    volumes: ["shared:/shared:ro", "cache-2:/root/.cache/colosseum"]
    depends_on: {hub: {condition: service_healthy}}
volumes: {shared: {}, runs: {}, cache-1: {}, cache-2: {}}
```
(Each node has its own cache volume: two nodes of one host would otherwise share `logs/<ca8>/<run_id>/node.log`. The hub's identity lives in the hub container; the nodes reach it as `hub:47100` with TLS and the fixed server name, so the address does not matter.)

`deployment/compose.node.yaml`:
```yaml
# One Colosseum node on a VM with Docker (SP5 spec block 10). The token file is mounted read-only; the cache and the
# node identity are volumes, so a restarted container stays the same node and does not download torch again.
#   HUB=hub.example.org:47100 TOKEN_FILE=./token docker compose -f deployment/compose.node.yaml up -d node
#   HUB=hub.example.org:47100 TOKEN_FILE=./token docker compose -f deployment/compose.node.yaml --profile gpu up -d node-gpu
# Chunks to this node's learners reach it through the hub (relay) unless its data port 47102 is reachable: with
# `network_mode: host` and the port open, other nodes connect directly (docs/DISTRIBUTED_GUIDE.md).
name: colosseum-node
services:
  node:
    build: {context: .., dockerfile: deployment/Dockerfile.node, args: {VARIANT: cpu}}
    image: colosseum-node:cpu
    command: ["join", "${HUB:?set HUB=host:port of the hub}", "--token-file", "/run/colosseum/token"]
    shm_size: "1gb"
    restart: unless-stopped
    volumes: ["cache:/root/.cache/colosseum", "identity:/root/.local/share/colosseum",
              "${TOKEN_FILE:-./token}:/run/colosseum/token:ro"]
  node-gpu:
    profiles: ["gpu"]
    build: {context: .., dockerfile: deployment/Dockerfile.node, args: {VARIANT: cuda}}
    image: colosseum-node:cuda
    command: ["join", "${HUB:?set HUB=host:port of the hub}", "--token-file", "/run/colosseum/token"]
    shm_size: "2gb"
    restart: unless-stopped
    volumes: ["cache:/root/.cache/colosseum", "identity:/root/.local/share/colosseum",
              "${TOKEN_FILE:-./token}:/run/colosseum/token:ro"]
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
volumes: {cache: {}, identity: {}}
```

Delete the old files:
```bash
git rm deployment/Dockerfile.base deployment/Dockerfile.weight-store deployment/Dockerfile.worker \
       deployment/docker-compose.yaml deployment/k8s/namespace.yaml deployment/k8s/weight-store.yaml \
       deployment/k8s/learner.yaml deployment/k8s/worker.yaml
```
and delete `test_deployment_learner_keeps_run_dir_on_the_mounted_volume` from `tests/unit/test_example_configs.py` (its `yaml` import too if unused).

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_deployment_files.py tests/unit/test_example_configs.py -q -rw`
Expected: all pass.

- [ ] **Step 3: The `docker` marker and its skip**

`pyproject.toml`, `[tool.pytest.ini_options] markers` gains:
```toml
    "docker: needs a working Docker (docker info); skipped without it (also marked slow)",
```
`tests/conftest.py`, `pytest_collection_modifyitems` (next to the GPU skip; keep the GPU logic unchanged):
```python
def _docker_works() -> bool:
    """``docker info`` answers (Docker Desktop's WSL stub without the integration fails it)."""
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False
```
and in `pytest_collection_modifyitems`, before the CUDA early return:
```python
    docker_items = [item for item in items if "docker" in item.keywords]
    if docker_items and not _docker_works():
        skip_docker = pytest.mark.skip(reason="requires a working Docker (docker info failed)")
        for item in docker_items:
            item.add_marker(skip_docker)
```
(`docker info` is run only when a collected test carries the marker.)

- [ ] **Step 4: The Docker tests**

Create `tests/integration/test_sp5_docker_slow.py`:
```python
"""The node image and the compose project with a real Docker (SP5 spec block 10, criterion 14). Skipped without a
working Docker (tests/conftest.py); also slow: the first build downloads Python, uv and, in compose, torch."""
from __future__ import annotations

import os
import subprocess
import uuid

import pytest

from cli_runner import REPO_ROOT

pytestmark = [pytest.mark.docker, pytest.mark.slow, pytest.mark.timeout(3600)]
IMAGE = "colosseum-node:sp5-test"


def _run(cmd: list[str], *, timeout: float, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, **(env or {})})


def test_node_image_builds_and_runs_the_loader():
    build = _run(["docker", "build", "-f", "deployment/Dockerfile.node", "--build-arg", "VARIANT=cpu", "-t", IMAGE,
                  "."], timeout=1800)
    assert build.returncode == 0, build.stderr[-3000:]
    usage = _run(["docker", "run", "--rm", IMAGE, "--help"], timeout=120)
    assert usage.returncode == 0 and "join HUB:PORT" in usage.stdout
    loader = _run(["docker", "run", "--rm", IMAGE, "join", "--help"], timeout=120)
    assert loader.returncode == 0 and "--token-file" in loader.stdout + loader.stderr
    tools = _run(["docker", "run", "--rm", "--entrypoint", "sh", IMAGE, "-c", "uv --version && python3 -c 'import grpc'"],
                 timeout=120)
    assert tools.returncode == 0 and "uv 0.13.1" in tools.stdout


def test_compose_trains_to_the_budget():
    project = f"sp5test{uuid.uuid4().hex[:8]}"
    base = ["docker", "compose", "-f", "deployment/compose.yaml", "-p", project]
    try:
        up = _run([*base, "up", "--build", "--abort-on-container-exit", "--exit-code-from", "hub"], timeout=3000,
                  env={"BUDGET": "20000"})
        assert up.returncode == 0, (up.stdout[-3000:], up.stderr[-3000:])
        assert "Training finished" in up.stdout or "budget" in up.stdout.lower()
    finally:
        _run([*base, "down", "-v", "--remove-orphans"], timeout=300)
```
(The hub's "Training finished" line is SP4's INFO at exit 0 (`run_training`); T2.5 keeps it for `hub`. If its wording changes, assert on the hub service's exit code alone.)

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/integration/test_sp5_docker_slow.py -m docker -q -rs`
Expected without Docker: 2 skipped, reason "requires a working Docker". With Docker (owner's machine after turning the WSL integration on): 2 passed (record the durations in the task report).

- [ ] **Step 5: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings (the docker tests are slow: not collected by the fast suite's `-m "not gpu and not slow"`).

- [ ] **Step 6: Commit and push**

```bash
git add deployment pyproject.toml tests/conftest.py tests/unit/test_example_configs.py \
        tests/unit/test_sp5_deployment_files.py tests/integration/test_sp5_docker_slow.py
git commit -m "feat: node image (cpu/cuda), compose hub + 2 nodes and a VM node, docker marker; SP2 deployment removed"
git push
```

---

### Task T7.4: Measurements (criteria 2–5, 9) and the slow tests (localhost distributed pipeline, hub load)

Spec §3 criteria 2, 3, 4, 5 and 9, and §6 «Slow». Three scripts and two slow tests, then the measurements on the final tree, recorded in `docs/benchmarks.md` section «После SP5» with raw JSON in `docs/benchmarks/sp5-*.json`:
- `scripts/measure_common.py` — shared helpers: nearest-rank percentile, `SystemTail` (a run's `system` / `train` records sampled by read time on this process's monotonic clock), env steps/s by counters, mean queue depth, waiting for a quiet machine (1-minute load ≤ 2.0, SP4 practice).
- `scripts/measure_hub_load.py` — criterion 3: a real `colosseum hub` on localhost against 20 synthetic nodes × 16 workers in 2 generator processes; the verdict with the criterion's thresholds.
- `scripts/measure_scaling.py` — criterion 9: the workload rule, the learner capacity fed ready chunks, one run per variant (a / b / c) and the verdict; run on the two machines by T7.6 (here: implemented, unit-tested and rehearsed on one machine with `--dry-run`).
- `tests/learning/test_sp5_distributed_pipeline_slow.py` — hub + 2 joined nodes on localhost: the unit_harvest league with anchors, pulse on an evaluator, then `tournament --hub` (`select`) on an idle hub, then `colosseum report` with «Nodes».
- `tests/integration/test_sp5_hub_load_slow.py` — `measure_hub_load.run` for 45 s with the criterion 3 thresholds.

**Split point (if one session is not enough):** dispatch (a) Steps 1–7 (scripts, unit tests, the two slow tests run once), (b) Steps 8–13 (the measurements on a quiet machine, `docs/benchmarks.md`, the slow suite, commit).

**Thresholds, fixed here before any measurement:**

| Criterion | Measured by | Pass when |
|---|---|---|
| 2 | `scripts/bench_throughput.py --workers 1 2 4 --duration 60 --warmup 15` on the branch and on `main` `d61eb02` (`--tree`), two runs each, alternating main → branch → main → branch, by counters, 1-minute load ≤ 2.0 before each run | for each worker count, mean(branch) / mean(main) ≥ 0.95 |
| 3 | `scripts/measure_hub_load.py` (defaults: 20 nodes × 16 workers × 8 envs, 25 000 results/s offered, 20 s warm-up, 60 s window) | acked ≥ 20 000 results/s over the window; lost = 0 (sent = acked = journal train rows, `dropped_match_results` 0); lineup reply p95 ≤ 200 ms; the slowest lineup reply sent during a refit with ≥ 1 800 entities ≤ 1 s; ≥ 2 such refits in the window |
| 4 (latency) | `scripts/measure_weight_cache.py` of T1.5 (50 MB model, 2 publishes/s, workers reading every leaf with `mmap` + `np.frombuffer`) | read p95 ≤ 100 ms |
| 4 (fetches) | T3.1's localhost test; on two machines `nodes.json` `weight_fetches` (T7.6) | `fetches == versions` per (machine, agent) |
| 5 | `scripts/measure_frame_decode.py` of T1.1 (256-slot `space_miners`-size chunk, ~260 KB) | decode + hostile-input checks + `ChunkValidator` ≥ 10 × faster than npz + JSON with `validate_chunk_payload` (T1.1's baseline measured on the pre-SP5 code it embeds) |
| 9 | `scripts/measure_scaling.py` on the two machines (T7.6) | the workload rule picks a workload; c / (a + b) ≥ 0.85 over two runs per variant |

**Files:**
- Create: `scripts/measure_common.py`, `scripts/measure_hub_load.py`, `scripts/measure_scaling.py`
- Create: `tests/unit/test_sp5_measure_scripts.py`, `tests/learning/test_sp5_distributed_pipeline_slow.py`, `tests/integration/test_sp5_hub_load_slow.py`
- Modify: `docs/benchmarks.md` (section «После SP5»), create `docs/benchmarks/sp5-throughput/`, `docs/benchmarks/sp5-hub-load.json`, `docs/benchmarks/sp5-weight-cache.json`, `docs/benchmarks/sp5-frame-decode.json`, `docs/benchmarks/sp5-scaling-dry-run/`
- Modify (only if missing): `src/colosseum/node/learner_main.py` (`algorithm_factory(cfg, agent)`), `src/colosseum/node/worker_main.py` (`collect_chunks(cfg, agent, n)`) — see Step 3

**Interfaces:**
- Consumes: `colosseum hub` / `train` / `join` / `hub token` / `hub --idle` / `tournament --hub` / `report` (T2.5, T5.2, T6.3, T7.1); `hub.registry.read_hub_json` (T2.5); `nodes.json`, `live_ratings` and the cluster keys (T7.1); the protobuf session and the security helpers (T1.2, T1.3, T2.3); `messages.result_to_proto`, `lineup_from_proto` (T1.2); `encode_checkpoint`, `encode_weights`, `WeightsMeta` (T1.1); `learner_process`, `ChunkIntake`, `LearnerOutputs` (T2.2); `node.hardware.probe` (T2.4); `bench_throughput._make_config`, `classify_bound` (SP4, T1.4, T2.5); `MatchLog.open_readonly` (SP4); `cluster_kit`, `cli_runner`, `pipeline_kit`, `arena_kit` (T3.1, T2.5 / T6.3, SP3, SP4).
- Produces: the three scripts and their CLIs (module docstrings below); `measure_hub_load.run(args) -> dict` (used by the slow test); the two slow tests; the «После SP5» section.

- [ ] **Step 1: Write the failing unit tests of the scripts' rules**

Create `tests/unit/test_sp5_measure_scripts.py`:
```python
"""Pure rules of the SP5 measurement scripts (plan T7.4): criterion 3's verdict (results/s, losses, lineup latency,
refit windows), criterion 9's workload rule and verdict, the system-record tail by counters, and the workload configs
(they validate). No processes."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))           # the scripts import measure_common / bench_throughput by name
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module                     # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


common = _load("measure_common")
load = _load("measure_hub_load")
scaling = _load("measure_scaling")


def test_percentile_is_nearest_rank():
    assert common.percentile([], 95) is None and common.percentile([3.0], 95) == 3.0
    assert common.percentile([float(i) for i in range(1, 101)], 95) == 95.0
    assert common.percentile([1, 2, 3, 4], 50) == 2.0


def _samples():
    S = load.RefitSample
    return [S(10.0, None, None, 0), S(11.0, 1, 1.5, 1900), S(12.0, 1, 1.5, 1900), S(31.0, 2, 2.0, 2100)]


def test_refit_windows_cover_each_published_refit_with_the_read_slack():
    assert load.refit_windows(_samples(), read_slack=1.25) == [(10.0 - 1.5 - 1.25, 11.0, 1900),
                                                                (12.0 - 2.0 - 1.25, 31.0, 2100)]


def test_latencies_during_count_requests_sent_in_big_refits_only():
    windows = load.refit_windows(_samples(), read_slack=1.25)
    requests = [(5.0, 0.01), (8.0, 0.4), (20.0, 0.9), (40.0, 2.0)]
    assert load.latencies_during(windows, requests) == [0.4, 0.9]
    assert load.latencies_during(windows, requests, min_entities=2000) == [0.9]


def _stats(**kw):
    windows = load.refit_windows(_samples(), read_slack=1.25)
    base = dict(window_sec=60.0, results_sent=1_300_000, results_acked=1_300_000, journal_rows=1_300_000,
                acked_in_window=1_250_000, dropped_match_results=0,
                lineup_requests=[(8.0, 0.4), (20.0, 0.9)] + [(50.0 + i, 0.02) for i in range(98)], refits=windows)
    base.update(kw)
    return load.LoadStats(**base)


def test_the_hub_load_verdict_passes_at_the_thresholds():
    verdict = load.hub_load_verdict(_stats(acked_in_window=1_200_000))
    assert verdict["passed"], verdict
    assert verdict["checks"]["results_per_sec"]["value"] == 20_000.0


@pytest.mark.parametrize("change, check", [
    ({"acked_in_window": 1_199_999}, "results_per_sec"),
    ({"results_acked": 1_299_999}, "lost"),
    ({"journal_rows": 1_299_995}, "lost"),
    ({"dropped_match_results": 1}, "lost"),
    ({"lineup_requests": [(50.0 + i, 0.25) for i in range(100)]}, "lineup_p95_sec"),
    ({"lineup_requests": [(8.0, 1.2)] + [(50.0 + i, 0.02) for i in range(99)]}, "lineup_max_during_refit_sec"),
    ({"refits": [(0.0, 1.0, 1000), (5.0, 6.0, 1900)]}, "refits_in_window"),
])
def test_the_hub_load_verdict_fails_each_threshold(change, check):
    verdict = load.hub_load_verdict(_stats(**change))
    assert not verdict["passed"] and not verdict["checks"][check]["pass"]


def test_the_load_config_validates():
    from colosseum.core.config import ColosseumConfig

    cfg = ColosseumConfig(**load.load_config_dict(envs=8, run_dir=Path("runs")))
    assert cfg.fixed_agent_ids() == ["f0", "f1"] and cfg.checkpoint.interval == 1
    assert cfg.ratings.refit_interval_sec == 20.0 and cfg.checkpoint.keep_top_k == 5


def _runs(*values):
    return [scaling.VariantRun(v, x, b) for v, x, b in values]


def test_the_workload_rule_needs_worker_bound_a_and_capacity_for_both():
    a = _runs(("a", 30000, "worker"), ("a", 31000, "worker"))
    b = _runs(("b", 9000, "worker"))
    rule = scaling.workload_rule(a, b, 48000)
    assert rule["pass"] and rule["capacity_needed"] == pytest.approx(1.2 * 39500)
    assert not scaling.workload_rule(_runs(("a", 30000, "learner")), b, 1e9)["pass"]
    assert not scaling.workload_rule(a, b, 47000)["pass"]


def test_choose_workload_takes_the_first_passing_candidate_in_order():
    calib = {"ttt-gpu": {"pass": False}, "harvest-gpu": {"pass": True}, "harvest-gpu-b16": {"pass": True}}
    assert scaling.choose_workload(calib) == "harvest-gpu" and scaling.choose_workload({}) is None


def test_the_scaling_verdict_uses_two_runs_per_variant():
    runs = _runs(("a", 30000, "worker"), ("b", 9000, "worker"), ("c", 34000, "worker"), ("a", 31000, "worker"),
                 ("b", 9500, "worker"), ("c", 33000, "worker"))
    verdict = scaling.scaling_verdict(runs)
    assert verdict["ratio"] == pytest.approx(33500 / 39750) and verdict["pass"] is False
    runs[2] = scaling.VariantRun("c", 35000, "worker")
    runs[5] = scaling.VariantRun("c", 34500, "worker")
    assert scaling.scaling_verdict(runs)["pass"] is True
    with pytest.raises(ValueError):
        scaling.scaling_verdict(runs[:3])


def test_capacity_in_env_steps():
    assert scaling.capacity_env_steps(100.0, 8, 32.0) == 25600.0


@pytest.mark.parametrize("name", scaling.WORKLOADS)
def test_workload_configs_validate(name, tmp_path):
    from colosseum.core.config import ColosseumConfig

    cfg = ColosseumConfig(**scaling.workload_config(name, workers=3, run_parent=str(tmp_path), run_name="w"))
    assert cfg.rollout.num_workers == 3 and cfg.learner.device == "cuda"
    assert cfg.distributed.heartbeat_sec == 0.25 and cfg.match_log.enabled and not cfg.evaluation.enabled
    assert scaling.queue_size_of(scaling.workload_config(name, workers=3, run_parent="r", run_name="w")) >= 32


def test_the_system_tail_samples_by_read_time(tmp_path):
    path = tmp_path / "metrics.jsonl"
    clock = [100.0]
    tail = common.SystemTail(path, "main", clock=lambda: clock[0])
    assert tail.poll() == 0
    path.write_text(json.dumps({"kind": "train", "agent": "main", "train_step": 3, "chunks_received": 24}) + "\n"
                    + json.dumps({"kind": "system", "env_steps": 10, "queue_depths": {"main": 3}}) + "\n"
                    + '{"kind": "sys')
    assert tail.poll() == 1 and tail.samples[0][:3] == (100.0, 10, 3) and tail.train == [(100.0, 3, 24)]
    clock[0] = 102.0
    with path.open("a") as fh:
        fh.write('tem", "env_steps": 2010, "queue_depths": {}}\n')
    assert tail.poll() == 1 and tail.samples[1][:3] == (102.0, 2010, -1)
    assert common.env_rate(tail.samples) == 1000.0 and len(tail.window(101, 103)) == 1
    assert common.mean_depth(tail.samples) is None and common.mean_depth(tail.samples[:1]) == 3.0
    with pytest.raises(ValueError):
        common.env_rate(tail.samples[:1])


def test_wait_quiet_returns_without_sleeping_on_a_quiet_machine(monkeypatch):
    monkeypatch.setattr(common, "load_average", lambda: 0.5)
    assert common.wait_quiet(sleep=lambda s: pytest.fail("slept")) == 0.5
```

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_measure_scripts.py -q -rw`
Expected: FAIL at collection (`FileNotFoundError: .../scripts/measure_common.py`).

- [ ] **Step 2: Write `scripts/measure_common.py` and `scripts/measure_hub_load.py`**

`scripts/measure_common.py`:
```python
"""Shared helpers of the SP5 measurement scripts (``measure_hub_load.py``, ``measure_scaling.py``; plan T7.4).

Everything is measured by counters on this process's ``time.monotonic()`` (WSL2 wall clocks jump; a record's ``ts``
is never used): ``SystemTail`` samples a run's ``system`` records as they are appended, with the time they were read.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

QUIET_LOAD = 2.0                 # wait for a 1-minute load average at or below this before a run (SP4 practice)
QUIET_TIMEOUT_SEC = 900.0


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in [0, 100]); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return float(ordered[rank - 1])


class SystemTail:
    """Samples a run's ``system`` records as they are appended: ``(t, env_steps, queue_depth, record)`` with ``t``
    this process's monotonic time when the record was read. The hub writes a record every
    ``metrics.console_interval_sec``; ``poll`` every ``interval`` keeps the read lag below ``interval``. A partial last
    line waits for the next poll."""

    def __init__(self, metrics_path: str | Path, agent_id: str, clock: Callable[[], float] = time.monotonic) -> None:
        self.path = Path(metrics_path)
        self.agent_id = agent_id
        self.clock = clock
        self.samples: list[tuple[float, int, int, dict]] = []
        self.train: list[tuple[float, int, int]] = []        # (t, train_step, chunks_received) of agent_id
        self._offset = 0

    def poll(self) -> int:
        """Read the new complete lines; returns the number of new ``system`` samples."""
        try:
            with self.path.open("rb") as fh:
                fh.seek(self._offset)
                chunk = fh.read()
        except OSError:
            return 0
        end = chunk.rfind(b"\n")
        if end < 0:
            return 0
        self._offset += end + 1
        now, n = self.clock(), 0
        for line in chunk[:end].split(b"\n"):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") == "system":
                depth = (rec.get("queue_depths") or {}).get(self.agent_id, -1)
                self.samples.append((now, int(rec["env_steps"]), int(depth), rec))
                n += 1
            elif rec.get("kind") == "train" and rec.get("agent") == self.agent_id:
                self.train.append((now, int(rec["train_step"]), int(rec.get("chunks_received") or 0)))
        return n

    def window(self, start: float, end: float) -> list[tuple[float, int, int, dict]]:
        return [s for s in self.samples if start <= s[0] <= end]


def env_rate(samples: Sequence[tuple]) -> float:
    """Env steps/s between the first and the last sample (by counters)."""
    if len(samples) < 2 or samples[-1][0] <= samples[0][0]:
        raise ValueError("need two samples over a positive time")
    return (samples[-1][1] - samples[0][1]) / (samples[-1][0] - samples[0][0])


def mean_depth(samples: Sequence[tuple]) -> float | None:
    """Mean learner queue depth of the samples; None when a sample does not know it (-1)."""
    depths = [s[2] for s in samples]
    return sum(depths) / len(depths) if depths and all(d >= 0 for d in depths) else None


def load_average() -> float | None:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def wait_quiet(max_load: float = QUIET_LOAD, timeout: float = QUIET_TIMEOUT_SEC,
               sleep: Callable[[float], None] = time.sleep) -> float | None:
    """Wait until the 1-minute load average is at most ``max_load`` (at most ``timeout`` s); returns the last value
    (None where /proc/loadavg does not exist). The value is recorded with every run."""
    deadline = time.monotonic() + timeout
    while True:
        load = load_average()
        if load is None or load <= max_load or time.monotonic() >= deadline:
            return load
        sleep(5.0)


def machine_info() -> dict:
    import platform

    return {"platform": platform.platform(), "cpu_count": os.cpu_count(), "python": platform.python_version()}
```

`scripts/measure_hub_load.py` (the rules at the top are complete and unit-tested; the synthetic node and the driver are written on the contract's clients — adapt each call to the module as it is, keep the behaviour of the module docstring):
```python
"""Hub load (SP5 spec §3 criterion 3; plan T7.4): a real ``colosseum hub`` process on localhost against synthetic nodes.

Workload (fixed before measuring): the tic-tac-toe spec, 4 trainable agents (``a0``..``a3``, CPU learners placed on the
synthetic nodes 0..3) and 2 fixed agents (``f0`` RandomBot, ``f1`` TicTacToeBot); opponents latest 0.3 / snapshots 0.4
/ rivals 0.1 / anchors 0.2, so lineups name snapshots; 50 snapshots per trainable agent (the synthetic learners
announce versions 1..500 during the warm-up and upload every 10th as a snapshot; ``checkpoint.interval: 1`` makes every
version a rating bucket, so a refit has ~2 000 entities); the match log, live ratings (``ratings.refit_interval_sec``
20) and top-k (``checkpoint.keep_top_k`` 5) on; ``rollout.match_refresh_interval_sec`` 30.

``--nodes`` synthetic nodes (default 20) with ``--workers`` workers (16) of ``--envs`` envs (8) each run as threads of
``--generators`` processes (2): each registers with the hub's token over TLS, opens a session (Hello with the hub's
code hash, Python and torch versions), requests lineups per worker when the grants expire, plays no env: for every
granted cell it makes match results from the lineup (seats, sources, versions and generations as a node would report
them, outcomes from hidden strengths) at ``--rate`` results/s over all nodes (25 000), in batches of <= 256 every
0.25 s, sends heartbeats with cumulative env steps (9 per result) and counts acks.

Measured over ``--duration`` s (60) after ``--warmup`` s (20), on this machine's monotonic clock (all processes are on
one machine; nothing uses ``ts``): results acked per second (acks arrive after the journal commit, R8), lineup reply
latency (request sent -> grants received, per worker group), and the refits of the live ratings (from the hub's
``system`` records, ``live_ratings``). After the window the generators stop offering and wait up to 30 s for every ack;
then the hub is stopped (SIGINT) and the journal is read. Verdict (``hub_load_verdict``; thresholds of criterion 3):
>= 20 000 results/s, 0 lost (sent = acked = journal rows, ``dropped_match_results`` 0), lineup p95 <= 200 ms, the
slowest lineup reply during a refit with >= 1 800 entities <= 1 s, and at least 2 such refits in the window.

Usage (quiet machine, nothing else CPU-heavy)::

    env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/measure_hub_load.py \
        --json docs/benchmarks/sp5-hub-load.json
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_common import SystemTail, machine_info, percentile, wait_quiet  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------- criterion 3, fixed before measuring ----------------
MIN_RESULTS_PER_SEC = 20_000.0
MAX_LOST = 0
MAX_LINEUP_P95_SEC = 0.200
MAX_LINEUP_DURING_REFIT_SEC = 1.0
MIN_REFIT_ENTITIES = 1_800          # "a BT refit on ~2 000 entities": at least this many in a refit of the window
MIN_REFITS_IN_WINDOW = 2
TRAINABLE = ("a0", "a1", "a2", "a3")
FIXED = {"f0": "colosseum.players.RandomBot", "f1": "examples.tic_tac_toe.bots.TicTacToeBot"}
WARMUP_VERSIONS = 500               # versions each synthetic learner announces during the warm-up
SNAPSHOT_EVERY = 10                 # ... and every 10th is uploaded as a snapshot (50 per agent)
BATCH = 256
FLUSH_SEC = 0.25
ENV_STEPS_PER_RESULT = 9
DRAIN_SEC = 30.0


@dataclass(frozen=True)
class RefitSample:
    t: float                          # this process's monotonic time when the system record was read
    version: int | None               # live_ratings.version of the record
    refit_sec: float | None
    entities: int


def refit_windows(samples: Sequence[RefitSample], *, read_slack: float) -> list[tuple[float, float, int]]:
    """``(start, end, entities)`` of every refit published between two samples: it ran within
    ``[t_prev - refit_sec - read_slack, t_cur]`` on this process's clock (``read_slack``: the record interval plus
    the tail poll interval, the worst lag between a publication and its sample)."""
    out = []
    for prev, cur in zip(samples, samples[1:]):
        if cur.version is not None and cur.version != prev.version and cur.refit_sec is not None:
            out.append((prev.t - float(cur.refit_sec) - read_slack, cur.t, int(cur.entities)))
    return out


def latencies_during(windows: Sequence[tuple[float, float, int]], requests: Sequence[tuple[float, float]],
                     *, min_entities: int = MIN_REFIT_ENTITIES) -> list[float]:
    """Latencies of the lineup requests (``(sent_t, latency)``) sent during a refit window with enough entities."""
    big = [(a, b) for a, b, n in windows if n >= min_entities]
    return [lat for sent, lat in requests if any(a <= sent <= b for a, b in big)]


@dataclass
class LoadStats:
    window_sec: float
    results_sent: int                 # by the synthetic nodes over the whole run
    results_acked: int                # acked by the hub over the whole run (after the drain)
    journal_rows: int                 # source "train" rows in matches.db at the end
    acked_in_window: int              # results whose batch ack arrived inside the window
    dropped_match_results: int        # the hub's last system record
    lineup_requests: list[tuple[float, float]] = field(default_factory=list)   # (sent_t, latency) in the window
    refits: list[tuple[float, float, int]] = field(default_factory=list)


def hub_load_verdict(s: LoadStats) -> dict:
    """Criterion 3: every threshold with its value and pass flag, and ``passed``."""
    rate = s.acked_in_window / s.window_sec if s.window_sec > 0 else 0.0
    lost = s.results_sent - s.results_acked
    lat = [x for _, x in s.lineup_requests]
    p95 = percentile(lat, 95)
    big = [w for w in s.refits if w[2] >= MIN_REFIT_ENTITIES]
    during = latencies_during(s.refits, s.lineup_requests)
    worst = max(during) if during else None
    checks = {
        "results_per_sec": {"value": rate, "threshold": f">= {MIN_RESULTS_PER_SEC:g}",
                            "pass": rate >= MIN_RESULTS_PER_SEC},
        "lost": {"value": lost, "threshold": f"== {MAX_LOST}",
                 "pass": lost == MAX_LOST and s.journal_rows == s.results_acked and s.dropped_match_results == 0},
        "lineup_p95_sec": {"value": p95, "threshold": f"<= {MAX_LINEUP_P95_SEC}",
                           "pass": p95 is not None and p95 <= MAX_LINEUP_P95_SEC},
        "lineup_max_during_refit_sec": {"value": worst, "threshold": f"<= {MAX_LINEUP_DURING_REFIT_SEC}",
                                        "pass": worst is not None and worst <= MAX_LINEUP_DURING_REFIT_SEC},
        "refits_in_window": {"value": len(big), "threshold": f">= {MIN_REFITS_IN_WINDOW} with >= {MIN_REFIT_ENTITIES} "
                             f"entities", "pass": len(big) >= MIN_REFITS_IN_WINDOW},
    }
    return {"checks": checks, "passed": all(c["pass"] for c in checks.values()),
            "journal_rows": s.journal_rows, "results_sent": s.results_sent, "results_acked": s.results_acked,
            "lineup_requests": len(lat)}


def load_config_dict(*, envs: int, run_dir: Path) -> dict:
    """The hub's config (module docstring); every workload-relevant value is explicit."""
    return {
        "run": {"dir": str(run_dir), "name": "load"},
        "env": {"env_class": "examples.tic_tac_toe.game.TicTacToeGame", "kwargs": {}, "max_idle_steps": 1000},
        "networks": {"encoder_class": "examples.tic_tac_toe.models.TicTacToeEncoder",
                     "policy_class": "examples.tic_tac_toe.models.TicTacToePolicy",
                     "value_class": "examples.tic_tac_toe.models.TicTacToeValue", "kwargs": {}, "core": None},
        "agents": {**{a: {} for a in TRAINABLE},
                   **{a: {"kind": "scripted", "class": cls} for a, cls in FIXED.items()}},
        "learner": {"device": "cpu"},
        "rollout": {"num_workers": 1, "envs_per_worker": envs, "match_refresh_interval_sec": 30.0},
        "matchmaking": {"opponents": {"latest": 0.3, "snapshots": 0.4, "rivals": 0.1, "anchors": 0.2},
                        "anchors": {a: 1.0 for a in FIXED}},
        "checkpoint": {"interval": 1, "keep_last": 60, "keep_every": 0, "keep_top_k": 5, "max_snapshots": 100},
        "match_log": {"enabled": True, "detail_window_mb": 4096.0, "min_free_disk_mb": 1024.0},
        "ratings": {"refit_interval_sec": 20.0},
        "evaluation": {"enabled": False},
        "metrics": {"use_wandb": False, "log_interval": 100, "console_interval_sec": 1.0},
        "training": {"total_timesteps": 10**13, "seed": 0},
        "distributed": {"heartbeat_sec": 1.0},
    }


def strength(agent: str, network_id: str, version: int | None) -> float:
    """Hidden strengths of the synthetic outcomes: rising with the version, fixed agents in between."""
    if network_id == "fixed":
        return {"f0": 0.0, "f1": 1.0}.get(agent, 0.5)
    return 2.0 * min(1.0, (version or 0) / WARMUP_VERSIONS)


def synthetic_result(lineup, *, run_id: str, node: int, worker: int, env: int, epoch: int, episode: int,
                     versions: dict[str, int], generations: dict[str, int], rng):
    """One ``MatchResult`` of a granted lineup, as a node would report it (spec 4.3): uid with the node and the
    cell's epoch, every seat of the lineup with its source, latest seats with a version at or below the newest
    announced one and the current generation, snapshot seats with their version, a 2-team outcome."""
    from colosseum.core.types import MatchResult, SeatResult, TeamResult

    seats, team_strength = [], {}
    for i, seat in enumerate(lineup.seats):
        version = generation = None
        if seat.network_id == "latest":
            version = int(rng.integers(1, max(1, versions.get(seat.agent_id, 1)) + 1))
            generation = generations.get(seat.agent_id)
        elif seat.network_id.startswith("ckpt_v"):
            version = int(seat.network_id[len("ckpt_v"):])
        team_strength[i] = strength(seat.agent_id, seat.network_id, version)
        seats.append(SeatResult(i, "player", i, seat.agent_id, seat.network_id, 0.0, source=seat.source,
                                version_start=version, version_end=version, generation=generation))
    p0 = 1.0 / (1.0 + math.exp(team_strength.get(1, 0.0) - team_strength.get(0, 0.0)))
    draw = rng.random() < 0.1
    win0 = (not draw) and rng.random() < p0
    ranks = (1.0, 1.0) if draw else ((1.0, 2.0) if win0 else (2.0, 1.0))
    for s in seats:
        s.reward = 0.0 if draw else (1.0 if ranks[s.team] == 1.0 else -1.0)
    match_id = f"w{worker}_e{env}_g{epoch}_ep{episode}"
    return MatchResult(match_id=match_id, layout=lineup.layout, outcome_kind="wdl", seats=seats,
                       teams=[TeamResult(t, ranks[t], 1.0 if ranks[t] == 1.0 else 0.0) for t in (0, 1)],
                       episode_length=ENV_STEPS_PER_RESULT, uid=f"{run_id}/train/n{node}/{match_id}", source="train",
                       node=node)


# ---------------- the synthetic nodes (generator processes) ----------------
@dataclass
class GenStats:
    sent: int = 0
    acked: int = 0
    acked_in_window: int = 0
    requests: list[tuple[float, float]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class SyntheticNode:
    """One synthetic node (module docstring). Built on the contract's clients: ``security.fetch_hub_ca`` /
    ``hub_channel_credentials`` / ``NodeIdentity`` (T1.3), the ``Hub`` stub and messages of
    ``colosseum.net.proto.colosseum_v1_pb2`` (T1.2), ``messages.result_to_proto`` / ``lineup_from_proto`` (T1.2) and
    the upload client of T2.3 / T4.2. Adapt the calls to those modules as they are; keep the behaviour."""

    def __init__(self, index: int, *, address: str, token, info, args, window: tuple[float, float], workdir: Path,
                 stop_offering: threading.Event, done: threading.Event, stats: GenStats, lock: threading.Lock):
        self.i, self.address, self.token, self.info, self.args = index, address, token, info, args
        self.window, self.workdir, self.stop_offering, self.done = window, workdir, stop_offering, done
        self.stats, self.lock = stats, lock
        self.outbox: queue.SimpleQueue = queue.SimpleQueue()
        self.cells: dict[tuple[int, int], tuple] = {}            # (worker, env) -> (lineup, epoch, expires_at)
        self.pending: dict[int, float] = {}                     # worker -> request sent at
        self.versions: dict[str, int] = {}
        self.generations: dict[str, int] = {}
        self.unacked: dict[int, int] = {}                       # seq -> results
        self.episode: dict[tuple[int, int], int] = {}
        self.seq = 0
        self.node_number = -1
        self.run_id = ""

    # The session: Register, Hello, a reader thread, and the offering loop (lineups, results, heartbeats).
    def run(self) -> None:  # pragma: no cover - needs the hub; exercised by the slow test
        import grpc
        import numpy as np

        from colosseum.net import grpc_util, messages, security
        from colosseum.net.proto import colosseum_v1_pb2 as pb
        from colosseum.net.proto import colosseum_v1_pb2_grpc as pbg

        rng = np.random.default_rng(1000 + self.i)
        host, port = self.address.rsplit(":", 1)
        ca = security.fetch_hub_ca(host, int(port), fingerprint=self.token.ca_fingerprint, timeout=10.0)
        channel = grpc.secure_channel(self.address, security.hub_channel_credentials(ca),
                                      options=grpc_util.channel_options(64, server_name=security.HUB_SERVER_NAME))
        hub = pbg.HubStub(channel)
        identity = security.NodeIdentity.load_or_create(self.token.ca_fingerprint, root=self.workdir / f"node{self.i}")
        reply = hub.Register(pb.RegisterRequest(public_key_pem=identity.public_key_pem().decode(),
                                                machine_id=f"synthetic-{self.i}", name=f"syn-{self.i:02d}",
                                                tags=["synthetic"]),
                             metadata=[(security.MD_TOKEN, self.token.secret)])
        identity.save_registration(reply)
        auth = [(security.MD_NODE_ID, reply.node_id), (security.MD_NODE_SECRET, reply.node_secret)]
        session_id = uuid.uuid4().hex
        slots = pb.Slots(workers=self.args.workers, evaluators=0, cpu_learners=1 if self.i < len(TRAINABLE) else 0,
                         envs_per_worker=self.args.envs)
        hardware = pb.Hardware(logical_cores=32, physical_cores=16, memory_bytes=64 << 30, os="Linux",
                               platform="synthetic", python=self.info.python, torch_version=self.info.torch_version,
                               torch_build=self.info.torch_build, shm_free_bytes=8 << 30, cache_free_bytes=64 << 30)
        major, minor = messages.PROTOCOL_VERSION
        self.outbox.put(pb.NodeMessage(hello=pb.Hello(
            node_id=reply.node_id, session_id=session_id, protocol_major=major, protocol_minor=minor,
            code_hash=self.info.code_hash, hardware=hardware, slots=slots, name=f"syn-{self.i:02d}",
            tags=["synthetic"])))

        def requests():
            while True:
                item = self.outbox.get()
                if item is None:
                    return
                yield item

        responses = hub.Session(requests(), metadata=auth)
        reader = threading.Thread(target=self._read, args=(responses, pb, messages, hub, auth), daemon=True)
        reader.start()
        last_flush = last_beat = time.monotonic()
        while not self.done.is_set():
            now = time.monotonic()
            self._request_lineups(now, pb)
            if not self.stop_offering.is_set() and now - last_flush >= FLUSH_SEC:
                self._offer(now - last_flush, rng, pb, messages)
                last_flush = now
            if now - last_beat >= 1.0:
                with self.lock:
                    sent = self.stats.sent
                self.outbox.put(pb.NodeMessage(heartbeat=pb.Heartbeat(
                    env_steps=ENV_STEPS_PER_RESULT * sent, load=1.0,
                    procs=[pb.ProcState(name=f"worker-{w}", state="running") for w in range(self.args.workers)],
                    counters={"results_sent": sent})))
                last_beat = now
            time.sleep(0.01)
        self.outbox.put(None)
        channel.close()

    def _read(self, responses, pb, messages, hub, auth) -> None:  # pragma: no cover - needs the hub
        for msg in responses:
            kind = msg.WhichOneof("m")
            now = time.monotonic()
            if kind == "welcome":
                self.node_number, self.run_id = msg.welcome.node_number, msg.welcome.run_id
                self._versions(msg.welcome.versions)
                for assignment in msg.welcome.learners:
                    self._start_learner(assignment, pb, hub, auth)
            elif kind == "command" and msg.command.kind == "start_learner":
                self._start_learner(msg.command.learner, pb, hub, auth)
            elif kind == "announce":
                self._versions(msg.announce.versions)
            elif kind == "grants":
                for grant in msg.grants.grants:
                    lineup = messages.lineup_from_proto(grant.lineup)
                    expires = now + grant.ttl_sec if grant.ttl_sec > 0 else math.inf
                    self.cells[(grant.worker, grant.env)] = (lineup, grant.epoch, expires)
                    sent = self.pending.pop(grant.worker, None)
                    if sent is not None and self.window[0] <= sent <= self.window[1]:
                        with self.lock:
                            self.stats.requests.append((sent, now - sent))
            elif kind == "ack":
                n = sum(self.unacked.pop(seq, 0) for seq in msg.ack.seqs)
                with self.lock:
                    self.stats.acked += n
                    if self.window[0] <= now <= self.window[1]:
                        self.stats.acked_in_window += n

    def _versions(self, infos) -> None:
        for info in infos:
            if info.version > self.versions.get(info.agent_id, 0):
                self.versions[info.agent_id] = int(info.version)
                self.generations[info.agent_id] = int(info.generation)

    def _start_learner(self, assignment, pb, hub, auth) -> None:  # pragma: no cover - needs the hub
        """A synthetic learner: versions from the assignment's floor up to WARMUP_VERSIONS fast, then one per second;
        every SNAPSHOT_EVERY-th version is uploaded as a periodic checkpoint (the payload a real learner of a freshly
        built tic-tac-toe model would send, made once per agent with the learner's payload function)."""
        def loop() -> None:
            version = max(1, int(assignment.version_floor))
            while not self.done.is_set():
                self.outbox.put(pb.NodeMessage(version=pb.VersionAnnounce(info=pb.VersionInfo(
                    agent_id=assignment.agent_id, generation=assignment.generation, version=version, size=1,
                    digest="0" * 32, node=self.node_number, relay_only=True))))
                if version % SNAPSHOT_EVERY == 0:
                    upload_checkpoint(hub, auth, assignment.agent_id, assignment.generation, version,
                                      self.args.config)
                version += 1
                time.sleep(0.02 if version <= WARMUP_VERSIONS else 1.0)

        threading.Thread(target=loop, name=f"learner-{assignment.agent_id}", daemon=True).start()

    def _request_lineups(self, now: float, pb) -> None:
        if not self.versions:          # no collecting agent yet: the hub grants no training cells (spec 4.2)
            return
        for w in range(self.args.workers):
            if w in self.pending:
                continue
            due = [(e, self.cells.get((w, e))) for e in range(self.args.envs)]
            if all(c is not None and c[2] > now for _, c in due):
                continue
            cells = [pb.CellRequest(worker=w, env=e, epoch=c[1] if c else 0, incarnation=0) for e, c in due]
            self.pending[w] = now
            self.outbox.put(pb.NodeMessage(lineup_request=pb.LineupRequest(cells=cells)))
            return                     # one request per pass, like a node's single pass over expired cells

    def _offer(self, dt: float, rng, pb, messages) -> None:
        cells = list(self.cells.items())
        if not cells:
            return
        n = int(round(self.args.rate / self.args.nodes * dt))
        batch = []
        for k in range(n):
            (w, e), (lineup, epoch, _) = cells[k % len(cells)]
            episode = self.episode.get((w, e), 0)
            self.episode[(w, e)] = episode + 1
            batch.append(messages.result_to_proto(synthetic_result(
                lineup, run_id=self.run_id, node=self.node_number, worker=w, env=e, epoch=epoch, episode=episode,
                versions=self.versions, generations=self.generations, rng=rng)))
        for start in range(0, len(batch), BATCH):
            part = batch[start:start + BATCH]
            self.seq += 1
            self.unacked[self.seq] = len(part)
            self.outbox.put(pb.NodeMessage(results=pb.ResultBatch(seq=self.seq, results=part)))
            with self.lock:
                self.stats.sent += len(part)


def upload_checkpoint(hub, auth, agent_id: str, generation: int, version: int, config_path: str) -> None:
    """Upload one periodic checkpoint frame through ``Hub.Upload``: a header part, then the frame in parts
    (``grpc_util.send_parts``, T2.3), as a node uploads a learner's checkpoint (T4.2). Use the node's own upload
    function instead if T2.3 / T4.2 made one public."""  # pragma: no cover - needs the hub
    from colosseum.net import grpc_util
    from colosseum.net.proto import colosseum_v1_pb2 as pb

    frame = _checkpoint_frame(agent_id, generation, version, config_path)
    parts = list(grpc_util.send_parts(frame))
    header = pb.UploadHeader(kind="checkpoint", agent_id=agent_id, generation=generation, version=version,
                             total_size=len(frame), digest=parts[-1].digest, request_id=uuid.uuid4().hex)
    stream = [pb.UploadPart(header=header)] + [pb.UploadPart(data=p.data) for p in parts]
    hub.Upload(iter(stream), metadata=auth, timeout=60)


_MODELS: dict[str, dict] = {}


def _checkpoint_frame(agent_id: str, generation: int, version: int, config_path: str) -> bytes:  # pragma: no cover
    """``encode_checkpoint`` of a fresh tic-tac-toe model's payload at ``version`` (the model is built once per
    agent; the payload has the keys of ``learner.make_checkpoint_payload``)."""
    import io

    import torch

    from colosseum.core.config import load_config
    from colosseum.core.registry import build_model, env_spec
    from colosseum.core.roles import agent_role_spec
    from colosseum.net.chunks import encode_checkpoint

    if agent_id not in _MODELS:
        cfg = load_config(config_path)
        model = build_model(cfg.get_agent_config(agent_id), agent_role_spec(env_spec(cfg), ["player"]))
        _MODELS[agent_id] = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save({}, buf)
    payload = {"agent_id": agent_id, "policy_version": int(version), "model_state": _MODELS[agent_id],
               "trainer_state_bytes": buf.getvalue(), "final": False}
    return encode_checkpoint(payload, kind="periodic", generation=generation)


def generator_main(indices: list[int], address: str, token_text: str, info_json: str, args_json: str,
                   window: tuple[float, float], workdir: str, out: mp.Queue) -> None:  # pragma: no cover
    """One generator process: the synthetic nodes ``indices`` as threads until the window ends, then the drain."""
    from types import SimpleNamespace

    from colosseum.net import security

    args = SimpleNamespace(**json.loads(args_json))
    info = SimpleNamespace(**json.loads(info_json))
    token = security.parse_token(token_text)
    stats, lock = GenStats(), threading.Lock()
    stop_offering, done = threading.Event(), threading.Event()
    nodes = [SyntheticNode(i, address=address, token=token, info=info, args=args, window=window,
                           workdir=Path(workdir), stop_offering=stop_offering, done=done, stats=stats, lock=lock)
             for i in indices]
    threads = [threading.Thread(target=n.run, name=f"syn-{n.i}", daemon=True) for n in nodes]
    for t in threads:
        t.start()
    time.sleep(max(0.0, window[1] - time.monotonic()))
    stop_offering.set()
    deadline = time.monotonic() + DRAIN_SEC
    while time.monotonic() < deadline:
        with lock:
            if stats.acked >= stats.sent:
                break
        time.sleep(0.1)
    done.set()
    for t in threads:
        t.join(5)
    out.put(asdict(stats))


def run(args) -> dict:  # pragma: no cover - drives real processes
    """One measurement (module docstring); returns ``{"verdict", "stats", "workload", "machine", "load_before"}``."""
    from colosseum.net import security

    load_before = wait_quiet()
    keep = Path(args.keep) if args.keep else None
    with tempfile.TemporaryDirectory(prefix="hub-load-") as tmp:
        workdir = keep or Path(tmp)
        env = {**os.environ, "XDG_DATA_HOME": str(workdir / "xdg-data"), "XDG_CACHE_HOME": str(workdir / "xdg-cache"),
               "COLOSSEUM_SHM_DIR": str(workdir / "shm"), "OMP_NUM_THREADS": "1", "WANDB_MODE": "disabled"}
        os.environ.update({k: env[k] for k in ("XDG_DATA_HOME", "XDG_CACHE_HOME", "COLOSSEUM_SHM_DIR")})
        identity = security.HubIdentity.load_or_create()            # the identity the hub process will load
        cfg_path = workdir / "load.yaml"
        import yaml

        cfg_path.write_text(yaml.safe_dump(load_config_dict(envs=args.envs, run_dir=workdir / "runs")))
        hub = subprocess.Popen([sys.executable, "-m", "colosseum", "hub", "-c", str(cfg_path), "--listen", "127.0.0.1",
                                "--port", "0"], cwd=REPO_ROOT, env=env, start_new_session=True,
                               stdout=open(workdir / "hub.stdout", "w"), stderr=subprocess.STDOUT)
        try:
            root = workdir / "runs" / "load"
            address = _wait_address(root, timeout=120)
            info = _bootstrap_info(address, identity)
            tail = SystemTail(root / "metrics.jsonl", agent_id=TRAINABLE[0])
            t0 = time.monotonic()
            window = (t0 + args.warmup, t0 + args.warmup + args.duration)
            out: mp.Queue = mp.get_context("spawn").Queue()
            groups = [list(range(args.nodes))[g::args.generators] for g in range(args.generators)]
            procs = [mp.get_context("spawn").Process(
                target=generator_main, args=(g, address, str(identity.token), json.dumps(info),
                                             json.dumps({**vars(args), "config": str(cfg_path)}),
                                             window, str(workdir), out), daemon=True) for g in groups]
            for p in procs:
                p.start()
            refits: list[RefitSample] = []
            while time.monotonic() < window[1] + DRAIN_SEC + 5 and any(p.is_alive() for p in procs):
                if tail.poll():
                    for t, _, _, rec in tail.samples[len(refits):]:
                        live = rec.get("live_ratings") or {}
                        refits.append(RefitSample(t, live.get("version"), live.get("refit_sec"),
                                                  int(live.get("entities") or 0)))
                time.sleep(0.25)
            parts = [out.get(timeout=60) for _ in procs]
        finally:
            hub.send_signal(signal.SIGINT)
            try:
                hub.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(hub.pid, signal.SIGKILL)
        stats = LoadStats(window_sec=args.duration, results_sent=sum(p["sent"] for p in parts),
                          results_acked=sum(p["acked"] for p in parts), journal_rows=_journal_train_rows(root),
                          acked_in_window=sum(p["acked_in_window"] for p in parts),
                          dropped_match_results=_last_system(root).get("dropped_match_results", -1),
                          lineup_requests=[tuple(r) for p in parts for r in p["requests"]],
                          refits=[w for w in refit_windows(refits, read_slack=1.25)
                                  if window[0] <= w[1] and w[0] <= window[1]])
        verdict = hub_load_verdict(stats)
        return {"verdict": verdict, "stats": {k: v for k, v in asdict(stats).items() if k != "lineup_requests"},
                "workload": {k: v for k, v in vars(args).items() if k not in ("json", "keep")},
                "machine": machine_info(), "load_before": load_before,
                "errors": [e for p in parts for e in p["errors"]]}


def _wait_address(root: Path, timeout: float) -> str:  # pragma: no cover
    from colosseum.hub.registry import read_hub_json

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = read_hub_json(root)
        if info is not None and info.addresses:
            return info.addresses[0]
        time.sleep(0.2)
    raise RuntimeError(f"the hub wrote no hub.json in {timeout:g} s (see {root.parent.parent / 'hub.stdout'})")


def _bootstrap_info(address: str, identity) -> dict:  # pragma: no cover
    """Bootstrap/Info of the hub (code hash, Python, torch) so the synthetic nodes pass the registration checks."""
    import grpc

    from colosseum.net import grpc_util, security
    from colosseum.net.proto import colosseum_v1_pb2 as pb
    from colosseum.net.proto import colosseum_v1_pb2_grpc as pbg

    channel = grpc.secure_channel(address, security.hub_channel_credentials(identity.ca_pem),
                                  options=grpc_util.channel_options(64, server_name=security.HUB_SERVER_NAME))
    reply = pbg.BootstrapStub(channel).Info(pb.Empty(), timeout=30)
    channel.close()
    return {"code_hash": reply.code_hash, "python": reply.python, "torch_version": reply.torch_version,
            "torch_build": reply.torch_build}


def _journal_train_rows(root: Path) -> int:  # pragma: no cover
    from colosseum.arena.matchlog import MATCH_LOG_FILE, MatchLog

    with MatchLog.open_readonly(root / MATCH_LOG_FILE) as reader:
        return len(reader.matches(source="train", limit=None))


def _last_system(root: Path) -> dict:  # pragma: no cover
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines() if line.strip()]
    systems = sorted((r for r in records if r.get("kind") == "system"), key=lambda r: r["env_steps"])
    return systems[-1] if systems else {}


def main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--warmup", type=float, default=20.0)
    parser.add_argument("--generators", type=int, default=2)
    parser.add_argument("--rate", type=float, default=25_000.0, help="offered results/s over all nodes")
    parser.add_argument("--json", default=None)
    parser.add_argument("--keep", default=None, help="keep the work dir (hub run dir, logs) here")
    args = parser.parse_args()
    result = run(args)
    for name, check in result["verdict"]["checks"].items():
        print(f"{name}: {check['value']} (threshold {check['threshold']}) {'PASS' if check['pass'] else 'FAIL'}")
    print(f"criterion 3: {'PASS' if result['verdict']['passed'] else 'FAIL'}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```
Notes for the implementer:
- the synthetic nodes pass the registration checks of T6.3 because their `Hello` copies the hub's code hash, Python and torch versions from `Bootstrap/Info` and reports generous `/dev/shm` and cache space; their `Slots` give CPU learner slots on nodes 0–3 only, so placement (T4.1) puts the 4 learners there;
- they report no data addresses: nothing is ever sent to them over a data port; they play no env and need no weights (so they never fetch);
- if the hub answers no `LineupGrants` at all for cells while no agent can collect (spec 4.2), `_request_lineups` starts only after the first `Announce` of a version; a request that is never answered inside the window is recorded at the drain with its full wait (it counts against p95);
- the generators run in spawned processes (`OMP_NUM_THREADS=1`); with the hub that makes 3 loaded processes in the slow test (rule ≤ 3).

- [ ] **Step 3: Write `scripts/measure_scaling.py`**

```python
"""Two-machine scaling (SP5 spec §3 criterion 9; plan T7.4, run by the acceptance runbook T7.6).

The rule, fixed before measuring: the workload is the first of ``WORKLOADS`` for which (1) the PC alone is
worker-bound (the benchmark's classifier, ``bench_throughput.classify_bound``, on both calibration runs of variant
``a``) and (2) the learner's capacity, measured by feeding it ready chunks (``capacity``), is at least
``CAPACITY_FACTOR`` (1.2) x (a + b). Variants, by counters on this machine's monotonic clock (``measure_common``):

- ``a``: the PC alone, ``colosseum train`` (hub + local node), ``rollout.num_workers`` = the PC's physical cores - 1;
- ``b``: ``colosseum hub`` on the PC + the PC node with the learner only (``join --workers 0 --evaluators 0``) + the
  laptop's persistent node (its default workers): the laptop's workers with the PC's learner;
- ``c``: ``colosseum hub`` + the PC node with its workers (``join --evaluators 0``) + the laptop node: both machines.

Criterion 9 passes when c / (a + b) >= ``SCALING_MIN`` (0.85) over the means of two runs per variant, run in the order
a, b, c, a, b, c. A variant's env steps/s is measured over ``--duration`` s (60) after ``--warmup`` s (15) from the
hub's ``system`` records (``SystemTail``: read times on this process's clock, the records' env-step counter).

Subcommands::

    measure_scaling.py capacity --workload W --json cap.json        # learner fed ready chunks (PC, its GPU)
    measure_scaling.py run --variant a|b|c --workload W --laptop NAME --json run.json
    measure_scaling.py rule --capacity cap.json --runs a1.json a2.json b1.json      # calibration: the workload rule
    measure_scaling.py verdict --runs a1.json b1.json c1.json a2.json b2.json c2.json
    measure_scaling.py run ... --dry-run   # one machine: a second local join (its own identity) plays the laptop

Run on the PC (the hub machine) from the repo root, with the laptop's node joined once by the hub's one-liner
(``docs/DISTRIBUTED_GUIDE.md``); it stays connected and takes every run of this hub (spec 2.1).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_common import SystemTail, env_rate, machine_info, mean_depth, wait_quiet  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CAPACITY_FACTOR = 1.2               # learner capacity >= 1.2 x (a + b)
SCALING_MIN = 0.85                  # both machines >= 0.85 x (a + b)
WORKLOADS = ("ttt-gpu", "harvest-gpu", "harvest-gpu-b16")   # tried in this order
AGENT = {"ttt-gpu": "agent_0", "harvest-gpu": "main", "harvest-gpu-b16": "main"}


@dataclass(frozen=True)
class VariantRun:
    variant: str                      # "a" (PC alone), "b" (laptop workers, PC learner), "c" (both)
    env_steps_per_sec: float
    bound: str                        # "learner" | "worker" | "unknown" (bench classifier)
    env_steps_per_chunk: float | None = None


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def workload_rule(a_runs: Sequence[VariantRun], b_runs: Sequence[VariantRun], capacity_env_steps_per_sec: float
                  ) -> dict:
    """Spec criterion 9's workload rule on calibration runs: every run of variant a is worker-bound, and the learner
    capacity (fed ready chunks) is >= 1.2 (a + b)."""
    a = _mean([r.env_steps_per_sec for r in a_runs])
    b = _mean([r.env_steps_per_sec for r in b_runs])
    worker_bound = all(r.bound == "worker" for r in a_runs)
    need = CAPACITY_FACTOR * (a + b)
    return {"a": a, "b": b, "a_worker_bound": worker_bound, "capacity": capacity_env_steps_per_sec,
            "capacity_needed": need, "pass": worker_bound and capacity_env_steps_per_sec >= need}


def choose_workload(calibration: Mapping[str, dict]) -> str | None:
    """The first of ``WORKLOADS`` whose calibration passed ``workload_rule`` (None: none did)."""
    for name in WORKLOADS:
        result = calibration.get(name)
        if result is not None and result["pass"]:
            return name
    return None


def scaling_verdict(runs: Sequence[VariantRun]) -> dict:
    """Two runs per variant (a, b, c): c / (a + b) of the means >= 0.85."""
    by = {v: [r.env_steps_per_sec for r in runs if r.variant == v] for v in "abc"}
    if any(len(by[v]) < 2 for v in "abc"):
        raise ValueError(f"two runs per variant are needed, got { {v: len(x) for v, x in by.items()} }")
    a, b, c = (_mean(by[v]) for v in "abc")
    ratio = c / (a + b) if a + b > 0 else 0.0
    return {"a": a, "b": b, "c": c, "ratio": ratio, "threshold": SCALING_MIN, "pass": ratio >= SCALING_MIN,
            "runs": {v: by[v] for v in "abc"}}


def capacity_env_steps(updates_per_sec: float, batch_chunks: int, env_steps_per_chunk: float) -> float:
    """The learner's capacity in env steps/s: updates/s x chunks per update x env steps per chunk (the latter from
    the chunk generation of ``capacity``, or a variant run's env steps per received chunk)."""
    return updates_per_sec * batch_chunks * env_steps_per_chunk


def workload_config(name: str, *, workers: int, run_parent: str, run_name: str) -> dict:
    """The raw config of a workload (every workload-relevant value explicit)."""
    import bench_throughput as bench
    import yaml

    if name == "ttt-gpu":
        cfg = bench._make_config(workers, run_parent).model_dump(mode="json", by_alias=True)
        cfg["learner"]["device"] = "cuda"
    elif name in ("harvest-gpu", "harvest-gpu-b16"):
        cfg = yaml.safe_load((REPO_ROOT / "configs/examples/unit_harvest_league.yaml").read_text())
        cfg["agents"] = {"main": {}}                                  # latest-only self-play: every seat collects
        cfg["matchmaking"] = {"opponents": {"latest": 1.0, "snapshots": 0.0, "rivals": 0.0, "anchors": 0.0},
                              "anchors": {}}
        cfg["checkpoint"] = {**cfg.get("checkpoint", {}), "interval": 10**9, "keep_top_k": 0}
        cfg["rollout"] = {**cfg["rollout"], "num_workers": workers, "chunk_length": 32, "envs_per_worker": 16,
                          "weight_sync_interval_sec": 1.0, "match_refresh_interval_sec": 0.0, "torch_threads": 1}
        b16 = name.endswith("b16")
        cfg["learner"] = {**cfg["learner"], "device": "cuda", "batch_chunks": 16 if b16 else 8,
                          "queue_size": 64 if b16 else 32}
        cfg["training"] = {"total_timesteps": 10**12, "seed": 0}
        cfg["evaluation"] = {"enabled": False}
        cfg["match_log"] = {"enabled": True}
        cfg["metrics"] = {"use_wandb": False, "log_interval": 1, "console_interval_sec": 1.0}
    else:
        raise ValueError(f"unknown workload {name!r}; one of {WORKLOADS}")
    cfg["run"] = {"dir": run_parent, "name": run_name}
    cfg.setdefault("distributed", {})["heartbeat_sec"] = 0.25       # counter quantization < 1 % of the window
    return cfg


def queue_size_of(cfg: Mapping) -> int:
    return int(cfg["learner"]["queue_size"])


def run_variant(args) -> dict:  # pragma: no cover - drives real processes on two machines
    """One run of a variant (module docstring): start the processes, wait for the nodes, sample by counters, stop."""
    import bench_throughput as bench
    import yaml

    from colosseum.node.hardware import probe

    pc_workers = args.pc_workers or max(1, probe().physical_cores - 1)
    load_before = wait_quiet()
    with tempfile.TemporaryDirectory(prefix="scaling-") as tmp:
        tmp = Path(tmp)
        name = f"scale-{args.variant}-{int(time.monotonic())}"
        cfg = workload_config(args.workload, workers=pc_workers, run_parent=str(tmp / "runs"), run_name=name)
        path = tmp / "cfg.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        root = tmp / "runs" / name
        procs: list[subprocess.Popen] = []
        expect = {"pc"} if args.variant == "a" else {"pc", args.laptop}
        try:
            if args.variant == "a":
                procs.append(_spawn(["train", "-c", path], tmp / "train.log"))
            else:
                procs.append(_spawn(["hub", "-c", path], tmp / "hub.log"))
                _wait_file(root / "hub.json", 120)
                token = _hub_token(tmp)
                pc = ["join", "127.0.0.1:47100", "--token-file", token, "--name", "pc", "--evaluators", "0", "--once"]
                if args.variant == "b":
                    pc += ["--workers", "0"]
                procs.append(_spawn(pc, tmp / "pc-node.log"))
                if args.dry_run:
                    procs.append(_spawn(["join", "127.0.0.1:47100", "--token-file", token, "--name", args.laptop,
                                         "--workers", str(args.dry_run_workers), "--no-learners", "--evaluators", "0",
                                         "--once"], tmp / "laptop-sim.log", identity=tmp / "laptop-identity"))
            nodes = _wait_nodes(root, expect if args.variant != "a" else None, timeout=600)
            tail = SystemTail(root / "metrics.jsonl", AGENT[args.workload])
            start = time.monotonic() + args.warmup
            end = start + args.duration
            while time.monotonic() < end + 2.0:
                tail.poll()
                time.sleep(0.25)
            window = tail.window(start, end)
            rate = env_rate(window)
            depth = mean_depth(window)
            train = [s for s in tail.train if start <= s[0] <= end]
            per_chunk = None
            if len(train) >= 2 and train[-1][2] > train[0][2]:
                per_chunk = (window[-1][1] - window[0][1]) / (train[-1][2] - train[0][2])
            result = asdict(VariantRun(args.variant, rate, bench.classify_bound(depth, queue_size_of(cfg)), per_chunk))
            result.update({"workload": args.workload, "queue_depth_mean": depth, "pc_workers": pc_workers,
                           "nodes": nodes, "load_before": load_before, "machine": machine_info(),
                           "dry_run": bool(args.dry_run)})
            return result
        finally:
            for p in procs:
                if p.poll() is None:
                    p.send_signal(signal.SIGINT)
            for p in procs:
                try:
                    p.wait(60)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)


def _spawn(args: list, log: Path, identity: Path | None = None) -> subprocess.Popen:  # pragma: no cover
    env = {**os.environ, "OMP_NUM_THREADS": "1", "WANDB_MODE": "disabled"}
    if identity is not None:                     # a second node of one machine needs its own identity root
        env.update({"XDG_DATA_HOME": str(identity / "data"), "XDG_CACHE_HOME": str(identity / "cache")})
    return subprocess.Popen([sys.executable, "-m", "colosseum", *map(str, args)], cwd=REPO_ROOT, env=env,
                            stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)


def _hub_token(tmp: Path) -> str:  # pragma: no cover
    out = subprocess.run([sys.executable, "-m", "colosseum", "hub", "token"], cwd=REPO_ROOT, capture_output=True,
                         text=True, check=True).stdout.strip().splitlines()[-1]
    path = tmp / "token"
    path.write_text(out + "\n")
    path.chmod(0o600)
    return str(path)


def _wait_file(path: Path, timeout: float) -> None:  # pragma: no cover
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise RuntimeError(f"{path} did not appear in {timeout:g} s")
        time.sleep(0.2)


def _wait_nodes(root: Path, names: set[str] | None, timeout: float) -> list[dict]:  # pragma: no cover
    """``nodes.json`` rows once every named node is online (``None``: the local node of ``train``)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            rows = json.loads((root / "nodes.json").read_text())["nodes"]
        except (OSError, ValueError, KeyError):
            rows = []
        online = {r["name"] for r in rows if r["state"] == "online"}
        if rows and (names is None or names <= online):
            return [{k: r.get(k) for k in ("number", "name", "roles", "data_path", "hardware")} for r in rows]
        time.sleep(1.0)
    raise RuntimeError(f"nodes {sorted(names or [])} were not online in {timeout:g} s ({root / 'nodes.json'})")


class PrefedIntake:
    """``ChunkIntake`` (T2.2) over a pool of ready chunks: every ``get`` returns the next one at once."""

    def __init__(self, chunks: list) -> None:
        self._chunks, self._i = chunks, 0

    def get(self, timeout: float):
        chunk = self._chunks[self._i % len(self._chunks)]
        self._i += 1
        return chunk

    @property
    def depth(self) -> int:
        return len(self._chunks)

    @property
    def dropped(self) -> dict[str, int]:
        return {}


class TimingOutputs:
    """``LearnerOutputs`` (T2.2) that pays the per-publish encoding and records each train step's monotonic time."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.steps: list[tuple[float, int]] = []

    def publish_weights(self, payload) -> None:
        from colosseum.net.chunks import WeightsMeta, encode_weights

        encode_weights(payload.state_dict, WeightsMeta(self.agent_id, "latest", 1, int(payload.policy_version)))

    def send_checkpoint(self, payload: dict, *, kind: str, block: bool) -> bool:
        return True

    def send_metrics(self, metrics: dict) -> None:
        self.steps.append((time.monotonic(), int(metrics.get("train_step", 0))))


def run_capacity(args) -> dict:  # pragma: no cover - needs the workload's device
    """The learner fed ready chunks: the workload's learner, built as ``node.learner_main`` builds it (on its device),
    runs ``learner_process`` with a ``PrefedIntake`` over ``4 x batch_chunks`` real chunks of the workload and
    ``TimingOutputs``; updates/s over the window -> ``capacity_env_steps``."""
    import threading

    from colosseum.core.config import ColosseumConfig
    from colosseum.learner.learner import learner_process

    load_before = wait_quiet()
    with tempfile.TemporaryDirectory(prefix="capacity-") as tmp:
        cfg = ColosseumConfig(**workload_config(args.workload, workers=1, run_parent=tmp, run_name="capacity"))
        agent = AGENT[args.workload]
        learner_cfg = cfg.get_agent_config(agent).learner
        chunks, env_steps = make_chunks(cfg, agent, 4 * learner_cfg.batch_chunks)
        per_chunk = env_steps / len(chunks)
        outputs = TimingOutputs(agent)
        stop = threading.Event()
        timer = threading.Timer(args.warmup + args.duration, stop.set)
        t0 = time.monotonic()
        timer.start()
        try:
            learner_process(agent_id=agent, algorithm_factory=algorithm_factory(cfg, agent),
                            intake=PrefedIntake(chunks), outputs=outputs, config=learner_cfg, stop_event=stop)
        finally:
            timer.cancel()
    steps = [s for s in outputs.steps if t0 + args.warmup <= s[0] <= t0 + args.warmup + args.duration]
    if len(steps) < 2:
        raise RuntimeError(f"only {len(steps)} train steps in the window")
    updates = (steps[-1][1] - steps[0][1]) / (steps[-1][0] - steps[0][0])
    return {"workload": args.workload, "device": learner_cfg.device, "updates_per_sec": updates,
            "batch_chunks": learner_cfg.batch_chunks, "env_steps_per_chunk": per_chunk,
            "capacity_env_steps_per_sec": capacity_env_steps(updates, learner_cfg.batch_chunks, per_chunk),
            "load_before": load_before, "machine": machine_info()}


def algorithm_factory(cfg, agent: str):  # pragma: no cover - T2.4 / T4.1's learner construction
    """The zero-argument algorithm factory ``node.learner_main`` gives ``learner_process`` for ``agent`` (role
    spec, networks, algorithm, device); call the helper ``node.learner_main`` uses (T2.4, T4.1) rather than
    rebuilding it here."""
    from colosseum.node import learner_main

    return learner_main.algorithm_factory(cfg, agent)


def make_chunks(cfg, agent: str, n: int) -> tuple[list, int]:  # pragma: no cover - drives a RolloutLoop
    """``n`` real chunks of the workload and the env steps it took: a ``RolloutLoop`` (T2.2) in this process with
    latest-only self-play lineups (every seat collects), a ``LoopIO`` whose ``send_chunk`` keeps the chunks and
    whose ``add_env_steps`` counts, built with ``node.worker_main.create_env`` and the agent's model factory, as
    T2.2's loop tests build theirs; stopped as soon as ``n`` chunks were sent."""
    from colosseum.node import worker_main

    return worker_main.collect_chunks(cfg, agent, n)


def _load_runs(paths: Sequence[str]) -> list[VariantRun]:
    runs = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        runs.append(VariantRun(d["variant"], float(d["env_steps_per_sec"]), d["bound"], d.get("env_steps_per_chunk")))
    return runs


def main() -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capacity")
    cap.add_argument("--workload", choices=WORKLOADS, required=True)
    cap.add_argument("--duration", type=float, default=60.0)
    cap.add_argument("--warmup", type=float, default=15.0)
    cap.add_argument("--json", required=True)
    run = sub.add_parser("run")
    run.add_argument("--variant", choices="abc", required=True)
    run.add_argument("--workload", choices=WORKLOADS, required=True)
    run.add_argument("--laptop", default="laptop", help="the laptop node's name (nodes list)")
    run.add_argument("--pc-workers", type=int, default=None, help="default: the PC's physical cores - 1")
    run.add_argument("--duration", type=float, default=60.0)
    run.add_argument("--warmup", type=float, default=15.0)
    run.add_argument("--dry-run", action="store_true", help="one machine: a local join plays the laptop")
    run.add_argument("--dry-run-workers", type=int, default=2)
    run.add_argument("--json", required=True)
    rule = sub.add_parser("rule")
    rule.add_argument("--capacity", required=True)
    rule.add_argument("--runs", nargs="+", required=True)
    verdict = sub.add_parser("verdict")
    verdict.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    if args.cmd == "capacity":
        out = run_capacity(args)
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out))
        return 0
    if args.cmd == "run":
        out = run_variant(args)
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n")
        print(f"variant {out['variant']}: {out['env_steps_per_sec']:.0f} env steps/s, {out['bound']}-bound")
        return 0
    if args.cmd == "rule":
        runs = _load_runs(args.runs)
        capacity = json.loads(Path(args.capacity).read_text())["capacity_env_steps_per_sec"]
        result = workload_rule([r for r in runs if r.variant == "a"], [r for r in runs if r.variant == "b"], capacity)
        print(json.dumps(result, indent=1))
        return 0 if result["pass"] else 1
    runs = _load_runs(args.runs)
    result = scaling_verdict(runs)
    print(json.dumps(result, indent=1))
    print(f"criterion 9: {'PASS' if result['pass'] else 'FAIL'} (c / (a + b) = {result['ratio']:.3f}, "
          f">= {SCALING_MIN})")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
```
`algorithm_factory` and `make_chunks` call two helpers: `node.learner_main.algorithm_factory(cfg, agent)` (the zero-argument factory the learner child builds for `learner_process`; T2.4 / T4.1 have it under some name — reuse it, or expose it under this name) and `node.worker_main.collect_chunks(cfg, agent, n) -> (chunks, env_steps)` (add it if missing: a `RolloutLoop` (T2.2) in the calling process with `create_env`, the agent's model factory, latest-only lineups with every seat collecting, and a `LoopIO` whose `send_chunk` keeps the chunk and whose `add_env_steps` adds to a counter; it stops once `n` chunks were sent; no test besides the dry run, it is a measurement helper).

- [ ] **Step 4: Run the unit tests**

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_measure_scripts.py tests/unit/test_bench_throughput.py -q -rw`
Expected: all pass.

- [ ] **Step 5: The slow test of the localhost distributed pipeline**

Create `tests/learning/test_sp5_distributed_pipeline_slow.py` (adapt the kit helper names to those of T2.5 / T3.1 / T6.3; the assertions stay):
```python
"""Slow localhost distributed pipeline (SP5 spec §6 «Slow», plan T7.4). A hub and two joined nodes on this machine
train the unit_harvest league (anchors greedy and random) from scratch with pulse: node alpha runs the learner and
the pulse evaluator, node beta the rollout worker, so every chunk crosses nodes (3 loaded processes). Then an idle hub
runs `tournament --hub ... --mode select` over the run's snapshots on two evaluators (the same nodes, joined again),
and `colosseum report` draws the «Nodes» section. Thresholds = the SP4 pipeline test's (same game, budget, worker
count and pulse settings; part 07 ruling)."""
from __future__ import annotations

import json
import sqlite3
import sys

import pytest

import arena_kit as kit
import cluster_kit as ck
import pipeline_kit as pk
from cli_runner import run_in_session
from colosseum.arena.matchlog import MATCH_LOG_FILE
from colosseum.arena.report.data import build_report_data
from demo_learning import LEARNING_SEED

pytestmark = [pytest.mark.slow, pytest.mark.timeout(2700), pytest.mark.usefixtures("restore_global_rng")]

RHO_MIN, GAIN_MIN, VS_RANDOM_MIN_WIN = 0.85, 0.0, 0.95       # tests/learning/test_sp4_pipeline_slow.py
LAYOUT = "2p"


def test_two_node_league_pulse_tournament_and_report(tmp_path):
    config = pk.write_config(tmp_path / "harvest.yaml", pk.pipeline_config(pk.UNIT_HARVEST))
    sets = {"training.seed": LEARNING_SEED, "evaluation.enabled": True, "evaluation.every": 1,
            "evaluation.blocks": 25}
    # 1. training: hub + alpha (learner, evaluator) + beta (worker); --once: both leave when the run ends
    with ck.hub_process(config, tmp_path, "dist", overrides=sets, args=["--listen", "127.0.0.1", "--port", "0"]) as hub:
        alpha = ck.start_join(tmp_path, hub.address, hub.token_file, identity="alpha",
                              args=["--name", "alpha", "--workers", "0", "--cpu-learners", "1", "--evaluators", "1",
                                    "--once"])
        beta = ck.start_join(tmp_path, hub.address, hub.token_file, identity="beta",
                             args=["--name", "beta", "--workers", "1", "--no-learners", "--evaluators", "0", "--once"])
        assert hub.wait(timeout=1800) == 0, hub.tail()
        assert alpha.wait(timeout=60) == 0 and beta.wait(timeout=60) == 0
    root = hub.root
    nodes = {n["name"]: n for n in json.loads((root / "nodes.json").read_text())["nodes"]}
    assert set(nodes) == {"alpha", "beta"} and nodes["alpha"]["roles"]["learners"] == ["main"]
    with sqlite3.connect(f"file:{root / MATCH_LOG_FILE}?mode=ro", uri=True) as db:
        by_source = dict(db.execute("SELECT source || '/' || COALESCE(node, -1), COUNT(*) FROM matches "
                                    "GROUP BY source, node").fetchall())
    assert by_source.get(f"train/{nodes['beta']['number']}", 0) > 0              # training matches of beta only
    assert not any(k.startswith("train/") and k != f"train/{nodes['beta']['number']}" for k in by_source)
    assert by_source.get(f"pulse/{nodes['alpha']['number']}", 0) > 0              # pulse on alpha's evaluator
    for row in nodes.values():                                                    # sent = journaled + counted losses
        assert row["results_sent"] == by_source.get(f"train/{row['number']}", 0) + row["results_dropped"]
    system = sorted((json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()
                     if '"kind": "system"' in line), key=lambda r: r["env_steps"])[-1]
    assert system["dropped_match_results"] == 0 and (system["match_log"] or {}).get("duplicates", 0) == 0
    data = build_report_data(root)
    trend = kit.rating_trend(data)
    print(f"[sp5 pipeline] trend rho {trend.spearman:.3f} gain {trend.gain:.1f}; matches {by_source}; "
          f"beta data path {nodes['beta']['data_path']}")
    assert trend.spearman >= RHO_MIN and trend.gain > GAIN_MIN
    assert nodes["beta"]["data_path"] in ("direct", "relay")

    # 2. selection on the cluster: an idle hub, both nodes joined again as evaluators (same identities, numbers)
    with ck.hub_process(config, tmp_path, "idle", args=["--idle", "--listen", "127.0.0.1", "--port", "0"]) as idle:
        evaluators = [ck.start_join(tmp_path, idle.address, idle.token_file, identity=name,
                                    args=["--name", name, "--workers", "0", "--no-learners", "--evaluators", "1",
                                          "--once"]) for name in ("alpha", "beta")]
        proc = run_in_session([sys.executable, "-m", "colosseum", "tournament", "--hub", str(idle.root), "--run",
                               str(root), "--agent", pk.MAIN, "--snapshots", "all", "--field",
                               f"{pk.UNIT_HARVEST.bot},{pk.RANDOM}", "--mode", "select", "--seed", "7", "--name",
                               "sel"], timeout=1500, env=idle.env)
        assert proc.returncode == 0, proc.stderr[-3000:]
        idle.interrupt()                                                          # SIGINT: exit 130
        assert idle.wait(timeout=60) == 130 and all(e.wait(timeout=60) == 0 for e in evaluators)
    renumbered = {n["name"]: n["number"] for n in json.loads((idle.root / "nodes.json").read_text())["nodes"]}
    assert renumbered == {name: nodes[name]["number"] for name in ("alpha", "beta")}   # persistent identities
    from colosseum.arena.tournament.modes import report_view

    report = report_view(json.loads((root / "tournaments" / "sel" / "report.json").read_text()))
    winner = kit.group_of(report, LAYOUT)["select"]["winner"]
    version = int(winner.rsplit("@v", 1)[1])
    pairs = pk.evaluate(config, root / "checkpoints" / pk.MAIN / f"ckpt_v{version}", [pk.RANDOM],
                        tmp_path / "winner-eval.json", layout=LAYOUT, num_matches=60)
    print(f"[sp5 pipeline] select winner {winner}: win vs random {pairs[pk.RANDOM]['win_rate']:.3f}")
    assert pairs[pk.RANDOM]["win_rate"] >= VS_RANDOM_MIN_WIN

    # 3. the report of the distributed run has the «Nodes» section with both machines
    proc = run_in_session([sys.executable, "-m", "colosseum", "report", str(root)], timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    page = (root / "report.html").read_text()
    assert 'id="sec-nodes"' in page and '"name":"alpha"' in page and '"name":"beta"' in page
```
Helpers this test needs from `cluster_kit` (T3.1; add the missing ones here): `hub_process(config, tmp_path, name, *, overrides=None, args=()) -> ContextManager[HubProc]` (a `colosseum hub` process in its own session under `tmp_path/runs/<name>`, with its own `XDG_*` roots; `HubProc.address`, `.token_file` (the output of `colosseum hub token` with the same environment, mode 0600), `.root`, `.env`, `.wait(timeout)`, `.interrupt()`, `.tail()`; the session is killed on exit) and `start_join(tmp_path, address, token_file, *, identity, args) -> Popen` (a `colosseum join` process from the dev venv with the identity root `tmp_path/identities/<identity>` — the same root keeps the same node; its own session, reaped at test end). The hub's identity root is the same for both hubs of the test (one machine, one hub identity), so the nodes of phase 2 are the same nodes.

- [ ] **Step 6: The slow test of the hub load**

Create `tests/integration/test_sp5_hub_load_slow.py`:
```python
"""Hub load (SP5 spec §3 criterion 3, §6 «Slow»): a real hub against 20 synthetic nodes x 16 workers in two generator
processes (3 loaded processes) for 45 s after a 20 s warm-up, with the criterion's thresholds (scripts/
measure_hub_load.py). Run it on a quiet machine, never next to another CPU-heavy job."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.timeout(900)]
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _script():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("measure_hub_load", SCRIPTS / "measure_hub_load.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["measure_hub_load"] = module
    spec.loader.exec_module(module)
    return module


def test_hub_takes_the_load_of_twenty_synthetic_nodes(tmp_path):
    load = _script()
    args = SimpleNamespace(nodes=20, workers=16, envs=8, duration=45.0, warmup=20.0, generators=2, rate=25_000.0,
                           json=None, keep=str(tmp_path / "work"))
    result = load.run(args)
    print(json.dumps(result["verdict"], indent=1))
    assert not result["errors"], result["errors"]
    assert result["verdict"]["passed"], result["verdict"]["checks"]
```
(45 s: with refits every 20 s from the hub's start, a window that starts 20 s after the generators holds at least two refits once the versions reached 500.)

- [ ] **Step 7: Run the new slow tests once, alone**

Run (quiet machine, in the background, nothing else running): `env -u PYTHONPATH .venv/bin/python -m pytest tests/learning/test_sp5_distributed_pipeline_slow.py tests/integration/test_sp5_hub_load_slow.py -v -s`
Expected: both pass; record the durations and the printed numbers in the task report. If the pipeline's thresholds fail, follow the SP3 T6.4 rule: three calibration seeds, report, the controller decides (never edit a threshold without a ruling).

- [ ] **Step 8: Measure criterion 2 (one machine, ≥ 95 %)**

```bash
git worktree add /tmp/colosseum-main-d61eb02 d61eb02
for i in 1 2; do
  env -u PYTHONPATH PYTHONPATH=/tmp/colosseum-main-d61eb02/src .venv/bin/python scripts/bench_throughput.py \
      --tree /tmp/colosseum-main-d61eb02 --workers 1 2 4 --duration 60 --warmup 15 \
      --json docs/benchmarks/sp5-throughput/main-$i.json
  env -u PYTHONPATH .venv/bin/python scripts/bench_throughput.py --workers 1 2 4 --duration 60 --warmup 15 \
      --json docs/benchmarks/sp5-throughput/branch-$i.json
done
git worktree remove /tmp/colosseum-main-d61eb02
```
Before each run wait until `cat /proc/loadavg` shows a 1-minute load ≤ 2.0 and write it into the table. Pass: for each of 1 / 2 / 4 workers, mean(branch-1, branch-2) / mean(main-1, main-2) ≥ 0.95. (`--tree` runs `main`'s own `_make_config` and its `Launcher`; T2.5's `run_one` falls back to it.)

- [ ] **Step 9: Measure criteria 3, 4 and 5**

```bash
env -u PYTHONPATH OMP_NUM_THREADS=1 .venv/bin/python scripts/measure_hub_load.py --json docs/benchmarks/sp5-hub-load.json
env -u PYTHONPATH .venv/bin/python scripts/measure_weight_cache.py --json docs/benchmarks/sp5-weight-cache.json
env -u PYTHONPATH .venv/bin/python scripts/measure_frame_decode.py --json docs/benchmarks/sp5-frame-decode.json
```
(The last two are T1.5's and T1.1's scripts with their own defaults = the criterion's workload; their flags are theirs.) Pass/fail by the table at the top of this task. A FAIL is reported with the numbers; the controller decides (spec §8 risks: e.g. moving the journal writer or the relay to its own process).

- [ ] **Step 10: Rehearse criterion 9 on one machine**

```bash
mkdir -p docs/benchmarks/sp5-scaling-dry-run
for v in b c; do
  env -u PYTHONPATH .venv/bin/python scripts/measure_scaling.py run --variant $v --workload ttt-gpu --dry-run \
      --dry-run-workers 2 --pc-workers 2 --duration 20 --warmup 10 --json docs/benchmarks/sp5-scaling-dry-run/$v.json
done
env -u PYTHONPATH .venv/bin/python scripts/measure_scaling.py capacity --workload ttt-gpu --duration 20 --warmup 10 \
    --json docs/benchmarks/sp5-scaling-dry-run/capacity.json
```
Expected: three JSON files; `b.json` / `c.json` list the nodes `pc` and `laptop` online with a data path; no assertion on the numbers (one machine). This proves the driver before T7.6 runs it on two machines.

- [ ] **Step 11: `docs/benchmarks.md` «После SP5» (Russian)**

A new section after «После SP4», in the style of the earlier ones (date, machine, commit, load average before each run, raw data links):
- «Скорость на одной машине (критерий 2)» — the table `воркеры | main (1 / 2 / среднее) | ветка (1 / 2 / среднее) | ветка / main | апдейты/с | узкое место`, the verdict, the order of runs;
- «Нагрузка на хаб (критерий 3)» — workload, the five checks with values and thresholds, the verdict;
- «Кэш весов (критерий 4)» and «Формат кадра (критерий 5)» — the scripts' tables and verdicts;
- «Масштабирование (критерий 9)» — the rule and the candidate workloads (as fixed in this task), the dry run, "числа двух машин — в отчёте приёмки";
- «Slow-набор» — the durations of the SP5 slow tests.

- [ ] **Step 12: Full fast suite + ruff, then the slow suite once**

Run the full fast suite and ruff (green, zero warnings), then the whole slow suite once, alone, in the background: `env -u PYTHONPATH .venv/bin/python -m pytest -m slow -v` (16 tests: SP4's 14 + the two of this task; docker tests skip without Docker). Record its duration.

- [ ] **Step 13: Commit and push**

```bash
git add scripts/measure_common.py scripts/measure_hub_load.py scripts/measure_scaling.py tests/unit/test_sp5_measure_scripts.py \
        tests/learning/test_sp5_distributed_pipeline_slow.py tests/integration/test_sp5_hub_load_slow.py tests/cluster_kit.py \
        docs/benchmarks.md docs/benchmarks src/colosseum/node
git commit -m "test: SP5 measurements (hub load, scaling driver), localhost distributed pipeline and hub load slow tests"
git push
```

---

### Task T7.5: Documentation — `docs/DISTRIBUTED_GUIDE.md`, README, CLAUDE.md, ARENA_GUIDE, LEAGUE_GUIDE

Spec block 11. Written once, after the measurements (T7.4), against the code as it is: every command in the docs is run while writing them (on one machine: the two-machine commands with two local `join` processes and their own identity roots), and every flag is checked against `--help` by a fast test. Language: `DISTRIBUTED_GUIDE.md`, `ARENA_GUIDE.md`, `LEAGUE_GUIDE.md` in Russian; README and CLAUDE.md in English (CLAUDE.md conventions).

**Files:**
- Create: `docs/DISTRIBUTED_GUIDE.md`
- Modify: `README.md`, `CLAUDE.md`, `docs/ARENA_GUIDE.md`, `docs/LEAGUE_GUIDE.md`
- Test: `tests/unit/test_sp5_docs.py`

**Interfaces:**
- Consumes: the CLI (`colosseum.cli.main`, every command and option); the measurements of T7.4; `deployment/` (T7.3); part 08's state (done or moved to SP5b, R14).
- Produces: the documents; `tests/unit/test_sp5_docs.py` with the flag checker (`command_lines`, `unknown_flags` inside the test module).

- [ ] **Step 1: Write the failing doc test**

Create `tests/unit/test_sp5_docs.py`:
```python
"""SP5 documentation (spec block 11, criterion 15): every `colosseum ...` command line in the guides and the README
names a real command and only flags that its --help lists; DISTRIBUTED_GUIDE.md has every section the spec asks for;
nothing documents the removed SP2 roles as usable."""
from __future__ import annotations

import re
import shlex
from pathlib import Path

import click
import pytest

from colosseum.cli import REMOVED_COMMANDS, main

REPO = Path(__file__).resolve().parents[2]
DOCS = ("README.md", "docs/DISTRIBUTED_GUIDE.md", "docs/ARENA_GUIDE.md", "docs/LEAGUE_GUIDE.md")
_FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.S | re.M)


def command_lines(markdown: str) -> list[str]:
    """Every ``colosseum ...`` invocation inside fenced code blocks (continuation lines joined, comments cut)."""
    out = []
    for block in _FENCE.findall(markdown):
        joined = re.sub(r"\\\n\s*", " ", block)
        for line in joined.splitlines():
            line = line.split(" #", 1)[0].strip()
            for part in re.split(r"\s*(?:&&|\|\||;|\|)\s*", line):
                m = re.search(r"(?:^|[\s(=$])colosseum\s+(.*)$", part)
                if m and not part.lstrip().startswith(("#", "pip ", "uv ")):
                    out.append("colosseum " + m.group(1).strip())
    return out


def _tokens(line: str) -> list[str]:
    try:
        return shlex.split(line)
    except ValueError:
        return line.split()


def resolve(group: click.Group, line: str) -> tuple[click.Command | None, str, list[str]]:
    """``(command, its path, the option tokens)`` of one command line; command None when unknown."""
    tokens = _tokens(line)[1:]
    cmd: click.Command = group
    path = ["colosseum"]
    i = 0
    while isinstance(cmd, click.Group) and i < len(tokens) and not tokens[i].startswith("-"):
        sub = cmd.commands.get(tokens[i])
        if sub is None:
            if getattr(cmd, "invoke_without_command", False) and path != ["colosseum"]:
                break
            return None, " ".join(path + [tokens[i]]), []
        cmd, i = sub, i + 1
        path.append(sub.name)
    flags = [t.split("=", 1)[0] for t in tokens[i:] if t.startswith("--") or re.fullmatch(r"-[A-Za-z]", t)]
    return cmd, " ".join(path), flags


def unknown_flags(group: click.Group, markdown: str) -> list[str]:
    problems = []
    for line in command_lines(markdown):
        cmd, path, flags = resolve(group, line)
        if cmd is None:
            problems.append(f"{path}: unknown command")
            continue
        known = {opt for p in cmd.params for opt in (*p.opts, *p.secondary_opts)} | {"--help"}
        problems.extend(f"{path}: {f}" for f in flags if f not in known)
    return sorted(set(problems))


def test_the_checker_finds_unknown_commands_and_flags():
    md = "```bash\ncolosseum train -c c.yaml --bogus 1   # --nope\ncolosseum no-such-cmd\ncolosseum tournament \\\n" \
         "   --mode select --zzz\n```\n"
    assert unknown_flags(main, md) == ["colosseum no-such-cmd: unknown command", "colosseum tournament: --zzz",
                                       "colosseum train: --bogus"]


@pytest.mark.parametrize("doc", DOCS)
def test_every_documented_flag_exists(doc):
    text = (REPO / doc).read_text(encoding="utf-8")
    assert command_lines(text), f"{doc} shows no colosseum command"
    assert unknown_flags(main, text) == []


GUIDE_SECTIONS = ("Быстрый старт", "WSL2", "netsh interface portproxy", "Tailscale", "SSH", "облачн", "Docker",
                  "compose", "Kubernetes", "Безопасность", "--insecure", "--ban", "--rotate", "--token-file",
                  "COLOSSEUM_TOKEN", "Отказы", "--resume", "training.resume_from", "Агенты на лету", "update-code",
                  "hub --idle", "tournament --hub", "Расстановка лёрнеров", "learner.placement", "max_policy_lag",
                  "Узлы", "relay", "Логи узлов", "Типичные проблемы")


def test_the_distributed_guide_has_every_section_of_spec_block_11():
    text = (REPO / "docs/DISTRIBUTED_GUIDE.md").read_text(encoding="utf-8")
    missing = [s for s in GUIDE_SECTIONS if s not in text]
    assert missing == []


def test_the_removed_roles_appear_only_as_removed():
    for doc in (*DOCS, "CLAUDE.md"):
        for line in (REPO / doc).read_text(encoding="utf-8").splitlines():
            for name in REMOVED_COMMANDS:
                if name in line:
                    assert re.search(r"removed|удален|удалён|exit 2|SP2", line), f"{doc}: {line}"
```
(The checker was run while writing this part against today's README, ARENA_GUIDE and LEAGUE_GUIDE: 43 command lines, no unknown flag; and on the synthetic document above.) If part 08 moved to SP5b, the guide still has the SSH / agents / update-code / Kubernetes sections, each saying «перенесено в SP5b» with the workaround of spec §2 (`hub --resume` with a new config; `join` or the one-liner), so the section check stays.

Run: `env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_sp5_docs.py -q -rw`
Expected: FAIL (`docs/DISTRIBUTED_GUIDE.md` missing; README still lists the old roles as usable).

- [ ] **Step 2: `docs/DISTRIBUTED_GUIDE.md` (Russian)**

Sections, in this order (each with the real commands; numbers from `docs/benchmarks.md` «После SP5»):
1. **Что это** — hub and nodes in five lines; what runs where; the one external port (47100); link to the spec.
2. **Быстрый старт на двух машинах** — exactly the commands of T7.6 criterion 15 (they are repeated verbatim at the acceptance): machine A: `colosseum hub -c configs/examples/unit_harvest_league.yaml --set run.name=harvest-2m`; in a second terminal of A: `colosseum hub token > ~/.colosseum-token && chmod 600 ~/.colosseum-token` and `colosseum join 127.0.0.1:47100 --token-file ~/.colosseum-token --name pc`; machine B (no colosseum): the one-liner the hub printed, with A's LAN address; then `colosseum nodes list`, `colosseum report runs/harvest-2m`; stop: Ctrl+C on the hub. What the hub prints (join commands per address, the one-liner, the WSL2 hint) shown verbatim.
3. **Сеть: WSL2 и NAT** — how the hub detects WSL2 NAT and what it prints; Windows 10 (no mirrored mode, no Hyper-V firewall): `netsh interface portproxy add v4tov4 listenport=47100 listenaddress=0.0.0.0 connectport=47100 connectaddress=<WSL IP>` + `New-NetFirewallRule -DisplayName "Colosseum hub 47100" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 47100` (PowerShell as administrator), how to find the WSL IP (`wsl.exe hostname -I`), that it changes after `wsl --shutdown`, how to remove both rules; Windows 11 22H2+: `networkingMode=mirrored` + `New-NetFirewallHyperVRule` with the WSL VMCreatorId; Tailscale as the alternative (not both on Windows and in WSL); a machine behind NAT that joins needs nothing (it dials out); the node data port 47102 is optional (relay through the hub otherwise).
4. **SSH-режим и облачная VM** — `colosseum nodes add user@host ...`, the reverse tunnel, `--nodes hosts.txt`, remembered nodes (part 08; or «перенесено в SP5b» with `join` / the one-liner over SSH by hand).
5. **Docker и compose** — `deployment/Dockerfile.node` (`cpu` / `cuda`), the `docker run` line of spec block 10, `--shm-size`, the cache volume, `--gpus all`, `compose.yaml` (hub + 2 nodes), `compose.node.yaml` (one node; `--profile gpu`; `network_mode: host` for a direct data port); Docker Desktop on Windows needs its WSL integration.
6. **Kubernetes** — the manifests of T8.4 and their static check (or «перенесено в SP5b»).
7. **Безопасность** — the token `cst1.<секрет>.<отпечаток>`, what it protects, TLS by default, the CA fingerprint pin, node certificates and node secrets, `--insecure` (trusted LAN only: keys and data readable and forgeable; the token is still required), `--token-file` / `COLOSSEUM_TOKEN` (shell history), `nodes remove NAME --ban`, `hub token --rotate` (every node revoked; refused while a hub of this identity runs), the dashboard has no authentication (SSH tunnel).
8. **Отказы и что делать** — the table of spec 7.1 with the operator's action per row; the restart limits; what `failed` means and the exit code 1 at the end; spec 7.3 (not automated: the hub's own restart — `systemd` unit and `docker --restart unless-stopped` recipes shown; no HA; budget not enforced while the hub is away).
9. **Продолжение прогона** — `hub --resume RUN` / `train --resume RUN` in place (refused while the run's process lives; config precedence; `--set training.total_timesteps=...`; SP4 run dirs migrated in place), versus `training.resume_from` (a new child run, unchanged).
10. **Агенты на лету и обновление кода** — `agents add / freeze / remove / list`, `nodes update-code [--restart-learners]` (part 08, or «перенесено в SP5b» with `hub --resume` and a new config).
11. **Турнир на кластере** — `tournament --hub ADDR|RUN_DIR`, Ctrl+C detaches (130), `--cancel NAME`, blocks ahead of pulse; `hub --idle -c cfg.yaml` for selection after training; reports in `runs/x/tournaments/` of the candidates' run.
12. **Расстановка лёрнеров** — `learner.device` (`auto` / `cpu` / `cuda[:N]` / `mps`), slots per node (one per GPU, `--cpu-learners`, `--no-learners`, `--gpus`), `learner.placement.{node,tags,strict}`, `pending`, relocation after `distributed.learner_relocate_after_sec`; what `colosseum validate` prints (`placement:` lines and the `distributed:` line, shown).
13. **Метрики узлов, пути данных и лаг политики** — `system` keys (`nodes`, `by_node`, `relay_bytes`, `dropped_chunks`, ...), `nodes.json`, the console nodes line, the report's «Узлы» section, `hub --dashboard`; direct vs relay and how to open a node's data port; `policy_lag_p50/p95/max` and how to choose `learner.max_policy_lag` (from the measured p95 of a run: e.g. 2–3 × p95; default off because V-trace handles lag); `weight_fetches` (one network fetch per machine and version).
14. **Логи узлов** — `~/.cache/colosseum/logs/<ca8>/<run_id>/` on each node, `join.log` before the first Welcome, the 64 KB tails in `runs/<name>/logs/nodes/<node>/<proc>.log.tail`, `train`'s local node in `runs/<name>/logs/`.
15. **Типичные проблемы** — at least: the laptop cannot reach the hub (WSL2 NAT / firewall / wrong address), "code hash mismatch" (the node downloads the new code itself), Python or torch version mismatch, `/dev/shm` full (`docker --shm-size`, `emptyDir: Memory`), no CUDA wheel for an old driver, a package without a wheel for the node's platform (`distributed.requirements`), a node stuck `pending`, dropped chunks `overflow` (relay buffer; open the data port), `stale_generation` drops after a network split, the hub refusing `--resume` while the run lives.

- [ ] **Step 3: README, CLAUDE.md, ARENA_GUIDE, LEAGUE_GUIDE**

- `README.md`: the status block (SP5 done; what distributed mode does now); a "Distributed training" section replacing "Distributed mode (limited)" (the quick start of the guide in 6–8 lines, a link to `docs/DISTRIBUTED_GUIDE.md`, the limits that remain: no hub HA, macOS checked by design only, the shared-memory trajectory ring in SP6, SP5b items if moved); the configuration table: `transport` row replaced by `distributed` (every key of spec 9.1 with its default) and the agent keys `learner.max_policy_lag` / `recovery_interval_sec` / `placement.*`; "Process lifecycle": `hub` / `join` exit codes (spec 7.5), the three SP2 roles listed as removed (exit 2); "What a run writes": `hub.json`, `hub_state.json`, `nodes.json`, `blobs/`, `recovery/`, `logs/nodes/`; `metrics.jsonl` table: the SP5 `system` keys and `train` keys; "Tests": 16 slow tests, the `docker` marker; "Project structure": `hub/`, `node/`, `net/`, `bootstrap/` instead of `launcher.py`, `distributed.py`, `transport/`, `weight_store/`.
- `CLAUDE.md`: Implementation Status (SP5), Commands (`hub`, `join`, `nodes …`, `hub token`, `tournament --hub`, `hub --idle`, `hub --dashboard`, `train --listen`, `--resume`; the measurement scripts), "Works", "Partial", "Not implemented", the Roadmap row (SP5 done; SP5b if moved), the open items (SP5 items closed with a pointer; new parked items from the task reports), "Next step"; architecture sections whose "(target; see Roadmap — SP5)" notes are now real get their SP5 wording (weight store = per-machine shm cache; trajectories = connections + frames; coordinator = the hub).
- `docs/ARENA_GUIDE.md`: `tournament --hub` and `hub --idle` (section 4), pulse on evaluators (section 6: blocks, leases, `max_backlog` cancels leased blocks), the version line after a learner restart (`v(M+1)` older than `vM`, BT smoothness; section 3 or 5), a changed scripted bot keeps its entity name (add it under a new id to compare), the «Узлы» section and `hub --dashboard` (section 7); the paragraphs "распределённые роли журнал не пишут" removed.
- `docs/LEAGUE_GUIDE.md`: section 15 "Ограничения": nothing of SP3 is refused on several machines any more; a missing snapshot is no longer replaced by latest (the lineup is kept and `lineup_fetch_failed` counted); section 16 untouched.

- [ ] **Step 4: Run every command the guide shows**

On one machine: the quick start with two local nodes (`colosseum join 127.0.0.1:47100 ...` with `XDG_DATA_HOME` / `XDG_CACHE_HOME` of a scratch dir for the second node), `nodes list`, `nodes remove`, `hub token`, `validate`, `report`, `tournament --hub`, `hub --idle`, `hub --dashboard`. Fix the docs where the output differs (the task report lists each command and its result). Commands that need a second machine or Docker are marked in the report "checked at the acceptance (T7.6)".

- [ ] **Step 5: Full fast suite + ruff**

Run the full fast suite and ruff. Expected: green, zero warnings.

- [ ] **Step 6: Commit and push**

```bash
git add docs/DISTRIBUTED_GUIDE.md README.md CLAUDE.md docs/ARENA_GUIDE.md docs/LEAGUE_GUIDE.md tests/unit/test_sp5_docs.py
git commit -m "docs: DISTRIBUTED_GUIDE and the SP5 state in README, CLAUDE.md, ARENA_GUIDE, LEAGUE_GUIDE"
git push
```

---

### Task T7.6: Acceptance runbook (every criterion of spec §3: commands, thresholds, owner actions)

Spec §3 (criteria 1–15) and §7 (acceptance on one machine, then on two). No code: this task is the procedure the controller follows after the final whole-branch review, its fix wave and the scoped re-review, so the acceptance measures the final code (SP3 / SP4 practice). Its output is the acceptance report `docs/superpowers/reports/<date>-sp5-acceptance.md` (Russian; a private phone-readable artifact page plus a short Russian summary to the owner), with, per criterion, the commands run, the numbers, the threshold, PASS / FAIL / «не проверено: <причина>» / «перенесено в SP5b» (part 08 moved, R14), and the data path of every two-machine run.

**General rules (fixed here).**
- Thresholds are the ones written below and in T7.4; nothing is re-tuned after a measurement. A failed criterion is reported with its numbers; the controller proposes a fix or a ruling and the owner decides (the merge needs the owner's "yes" anyway).
- Every measurement and the slow suite run alone on a quiet machine: before each run `cat /proc/loadavg` ≤ 2.0 for the 1-minute value (recorded); never next to another CPU-heavy job.
- Data by counters (`env_steps`, versions, row ids); durations on `time.monotonic()` of one machine (the controller's PC) or counted by the controller's watch loop; the clocks of the two machines are never compared.
- Every run dir of the acceptance stays under `/tmp/sp5-acceptance/` on the PC (or `runs/acceptance-*` of a scratch copy of the repo); identity dirs of test nodes on the PC go to scratch `XDG_DATA_HOME` / `XDG_CACHE_HOME`; the owner's real identity is used only for the two-machine runs.
- Owner actions are batched into as few sessions as possible; each is announced in Russian with the exact command to type and what to report back.

**Machines and network for criteria 8–15.** PC: Windows 10 Pro 10.0.19045 + WSL2 (NAT; `wslinfo --networking-mode` says `nat`), RTX 4070 Ti, the repo in WSL. Laptop: Windows 10 + WSL2, no colosseum installed (`command -v colosseum` empty, no `~/.cache/colosseum`, no `~/.local/share/colosseum` in the laptop's WSL — the owner checks once at the start). Mirrored WSL networking and the Hyper-V firewall do not exist on Windows 10 (`research/sp5_node_environment.md` §5.1), so the hub port is opened with a port proxy:

Owner action O1 (PC, PowerShell **as administrator**, once; redo after every `wsl --shutdown` or PC reboot, because the WSL address changes):
```powershell
$ip = (wsl.exe hostname -I).Trim().Split(' ')[0]
netsh interface portproxy add v4tov4 listenport=47100 listenaddress=0.0.0.0 connectport=47100 connectaddress=$ip
New-NetFirewallRule -DisplayName "Colosseum hub 47100" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 47100 -Profile Private
netsh interface portproxy show v4tov4
ipconfig | Select-String IPv4
```
The owner reports the PC's LAN IPv4 (`PC_LAN`, e.g. `192.168.1.10`). After the acceptance (owner action O9): `netsh interface portproxy delete v4tov4 listenport=47100 listenaddress=0.0.0.0` and `Remove-NetFirewallRule -DisplayName "Colosseum hub 47100"`. The laptop's WSL needs nothing (it dials out). Only port 47100 is open, so every chunk of the laptop to a PC learner goes through the hub port (relay); the report states it from `nodes.json` (`data_path`).

---

- [ ] **Step 1: Criterion 1 — tests**

```bash
env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw         # zero failures, zero warnings
env -u PYTHONPATH .venv/bin/ruff check .
env -u PYTHONPATH .venv/bin/python -m pytest -m slow -v                                # alone, background; all green
env -u PYTHONPATH .venv/bin/python -m pytest -m gpu -q -W error                        # CUDA present: all green
env -u PYTHONPATH .venv/bin/python -m pytest -m docker -q -rs                          # skipped without Docker (criterion 14)
```
PASS when: the fast suite is green with 0 warnings, ruff clean, the slow suite green whole, the GPU suite green. Plus three audits written into the report:
- **process count:** `grep -rln "workers\|evaluators\|cpu-learners\|num_workers" tests/**/test_sp5_*.py` → for each hit the controller checks the test starts at most 2 loaded CPU processes (3 for slow tests); the per-task reviews recorded the same; any excess is a FAIL to fix;
- **files under tmp_path:** the suite's own guard (`tests/conftest.py` fails a test that writes into the repo root) and the identity-dir fixture of T1.3; `git status --porcelain` after the suites is empty;
- **no wall clock:** `grep -rn "time\.time()\|\[\"ts\"\]\|\.ts\b" tests/ | grep -v "^tests/fixtures"` → every hit in an SP5 test is informational, never asserted (reviewed by hand, listed in the report);
- **a contract test per spec item of blocks 1–9:** the table of the report is assembled from the traceability tables of parts 01–08 (spec item → task → test); each test is run by name (`pytest <file>::<test>`) and listed green.

- [ ] **Step 2: Criterion 2 — speed on one machine**

The commands of T7.4 Step 8 on the final commit (the T7.4 numbers are not reused). PASS when for 1, 2 and 4 workers mean(branch) / mean(main `d61eb02`) ≥ 0.95, two runs each, alternating main → branch → main → branch, by counters. If a worker count fails narrowly, one supplementary pair in reverse order is run and shown (SP4 practice) but does not change the verdict.

- [ ] **Step 3: Criteria 3, 4, 5 — hub load, weight cache, frame format**

The commands of T7.4 Step 9 on the final commit:
- 3: PASS when all five checks of `measure_hub_load.py` pass (≥ 20 000 results/s acked, 0 lost, lineup p95 ≤ 200 ms, the slowest lineup reply during a ≥ 1 800-entity refit ≤ 1 s, ≥ 2 such refits in the window).
- 4: PASS when the 50 MB read p95 ≤ 100 ms at 2 publishes/s **and** the fetch counter holds: T3.1's localhost test green (one network fetch per (machine, agent, version) whatever the worker count) and, in the two-machine runs of Steps 9–10, `nodes.json` of every run has `weight_fetches[agent].fetches == weight_fetches[agent].versions` for the laptop (`jq '.nodes[] | {name, weight_fetches}' runs/<name>/nodes.json`).
- 5: PASS when `measure_frame_decode.py` reports ≥ 10 × and the hostile-input tests (T1.1, T7.2's ports) are green.

- [ ] **Step 4: Criterion 6 — compatibility**

```bash
W=/tmp/sp5-acceptance/compat; rm -rf $W && mkdir -p $W
cp -r tests/fixtures/sp4/run $W/sp4-run                                   # the SP4 run dir made by SP4 code (T0.1)
env -u PYTHONPATH .venv/bin/colosseum train --resume $W/sp4-run --set training.total_timesteps=<fixture env_steps + 3000>
env -u PYTHONPATH .venv/bin/colosseum report $W/sp4-run                  # the continued run reports
cp -r tests/fixtures/sp4/run $W/sp4-parent
env -u PYTHONPATH .venv/bin/colosseum train -c tests/fixtures/sp4/<fixture config> --set training.resume_from=$W/sp4-parent \
    --set run.dir=$W --set run.name=child --set training.total_timesteps=3000
for c in configs/examples/*.yaml; do env -u PYTHONPATH .venv/bin/colosseum validate -c $c || echo "FAIL $c"; done
env -u PYTHONPATH .venv/bin/python -m pytest tests/unit/test_example_configs.py tests/integration/test_demo_train_smoke.py -q
for cmd in serve-weight-store run-learner run-workers; do env -u PYTHONPATH .venv/bin/colosseum $cmd; echo "exit $?"; done
env -u PYTHONPATH .venv/bin/colosseum validate -c configs/examples/tic_tac_toe_multi.yaml | grep -E "placement:|distributed:"
# two train side by side (two terminals, the same moment):
env -u PYTHONPATH .venv/bin/colosseum train -c configs/examples/tic_tac_toe.yaml --set run.dir=$W --set run.name=a
env -u PYTHONPATH .venv/bin/colosseum train -c configs/examples/tic_tac_toe.yaml --set run.dir=$W --set run.name=b
```
PASS when: the SP4 dir continues in place (same `run_id` in `run.json`, the journal migrated: `sqlite3 $W/sp4-run/matches.db "PRAGMA table_info(matches)"` has `node`, schema version 2 in `meta`, `run_state.json` `finished`, exit 0) and as a child run (exit 0, a new run id, the parent untouched); every example config validates and trains (the example smoke tests green: every `configs/examples/*.yaml` trains on one machine without edits); each old command prints the hint and exits 2; `validate` prints the `placement:` lines and the `distributed:` line; both trains exit 0 with different loopback ports in their `hub.json` and different shm prefixes. (`<fixture env_steps>` and `<fixture config>` come from `tests/fixtures/sp4/manifest.json`.)

- [ ] **Step 5: Criterion 7 — security**

The library and integration tests of T1.3 and T6.3 by name (token missing / wrong, certificate missing / from another CA, wrong fingerprint, plain client against TLS, `--insecure` without the token, `--ban`), then by hand on the PC with two local nodes (scratch identity roots):
```bash
env -u PYTHONPATH .venv/bin/colosseum hub -c configs/examples/tic_tac_toe.yaml --set run.dir=/tmp/sp5-acceptance --set run.name=sec &
T=$(env -u PYTHONPATH .venv/bin/colosseum hub token)
env -u PYTHONPATH .venv/bin/colosseum join 127.0.0.1:47100 --once; echo "no token: exit $?"                       # 2 or 1, refused
env -u PYTHONPATH .venv/bin/colosseum join 127.0.0.1:47100 --token "cst1.$(head -c 32 /dev/urandom | base64 | tr '+/' '-_' | tr -d '=\n' | head -c 43).${T##*.}" --once; echo "wrong secret: exit $?"   # 1
env -u PYTHONPATH .venv/bin/colosseum join 127.0.0.1:47100 --token "${T%.*}.$(printf '0%.0s' $(seq 64))" --once; echo "wrong fingerprint: exit $?"  # 1
env -u PYTHONPATH .venv/bin/colosseum join 127.0.0.1:47100 --token "$T" --insecure --once; echo "plain against TLS: exit $?"  # 1
env -u PYTHONPATH .venv/bin/colosseum join 127.0.0.1:47100 --token "$T" --name banme --once &                    # joins
env -u PYTHONPATH .venv/bin/colosseum nodes remove banme --ban
env -u PYTHONPATH .venv/bin/colosseum join 127.0.0.1:47100 --token "$T" --name banme --once; echo "banned: exit $?"  # 1
kill -INT %1
# --insecure hub still needs the token:
env -u PYTHONPATH .venv/bin/colosseum hub -c configs/examples/tic_tac_toe.yaml --insecure --set run.dir=/tmp/sp5-acceptance --set run.name=insec &
env -u PYTHONPATH .venv/bin/colosseum join 127.0.0.1:47100 --insecure --once; echo "insecure, no token: exit $?"   # refused
kill -INT %1
```
PASS when every refusal happens with a message that names the reason, and the banned node (same `machine-id`) cannot register again.

- [ ] **Step 6: Criterion 8 — joining the laptop (two machines)**

Preconditions: O1 done; `PC_LAN` known. On the PC:
```bash
cd ~/dev/repos/Colosseum
env -u PYTHONPATH .venv/bin/colosseum hub -c configs/examples/unit_harvest_league.yaml --set run.dir=/tmp/sp5-acceptance \
    --set run.name=join-check --set training.total_timesteps=20000000 2>&1 | tee /tmp/sp5-acceptance/join-check.out
```
The hub prints the join commands and the one-liner per address and the WSL2 NAT hint. The controller copies the one-liner and replaces the WSL address with `PC_LAN` (the address in the URL; the pinned key is the same).

Owner action O2 (laptop, WSL terminal): paste the one-liner, followed by ` -s -- --name laptop --evaluators 1` after `sh` (so: `... | COLOSSEUM_TOKEN='cst1…' sh -s -- --name laptop --evaluators 1`), and leave the terminal open. The owner reports when the terminal prints that the node is connected.

PASS (one-liner) when, on the PC, `colosseum nodes list` shows `laptop` online with its hardware, data path `relay`, and the laptop's `~/.cache/colosseum/bundles/<hash>/` is the hub's code hash (`nodes list` code hash = the hub's).

Version mismatch: with the laptop connected, the controller stops the hub (Ctrl+C), edits a comment in `examples/unit_harvest/bots.py` (the bundle hash changes; reverted at the end of the step), and starts the hub again with `--resume /tmp/sp5-acceptance/join-check`. PASS when the hub log names the refusal ("code hash … differs …"), the laptop node downloads the new bundle and registers again by itself (no owner action) within 2 minutes, and `nodes list` shows the new code hash for `laptop` with the same node number.

SSH mode (part 08; else «перенесено в SP5b»): if the owner enabled sshd in the laptop's WSL (optional owner action O8): `colosseum nodes add <user>@<laptop LAN IP> --name laptop-ssh --workers 1`; otherwise on a container or `localhost` (`ssh localhost` with the controller's key) — the report says where. PASS when the SSH node appears online and plays matches.

Stop the hub (Ctrl+C) at the end of the step; the laptop node stays connected (persistent).

- [ ] **Step 7: Criterion 9 — scaling (two machines)**

On the PC, the laptop node connected (from Step 6, persistent):
```bash
cd ~/dev/repos/Colosseum; D=/tmp/sp5-acceptance/scaling; mkdir -p $D
for w in ttt-gpu harvest-gpu harvest-gpu-b16; do                                           # calibration, in order
  env -u PYTHONPATH .venv/bin/python scripts/measure_scaling.py capacity --workload $w --json $D/cap-$w.json
  env -u PYTHONPATH .venv/bin/python scripts/measure_scaling.py run --variant a --workload $w --json $D/cal-a1-$w.json
  env -u PYTHONPATH .venv/bin/python scripts/measure_scaling.py run --variant a --workload $w --json $D/cal-a2-$w.json
  env -u PYTHONPATH .venv/bin/python scripts/measure_scaling.py run --variant b --workload $w --laptop laptop --json $D/cal-b-$w.json
  env -u PYTHONPATH .venv/bin/python scripts/measure_scaling.py rule --capacity $D/cap-$w.json \
      --runs $D/cal-a1-$w.json $D/cal-a2-$w.json $D/cal-b-$w.json && { echo "chosen: $w"; W=$w; break; }
done
for i in 1 2; do for v in a b c; do                                                        # the measured runs
  env -u PYTHONPATH .venv/bin/python scripts/measure_scaling.py run --variant $v --workload $W --laptop laptop --json $D/$v$i.json
done; done
env -u PYTHONPATH .venv/bin/python scripts/measure_scaling.py verdict --runs $D/a1.json $D/b1.json $D/c1.json $D/a2.json $D/b2.json $D/c2.json
```
The rule (fixed in T7.4): the first workload in the order ttt-gpu, harvest-gpu, harvest-gpu-b16 whose two calibration runs of variant a are worker-bound and whose fed-chunk learner capacity ≥ 1.2 × (a + b). PASS when a workload is chosen and c / (a + b) ≥ 0.85 over the means of two runs per variant (order a, b, c, a, b, c; by counters). No workload satisfies the rule → «не проверено: ни одна нагрузка-кандидат не удовлетворяет правилу» with every number. The laptop's data path (relay) and its `weight_fetches` are recorded (criterion 4).

- [ ] **Step 8: Criterion 10 — the README pipeline across machines**

On the PC (record and BC are local commands, as in the README):
```bash
cd ~/dev/repos/Colosseum; P=/tmp/sp5-acceptance/pipeline; mkdir -p $P
cp configs/examples/unit_harvest_league.yaml $P/harvest.yaml
# edit $P/harvest.yaml as the README shows: main.init {from: bc_net, critic_warmup_steps: 30}, main.kickstart {teacher: greedy},
# bc_net {kind: frozen, path: <P>/bc.pt}, evaluation {enabled: true, every: 1, blocks: 25}, training.seed: 0
env -u PYTHONPATH .venv/bin/colosseum record -c $P/harvest.yaml --player greedy --num-matches 300 --output $P/data/greedy --seed 0
env -u PYTHONPATH .venv/bin/colosseum bc -c $P/harvest.yaml --agent main --data $P/data/greedy --output $P/bc.pt --epochs 5 --seed 0
env -u PYTHONPATH .venv/bin/colosseum hub -c $P/harvest.yaml --set run.dir=$P --set run.name=league --dashboard 2>&1 | tee $P/hub.out &
T=$P/token; env -u PYTHONPATH .venv/bin/colosseum hub token > $T; chmod 600 $T
env -u PYTHONPATH .venv/bin/colosseum join 127.0.0.1:47100 --token-file $T --name pc --evaluators 0 --once     # PC node, own process
# ... the hub reaches the budget and exits 0; the laptop node (persistent, --evaluators 1) ran the pulse
env -u PYTHONPATH .venv/bin/colosseum hub --idle -c $P/harvest.yaml --set run.dir=$P --set run.name=idle &      # selection on the cluster: the laptop node (persistent) joins it and evaluates
env -u PYTHONPATH .venv/bin/colosseum tournament --hub $P/idle --run $P/league --agent main --snapshots all \
    --field greedy,random --mode select --seed 7 --name sel
kill -INT %<idle hub>
NEW=$(ls -d $P/league/checkpoints/main/ckpt_v* | sort -V | tail -1)
env -u PYTHONPATH .venv/bin/colosseum eval -c $P/harvest.yaml -a trained=$NEW -a greedy -a random --layout 2p \
    --num-matches 60 --seed 12345 --deterministic --output $P/eval-newest.json
env -u PYTHONPATH .venv/bin/colosseum report $P/league
```
Checks (fixed here): win of the newest snapshot against `random` ≥ 0.95 and score (win + draw / 2) against `greedy` ≥ 0.40 (the SP3 pipeline thresholds, calibrated on the newest snapshot; `eval-newest.json`); the select winner's eval is reported, not judged; pulse ρ ≥ 0.85 (`RHO_MIN`) — the Spearman correlation of the BT rating with the version over the report's ratings section (`tests/arena_kit.rating_trend` applied to `build_report_data($P/league)`); the journal has train matches of both machines' nodes (`sqlite3 $P/league/matches.db "SELECT source, node, COUNT(*) FROM matches GROUP BY source, node"`: `train` rows with the PC node's and the laptop's numbers, `pulse` rows with the laptop's number); duplicates 0 (the last `system` record's `match_log.duplicates`); per node, `results_sent == train rows of the node + results_dropped` (`nodes.json`); `dropped_match_results == 0`; the tournament exits 0 (`confirmed` printed). PASS when all hold. A warm-started run can rate flatter by version than SP4's from-scratch calibration; if ρ alone fails, the report says so with the numbers (the threshold is the spec's).

- [ ] **Step 9: Criterion 11 — failures (two machines)**

One long run on the PC hub with a two-agent config so the relocation check fits in: `$P/harvest.yaml` with `training.total_timesteps` 50 000 000 (stopped by Ctrl+C at the end), `training.seed: 0`, `rollout.match_refresh_interval_sec: 30`, and a second trainable agent `cpu_agent` (`learner.device: cpu`, `learner.placement.node: laptop`); the PC node `join ... --name pc --cpu-learners 1 --evaluators 0` (not `--once`); the laptop node persistent. The controller runs a watch loop on the PC that samples `runs/<name>/nodes.json` / `hub_state.json` / the newest `system` record every second with `time.monotonic()` and the counters; every duration below is measured by that loop.

| Check | How (owner action marked O) | PASS when |
|---|---|---|
| laptop without network 5 s | O3: Wi-Fi off, count to 5, Wi-Fi on; the owner says "now" at both moments | `nodes_lost` unchanged, the laptop keeps its node number and session continues (hub log: no "lost"), its cells keep playing after the gap (journal rows with its node number continue) |
| laptop without network 60 s | O4: Wi-Fi off for 60 s, then on | during the gap the PC's env steps keep growing; `nodes_lost` +1 after ~10 s; after Wi-Fi is back the laptop is online again ≤ 30 s (watch loop: from the owner's "on" to `state: online`), same number |
| `kill -9` of the learner | PC: `kill -9 $(grep -o 'started (pid [0-9]*)' ~/.cache/colosseum/logs/<ca8>/<run_id>/learner-main.log \| tail -1 \| grep -o '[0-9]*')` | a new learner incarnation announces a version ≤ 60 s later; the announced versions never decrease (`hub_state.json` `versions.main.max_version` sampled every second is non-decreasing); `learner_restarts` +1; the PC node's workers keep running (`by_node` env steps/s > 0 during the restart) |
| `kill -9` of a worker | PC: the same with `worker-0.log` | `worker_restarts` +1; other workers and the learner keep running; no repeated uid (`match_log.duplicates` 0) and no repeated seed among train rows (`sqlite3 … "SELECT seed, COUNT(*) c FROM matches WHERE source='train' GROUP BY seed HAVING c > 1"` empty) |
| `kill -9` of the hub, `hub --resume` | PC: note `max_version` of `main`, `kill -9 $(jq .pid $RUN/hub.json)`, wait 60 s, `colosseum hub --resume $RUN` | both nodes online ≤ 30 s after the resume started; `main`'s `max_version` after the resume > the value before the kill (the PC node's learner trained in the gap, and the laptop's journal rows after the gap carry `version_start` above the pre-kill maximum: the workers got new versions through `GetLatest` / relay); the results not acked before the kill are delivered: duplicates 0 and per node `results_sent == train rows + results_dropped` at the end |
| relocation of `cpu_agent` | O5: laptop Wi-Fi off for 150 s (> `learner_relocate_after_sec` 120 s) | the hub log names the relocation of `cpu_agent` to `pc` with generation +1; `cpu_agent`'s versions continue above the last one; its learner runs on the PC's CPU slot (`nodes.json` roles) |
| Ctrl+C on the hub | PC: Ctrl+C in the hub terminal (watch loop timestamps it) | ≤ 30 s later: the hub has exited (130), the PC node's children are gone (`ps`), the hub log has the laptop node's stop confirmation, each trainable agent has a final snapshot of this run (`meta.json` `run_id` = this run's, the newest version) |

O6 (once, at the end of the step, as spec §3 asks): the owner closes the laptop node's terminal; the hub log records the laptop as lost after ~10 s (no hang on the PC).

- [ ] **Step 10: Criterion 12 — agents on the fly (part 08)**

If part 08 ran: the config of Step 9 with `opponents.anchors` > 0 and `match_refresh_interval_sec` 30; `colosseum agents add --id greedy2 -c greedy2.yaml` (a scripted agent with an anchor weight) → it appears in lineups within one refresh (journal rows with `greedy2` ≤ 30 s + one episode after the command, counted by the watch loop); `agents add --id rookie -c rookie.yaml` (trainable) → it gets a learner and its versions grow; `agents freeze rookie` → a final snapshot `rookie@v<N>` becomes a frozen entity; `agents remove greedy2` → no new lineups with it; `nodes update-code` after editing `examples/unit_harvest/bots.py` → workers restart, the run continues, every node shows the new code hash. Else: «перенесено в SP5b».

- [ ] **Step 11: Criterion 13 — report and dashboard**

`colosseum report $P/league` and the report of Step 9's run: the «Узлы» section lists `pc` and `laptop` with hardware, roles, load, rates, last heartbeat and data path; the training section shows the policy lag. The controller publishes the Step 8 report as a private artifact page (as for SP4). Owner action O7: open it on the phone and say whether it reads well. PASS when both reports build (exit 0) with the section, and the owner confirms the phone check. `hub --dashboard` of Step 8 answered `/api/nodes` during the run (checked by the controller with `curl -s http://127.0.0.1:8765/api/nodes`).

- [ ] **Step 12: Criterion 14 — packaging**

`docker info` in the PC's WSL. Exit 0 (the owner turned on Docker Desktop's WSL integration: optional owner action O8b) → `pytest -m docker -v` (image build, `--help`, compose to the budget with exit 0) and the commands of `deployment/compose.yaml`'s header by hand; PASS when both pass. Otherwise «не проверено: интеграция Docker Desktop с WSL выключена». K8s: T8.4's static check (`kubeconform` if present, else the structural test) or «перенесено в SP5b».

- [ ] **Step 13: Criterion 15 — documentation**

The quick start of `docs/DISTRIBUTED_GUIDE.md` is repeated verbatim on the two machines (a fresh run name; the owner pastes the laptop's command from the guide, owner action O2 again if the node is not connected), and its output matches the guide. `tests/unit/test_sp5_docs.py` green (flags = `--help`); additionally `colosseum <cmd> --help` of `hub`, `join`, `nodes list`, `nodes remove`, `hub token`, `tournament` diffed by eye against the guide's option lists. README and CLAUDE.md reviewed against the final state (status, commands, open items). PASS when the quick start works as written and the reviews find no mismatch (or the mismatches were fixed before the report).

- [ ] **Step 14: The report and the hand-over**

`docs/superpowers/reports/<date>-sp5-acceptance.md` (Russian): the criteria table (PASS / FAIL / не проверено / перенесено в SP5b with numbers and commands), every two-machine run's data path, the measurements (with `docs/benchmarks.md` links), every ruling of SP5 (R1–R18, the pre-flight P-rulings, the controller rulings of the execution, the review and fix-wave rulings) verbatim, the open items by addressee, the deferred minors, questions for the owner; published as a private artifact page; a short Russian summary to the owner; owner action O9 (remove the port proxy and the firewall rule) offered. Merge only after the owner's explicit "yes".

**Owner actions, all together** (Russian announcement to the owner before the two-machine session): O1 open port 47100 on the PC (PowerShell as administrator, commands above) and tell `PC_LAN`; O2 run the hub's one-liner on the laptop with ` -s -- --name laptop --evaluators 1`; O3 / O4 / O5 Wi-Fi off for 5 s, 60 s, 150 s on the controller's signal; O6 close the laptop node's terminal once; O7 look at the published report on the phone; optional O8 sshd in the laptop's WSL (SSH mode) and O8b Docker Desktop's WSL integration on the PC (criterion 14); O9 remove the port proxy and the rule afterwards.

---

## Contract notes

### Additions and deviations versus `00-overview.md`

1. **`colosseum.metrics.nodes` (new module, T7.1):** `NodeView`, `NodeRates`, `node_rows`, `by_node`, `nodes_payload`, `cluster_record`, `cluster_defaults`, `CLUSTER_KEYS`, `NODES_FILE = "nodes.json"`, `nodes_line`, `weight_fetches`, `DROPPED_CHUNK_REASONS`, `DROPPED_RESULT_REASONS`. The overview names the metrics files only as `metrics/{aggregator,hub,jsonl,console}.py`.
2. **`system` key `live_ratings`** (`{"version", "refit_sec", "entities"}` of the newest live refit; nulls / 0 without one) — added to the overview's SP5 `system` keys and to `REQUIRED_KEYS["system"]`; the hub load measurement (criterion 3) locates the refits with it, and the dashboard shows it.
3. **`by_node` cells gain `state`; `nodes.json` rows gain `env_steps`, `results_journaled`, `results_sent`, `results_dropped`, `weight_fetches` (`{agent: {"fetches", "versions"}}`)** — for criterion 10 ("sent = journaled + counted losses" per node) and criterion 4 ("one fetch per (machine, agent, version)").
4. **`Heartbeat.counters` (`map<string, int64>`, a new field of `colosseum_v1.proto`, T7.1 regenerates):** node-side cumulative counters `results_sent`, `results_dropped_buffer`, `weight_fetches/<agent>`, `weight_versions/<agent>`. If T3.1 / T2.4 already carry equivalent counters to the hub, T7.1 uses those and drops this field.
5. **The node sends a final `Heartbeat` after its final result flush** (stop order: stop children → flush results → wait for acks within the grace → final heartbeat → close), so the hub's last `nodes.json` covers every sent result (T4.4's stop path; edited by T7.1).
6. **Hub counters `counters["node_counters"]` (`{"<node_id>/<session_id>": {name: value}}`) and `counters["node_results"]` (`{"<node number>": n}`)** in `hub_state.json`; T4.3's save / load carries the `counters` dict whole.
7. **`SystemStats.snapshot(..., cluster=None)`, `MetricsHub.maybe_tick(..., cluster=None)` / `close(..., cluster=None)`** (`cluster`: a callable returning `(cluster_record, node rows)`, called only when the tick is due), **`ConsoleReporter.nodes_line(rows, nodes_lost)`**. `dropped_match_results` = workers' `dropped_results` + the hub's `dropped_match_results_by_reason` sum + the writer's `match_log.dropped`.
8. **Report:** `SECTIONS` gains `"nodes"` right after `"system"`; the training section gains `"lag"` (`{agent: {"train_step", "p50", "p95", "max"}}`); `arena.report.data.hardware_text`, `roles_text`, `LAG_KEYS`, `NODE_SERIES`; `page.js` `R.nodes`; `tests/report_kit.sample_report_data` carries a `nodes` section and `training.lag`.
9. **`arena.report.server.dashboard_child(run_dir, port, conn)`; `Hub._start_dashboard()` / `_stop_dashboard()`;** `hub --dashboard [PORT]` is a click option with `is_flag=False, flag_value=8765`, `IntRange(0, 65535)`; only on `colosseum hub` (spec 9.2), not on `train`.
10. **`utils.logging`:** `setup_process_logging(..., keep_until_file=False)`, `PendingRecords`, `flush_pending(path, process_name)`, `discard_pending()`, `PENDING_LIMIT = 10_000`; the node's pre-Welcome log `<cache_home>/colosseum/logs/<ca8>/join.log` (the overview has only `node_log_dir(ca8, run_id)`).
11. **`hub.placement.describe_distributed(config) -> str`** next to the overview's `describe_plan(config)` (T7.2 implements both; `validate_config` appends them in the train scope only).
12. **`cli.REMOVED_COMMANDS`** and three hidden commands that accept any arguments and exit 2 (instead of deleting the names, so a script calling them gets the hint, spec 4.10).
13. **`deployment/node-entrypoint.sh`** (new file) and the `Dockerfile.node` targets `node` (default, last stage) and `hub` (compose only); compose service names `hub`, `node-1`, `node-2`; `compose.node.yaml` services `node` and `node-gpu` (profile `gpu`).
14. **`docker` marker** is also `slow` on every test that carries it, and `tests/conftest.py` skips it when `docker info` fails.
15. **Scripts:** `scripts/measure_common.py` (new, shared by the two SP5 measurement scripts), `scripts/measure_hub_load.py` (`run(args) -> dict`, used by the slow test), `scripts/measure_scaling.py` (subcommands `capacity`, `run`, `rule`, `verdict`; `--dry-run`).
16. **Possibly new node helpers (T7.4, only if missing):** `node.learner_main.algorithm_factory(cfg, agent)` and `node.worker_main.collect_chunks(cfg, agent, n) -> (chunks, env_steps)` for the learner-capacity measurement of criterion 9.
17. **Test support:** `cluster_kit.hub_process(...)` (context manager: `address`, `token_file`, `root`, `env`, `wait`, `interrupt`, `tail`) and `cluster_kit.start_join(tmp_path, address, token_file, *, identity, args)` if T3.1 / T6.3 named them otherwise.
18. **The slow suite grows to 16 tests** (the two of T7.4); the CI command is unchanged.

### Proposed rulings (what — why — cost if wrong)

1. **`hub --dashboard` runs the SP4 dashboard in a child process of the hub** (spawned, `watch_parent`, 127.0.0.1) — why: a dashboard build reads the whole journal and fits BT in Python; inside the hub process it would hold the GIL the main loop needs (criterion 3's lineup latency); the dashboard only reads the run dir — cost: one more mostly idle process.
2. **A `--dashboard` port that cannot be bound stops the hub at start with exit 1** (SP4's bind message), before nodes are accepted — why: the user asked for it explicitly; a run without the asked-for page is a silent surprise — cost: a restart with another port.
3. **The console nodes line counts the run's cumulative `nodes_lost`** (not the nodes lost right now) — why: the same number as the `system` key; a lost node that returns would otherwise vanish from the line — cost: a long run's line shows old losses.
4. **`nodes.json` keeps rows of lost and removed nodes** (rates 0) and is written once more at the end, after the nodes' final heartbeats — why: the report of a finished run shows every machine that took part — cost: none.
5. **Records logged before the log file exists are kept in memory (≤ 10 000, INFO and above) and written at the top of the file; an invalid config's records are discarded** (no run dir is created, SP2 rule kept); a node that never got a Welcome writes them to `join.log` — why: spec 4.10 asks the checks' warnings in the log, and SP2 / SP3 require an invalid config to leave nothing behind — cost: records of a crash before the first flush exist only on the console.
6. **Docker tests are `docker` and `slow`** — why: an image build downloads Python, uv and torch (minutes); the CI command `-m "not gpu and not slow"` must stay fast on a machine where Docker works — cost: they run only with `-m docker` or in the slow suite.
7. **The CUDA node image is the CPU image plus the NVIDIA runtime variables** — why: torch's CUDA wheels bring the CUDA runtime, the driver comes with `--gpus`; the node's environment (and its torch build) is chosen per node by the hub's bootstrap anyway — cost: a host without the NVIDIA Container Toolkit gets a CPU node (the node's torch check warns).
8. **The compose hub is the target `hub` of the same Dockerfile** (the project installed with CPU torch) — why: spec block 10 asks one node image; a second Dockerfile for a compose-only hub would be one more file to keep — cost: the `hub` target installs torch at build time.
9. **The localhost distributed pipeline test uses the SP4 pipeline test's thresholds** (ρ ≥ 0.85, gain > 0, select winner vs random ≥ 0.95) — why: same game, budget, one rollout worker and the same pulse settings; only the process placement differs — cost: if it is flakier than SP4's, a recalibration by the T6.4 rule.
10. **Its topology is node alpha = learner + evaluator, node beta = worker** — why: ≤ 3 loaded processes per slow test, and every chunk then crosses nodes (the subject of the test) — cost: no worker on the learner's node in this test (T2.x / T3.x tests cover that path).
11. **Criterion 3 counts "during the BT refit" as the requests sent inside a refit window derived from the hub's `system` records** (`[t_prev − refit_sec − 1.25 s, t_cur]`), requires ≥ 2 refits with ≥ 1 800 entities in the window, and judges results/s by acks (after the journal commit, R8) — why: the hub publishes no refit start; the window is a conservative superset; acks are what "the hub accepts" means under R8 — cost: a slightly wider "during refit" set (stricter).
12. **Criterion 9's candidate workloads and their order are ttt-gpu, harvest-gpu, harvest-gpu-b16, and variant a is `colosseum train` on the PC** — why: the spec fixes the rule, not the workload; the bench workload comes first (comparable with criterion 2), the heavier unit_harvest env makes the PC worker-bound with a CUDA learner, a larger batch raises the learner's capacity; `train` is "the PC alone" in the product's own terms — cost: if none qualifies, criterion 9 is «не проверено» with the numbers.
13. **Criterion 10's thresholds are judged on the newest snapshot (SP3's calibration basis); the select winner's eval is reported only; ρ is computed from the run's report with `arena_kit.rating_trend`** — why: the SP3 thresholds were calibrated on the newest snapshot, the SP4 ρ on the report's BT by version — cost: a run whose winner is weaker than its newest passes.
14. **Criterion 11's evidence is fixed by the table of T7.6 Step 9** (counters and node states sampled every second on the PC's monotonic clock; the owner's Wi-Fi actions announced by "now") — why: the clocks of the two machines are never compared — cost: ± 1 s on the 30 s / 60 s bounds.
15. **`validate`'s placement lines are config-only** (no hardware probe) — why: `validate` runs on any machine, also one that will not train; the hub places by the nodes' reports at run time — cost: `validate` cannot say whether this machine has the GPU a `cuda:1` agent needs.
16. **The hub persists per-node counters in `hub_state.json` `counters`** — why: "sent = journaled + losses" per node must hold across `hub --resume` (criterion 11) — cost: a few hundred bytes per session in the file.
17. **Complex dtypes are refused by the frame codec** (the SP1 npz codec accepted them; noted when porting `test_sp2_serialization`) — why: the contract's whitelist; no model of the repo has complex weights — cost: a user model with complex parameters cannot be distributed (an error names the dtype).

### Assumptions about other parts

1. **T2.5:** `Hub` keeps the R5 names (`_open_metrics`, `_finish_metrics`, `_on_session_opened`, `_shutdown`, the metrics tick of the main loop calling `MetricsHub.maybe_tick`); `run_hub(..., dashboard=None)` exists; `HubOptions.dashboard_port`; `hub` accepts `--port 0` (bound port in `hub.json` `addresses`); `bench_throughput.run_one` falls back to `Launcher` for `--tree`; `run_training` keeps the SP4 INFO "Training finished" at exit 0.
2. **T2.1 / T2.5:** `NodeRegistry` lists every node of the run (online, lost, removed) with `last_heartbeat` on the hub clock; `BudgetCounter.state()["sessions"]` is keyed `<node_id>/<session_id>`; the hub's run-level counters (`worker_restarts`, `learner_restarts`, `lineup_fetch_failed`, `relay_bytes`, `dropped_chunks`, `dropped_match_results`, `nodes_lost`) live in one dict saved in `hub_state.json`.
3. **T2.4:** `Heartbeat.procs` names worker and evaluator processes `worker-<i>` / `evaluator-<i>` with state `running`; the node logs each child's "started (pid N)" line in its own log file (used by the runbook's `kill -9`).
4. **T3.1:** the node counts its network fetches of latest weights where it fetches them; its criterion-4 test exists.
5. **T4.1:** the placement plan exposes `assigned` (agent → `LearnerSlot`) and `pending`; `describe_plan` is either absent or a stub that T7.2 replaces.
6. **T4.3:** `hub_state.json` `counters` is saved and loaded whole (opaque keys survive); the SP4 fixture `tests/fixtures/sp4/` and its `manifest.json` give the run's env steps and config for the runbook.
7. **T4.4:** the node's stop path can send one more heartbeat after the final result flush; `watch_parent()` is in `colosseum.node.procs`.
8. **T5.1 / T5.2:** pulse results carry the evaluator's node number; `tournament --hub RUN_DIR` reads `hub.json`; `hub --idle` takes nodes for evaluation only; a tournament on an idle hub over another run's snapshots writes `runs/x/tournaments/<name>/`.
9. **T6.2:** the loader runs as `python3 loader.py ADDR [join flags]` with stdlib + grpcio (if it needs protobuf, T7.3 adds it to the image); `join --help` through the loader prints the join flags; the one-liner's installer passes `"$@"` to `join` (so `sh -s -- --name laptop` works) and joins the address the one-liner reached.
10. **T6.3:** `colosseum hub token` prints the token alone and creates the hub identity when missing; `join --workers 0` is allowed (a learner / evaluator-only node); `join` retries an unreachable hub with the session backoff instead of exiting at the first failure; a node started again with the same identity root keeps its number; dev-venv `join` processes are accepted on localhost without uv or network (cluster_kit).
11. **T1.1 / T1.5:** `scripts/measure_frame_decode.py` and `scripts/measure_weight_cache.py` take `--json PATH` and default to the criterion's workload.
12. **T1.4:** `distributed.*` keys can be set with `--set` and in YAML; `load_config(path, {"distributed.insecure": True})` works.
13. **Part 08:** if it moved to SP5b (R14), the runbook marks criteria 8 (SSH part), 12 and 14 (K8s part) «перенесено в SP5b», and the guide's sections say so.

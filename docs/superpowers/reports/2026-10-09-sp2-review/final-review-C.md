> Origin: `.superpowers/sdd/2026-10-08-sp2-game-model/final-review-C.md` (SP2 SDD workspace, git-ignored, deleted after the merge). Copied 2026-10-09.
> Frozen record; do not edit. Only absolute scratch paths were replaced with `<scratch>/`.

# SP2 final whole-branch review — AREA C (league, orchestration, CLI, docs, acceptance report)

Range `7430361..7d55974` (branch `sp2-game-model`). Read-only review; experiments in
`<scratch>/final-review/C/` (`mm_check.py`, `eval_sched.py`). Area unit tests run
(`test_sp2_matchmaker`, `_ratings`, `_coordinator`, `_eval_schedule`, `_eval_report`, `_checkpoints`,
`test_config_v2`, `test_sp2_metrics`, `_distributed_roles`, `test_process_lifecycle`): 176 passed.

## Strengths

- **Matchmaking is one general algorithm.** `LineupMatchmaker` (`coordinator/matchmaker.py`) has no player-count
  branches. A scratch experiment (`mm_check.py`, 4000 lineups per owner) covered FFA 2p/4p (self-play and a
  2-agent league), 2v1v1 with `teammates: mixed`, coop and solo, asymmetric 1v2, layouts whose teams mix roles,
  and 3 agents with two hunters. In every lineup:
  - every seat's role is played by its agent;
  - `collect == (network_id == latest)`;
  - the owner always has a collecting latest seat (PR-2), so "every trainable agent owns envs in rotation" holds.
  - The rotation itself (`Coordinator.generate_lineups`) is SP1's `(g + round) % n`.
- **A seat of a role the core cannot play** goes to the latest weights of a uniform player of that role, owner
  included (`_fill_team`, l.201-205). This matches spec block 7 item 4.
- **Ratings (`coordinator/ratings.py`) credit the right side.**
  - `member_pairs` scores from team A's side. A `past` pair is flipped to the latest seat's side (l.58-61).
  - Teammates are never compared. Same-entity pairs are skipped and leave the divisor.
  - For 3 teams ranked 1/2/3 with K=32 the deltas are +16 / 0 / −16, i.e. K·(T−1)·1/(T−1) exposure per seat.
  - The ELO update uses pre-match ratings.
  - Unknown layout or outcome kind raises (P20, T5.2 ruling).
- **Checkpoints, resume and roles are consistent end to end.**
  - `meta.json` carries `roles` and `role_signature` (coordinator l.129-130; distributed learner l.290).
  - Resume checks the signature (`resolve_resume` → `check_role_signature`) and the architecture
    (`check_model_state`) before any process starts.
  - Eval checks the signature for every listed role (T6.1 fix) and validates each distinct architecture once.
- **Lifecycle (SP1 rulings intact).**
  - Exit codes are 0/1/130/143.
  - Children block SIGINT from `start_process` until `init_child_process` ignores it, and they get `PDEATHSIG`.
  - Shutdown: 7 s grace, then terminate and kill (≤ 2 s), then bounded reads.
  - FIX-2 (`b97dfec`) is sound:
    - the handlers are installed before the watcher starts;
    - a lost Ctrl-C is recorded via `sys.unraisablehook` and taken over as SIGINT;
    - the CPython string-exec flag is cleared before `sys.exit(130)`.
- **Distributed mode** runs on chunk v2. It refuses asymmetric agents with a one-line ConfigError naming SP5
  before any run dir exists (`distributed_setup`), and it imports gRPC only after the scope check (T6.4 ruling).
- **Config.**
  - `extra="forbid"` everywhere.
  - `matchmaking`, `checkpoint`, `roles`, `ratio_mode` / `unit_trace` / `entropy_reduction` and
    `critic_encoder_class` follow spec block 9.
  - `self_play`, `training.phase` and `env.num_players` are gone.
  - The messages name the key and give a fix hint.
- **Acceptance report** (`docs/superpowers/reports/2026-10-09-sp2-acceptance.md`) is careful. Every claim I
  spot-checked matched the repo (see T8.7 below).

## Issues

### Critical

None.

### Important

**I-1. Eval cannot schedule pairs whose role sets only partly overlap: a misleading ConfigError, or matches that
are never attributed.** (`src/colosseum/eval.py:84-102`, `:136-138`, `:659-661`, `:684`)

What happens in the code:
- `_pair_lineup` gives team *i* to core `(a, b)[(i+m) % 2]`. A seat the core cannot play goes to the other agent.
- It returns `None` when the lineup does not contain both agents.
- `schedule_lineups` then drops the whole pair unless **every** `m` works (`if all(lineup is not None …)`).
- `default_layouts` probes with `num_matches=1`, so it only checks `m = 0`.
- The CLI always rounds a pairwise count up to an even number.

Example: a game with roles `att` and `def` that share spaces (ENV_GUIDE §2 recommends exactly this for roles that
differ only in meaning). Agent `gen` plays both roles; agent `defn` has `roles: [def]`. Training supports this
config.

Failure scenarios (reproduced with `eval_sched.py`):
- Layout `1v1` = (att team 0, def team 1).
  - `m = 0` is valid (gen att, defn def).
  - `m = 1` puts gen on every seat, so it returns None and the pair is dropped.
  - `default_layouts` still returns `['1v1']`.
  - `colosseum eval -a gen=… -a defn=… -n 100` fails with
    `Config error: layout '1v1': the agents ['defn', 'gen'] cannot fill its seats`, for a layout the user never
    named. The asymmetric match itself is perfectly playable.
- Layout `2v2` = (att, def | att, def).
  - The lineups are `[gen, gen, gen, defn]` and `[gen, defn, gen, gen]`, so the teams are mixed. Spec block 8
    requires "команды заполняются однородно".
  - `_summarize_wdl` counts every one of these matches as `unattributed`. The report has `n > 0` and no W/D/L.

Why it matters: spec block 8 says the rotation applies "там, где роли это позволяют", and that teams are filled
homogeneously. The current rule only handles the two extremes: fully symmetric pairs and disjoint pairs like
hunter/prey.

Fix:
- Build each orientation homogeneously: every seat of team *i* goes to that team's core, or else to the other agent
  only when the core cannot play the role.
- Accept an orientation only if both agents appear in it.
- When orientation `m % 2` is invalid, use the other one (as hunter/prey effectively does). Drop the pair only if
  both orientations are invalid.
- Make `default_layouts` use the same predicate (e.g. probe with `num_matches=2`).
- Add a test for the `gen`/`defn` pair in `tests/unit/test_sp2_eval_schedule.py`.

### Minor

- **M-1 (duplicate tests that are not gRPC-safe).** `tests/unit/test_sp2_launcher_lifecycle.py:480-569` duplicates
  three tests of `tests/unit/test_sp2_distributed_roles.py:102-165`:
  - `test_run_workers_returns_an_exit_code`;
  - `test_run_learner_returns_128_plus_signum`;
  - `test_run_learner_setup_failure_leaves_signal_handlers_untouched`.

  The launcher copy's `fake_learner_role` (l.517) has no `pytest.importorskip("grpc")`. It imports
  `colosseum.transport.grpc_transport` (top-level `import grpc`; gRPC is an optional extra in `pyproject.toml`), so
  on an install without gRPC these tests error. That contradicts the T6.4 ruling ("CI suite must skip cleanly").
  Fix: delete the launcher copies.
- **M-2 (legacy leftovers).** None of these has a caller in `src/`:
  - `coordinator/agent_pool.py`: `register_frozen` / `register_scripted`, `AgentHandle.elo` / `win_rates` /
    `match_counts` / `network_config`, `remove`, `list_all` (comment "populated later in Milestone 4");
  - `EloRating.update_pairs` ("SP1 form", used only by one test);
  - unused `logger` in `matchmaker.py` (T5.1 minor);
  - six blank lines at `launcher.py:124-129` (T5.4 minor);
  - the `_parse_meta` docstring (`checkpoint_manager.py:87-92`) does not mention the validated
    `roles` / `role_signature` (T5.3 minor).

  The brief's principle is "delete all legacy". Either trim them, or keep `AgentPool` deliberately for SP3 and say
  so in its docstring.
- **M-3 (stale wording).**
  - `core/config.py:454-464`, `transport/local.py:3,23` and `weight_store/shared_memory.py:8,54` still say
    "Unused in SP1; kept for SP5". It should read "unused until SP5".
  - README "Ratings" says the cross-play table exists "with `teammates: mixed`". `RatingBook.update` records it for
    every one-team layout with two or more seats, homogeneous teams included (T5.2 minor).
  - README `-m slow` says "12–16 min", but the T8.7 run took 11.6 min.
- **M-4 (distributed scope).** `distributed_setup` (`distributed.py:152-178`) refuses asymmetric agents. With
  several agents it silently ignores `matchmaking.mode: league`, `teammates: mixed` and `shuffle_seats`: every env
  is one agent's latest on all seats. README says "no league", so this is documented and matches SP1. Spec block 10
  says "Остальное — ConfigError". A one-line warning or ConfigError for `mode: league` would remove the surprise.
- **M-5 (`scripts/units_experiment.py`).** Its defaults (`--steps 400_000`, `--parallel 2`) do not reproduce the
  recorded experiment (340k steps, `--parallel 1`, see the T8.4 rulings). Add the exact command to
  `docs/benchmarks.md`, or change the defaults.
- **M-6 (acceptance report bookkeeping).** The ledger in the report stops at entry 37. Ledger line 209 ("T8.7's
  task review is folded into the final whole-branch review") and the outcome of this final review / fix wave are not
  in it yet. Add them when the report is finalised after the fix wave.

## Deferred minors to fix now (ledger references)

1. **T8.6 ruling (dropped-reward counter), recommended.**
   - Worker stats already reach the hub every 2 s, and `dropped_reward_episodes` is in them
     (`rollout_loop.py:220`, `rollout_worker.py:152-154`).
   - `SystemStats.snapshot` (`metrics/aggregator.py:170`) sums only `parked_buffers`.
   - Fix: add the same sum for `dropped_reward_episodes` to the `system` record (and so to WandB `system/*`).
   - Update the README `system` row and `ENV_GUIDE.md:149`.
   - This is about three lines of code and gives users a visible signal, which is the point of the ruling.
2. **FIX-2b (autouse fixture), recommended, cheap.** An autouse fixture in `tests/unit/test_process_lifecycle.py`
   that calls `take_lost_interrupt()` after each test. If the first assertion of
   `test_interrupt_lost_in_an_unraisable_context_reaches_the_supervisor` fails, the module-global record leaks into
   later in-process tests as a phantom SIGINT.
3. **T5.4, T5.1, T5.3 trivia** (blank lines, unused logger, `_parse_meta` docstring), together with M-2/M-3 if the
   wave touches those files anyway.
4. **Stay parked:**
   - FIX-2b `serve-weight-store` 0-vs-130 (documented in README) and the `PYTHONPATH` replacement in two tests;
   - T0.1 queue-release guards and private-attribute test;
   - T5.2 `games` semantics and untyped `_classify`;
   - T6.1 eval minors (architecture check on zero obs, n! cross-play orders, …);
   - T6.2 context order (already in CLAUDE.md «Parked during SP2»);
   - T6.4 `distributed_setup` / `setup_run` duplication.

## Recommendations

- Fix I-1 in the final wave, with the regression test described there. It is local to `eval.py` (schedule plus
  `default_layouts`).
- Delete the duplicated tests (M-1). It is the only test-hygiene item that can break the CI suite on a supported
  install (without the gRPC extra).
- After the fix wave, refresh the counts in README / CLAUDE.md and the report's §3.1 if the test count changes.
  Append ledger line 209 and the final-review outcome to the report (M-6).

## T8.7 (acceptance report) check

- **Numbers.**
  - Collected counts at HEAD are 1243/1270 fast (27 deselected), 9 slow and 18 gpu, as reported.
  - The per-step counts reproduce on collection: `tests/contract` 157, `-k vtrace` 33/1006, the 8-file contract
    supplement 174, distributed/grpc 7/94.
  - Throughput 4848 / 9298 / 13702 against SP1's 4641 / 9444 / 10664 matches `docs/benchmarks.md:87-89,145-149`.
  - `docs/benchmarks/units-experiment.json` has 12 rows with `recommendation = joint`.
- **Node ids.** Every `tests/…::name` in the report exists in the collection. The only path that is not a test node
  is the historical `tests/integration/test_bc_cli.py`, cited as "перенесён в T7.3", which is correct.
- **Commits.**
  - `7e28f2c` is the first code commit after the five spec/plan commits.
  - `f5e8a4b` is the last change to `after-sp2.json`.
  - `git diff --stat f5e8a4b..7d55974 -- src/` shows four files: `cli.py` and `utils/process.py` (FIX-2), plus
    `core/errors.py` and `core/specs.py` (docstrings only).
  - The QueueReader lines are `ipc.py:237/356/420` and `launcher.py:34`.
- **Rulings ledger.** It is complete against `progress.md`. All 37 `Ruling` lines up to T8.6 are present verbatim
  and in order, plus PR-1..PR-5, P1–P22 and the T8.3–T8.4 draft rulings. Line 209 was written after the report
  (M-6).
- **Open items.** All six match CLAUDE.md «Parked during SP2» one to one. The final-review candidates
  (dropped-reward signal, FIX-2b) are listed separately, which is correct.
- **Docs of the real state.**
  - README, CLAUDE.md (Implementation Status, Eval Mode, Partial, Roadmap) and ENV_GUIDE describe the SP2 code;
    I found nothing stale apart from M-3.
  - CLAUDE.md's SP4 item about unifying eval with `RolloutLoop` is correctly gone (MatchRunner now serves both).
- **Verdict:** the task meets its spec (Steps 1–13 of the brief and its carry file; Step 14 is correctly left to
  the owner), and the report is accurate. It only needs the bookkeeping update in M-6.

## Assessment

**With fixes.** No Critical issues. One Important (I-1, eval scheduling of partly overlapping role sets), local to
`eval.py`. Training-side matchmaking, ratings, checkpoints, resume, lifecycle and distributed scope are correct
as far as I could verify.

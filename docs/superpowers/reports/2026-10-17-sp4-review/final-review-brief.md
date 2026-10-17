<!-- Origin: .superpowers/sdd/2026-10-16-sp4-selection/final-review-brief.md in the git-ignored SDD workspace (deleted after the SP4 merge); copied verbatim on 2026-10-17. -->
<!-- Frozen record: never edit. -->

# SP4 final whole-branch review — shared brief

You are one of three Senior Code Reviewers of the SP4 branch `sp4-selection` of Colosseum, a distributed RL training framework for competitive bot programming (repo `/home/viv/dev/repos/Colosseum`).

SP4 = "selection and observability". It adds:
- **Match log.** A SQLite journal `runs/<name>/matches.db`: WAL, counts by team pairs, compaction of training matches beyond a detail window, a disk guard, resume import, and a writer thread in training.
- **Ratings.** Bradley–Terry snapshot ratings (`arena.bt`): side bias per team index, role groups and components with their own anchors, Laplace / perturb-and-solve uncertainty, a paired stratified bootstrap, the G² transitivity test, and score layouts. Live ratings run in a refit thread (`arena.live`) and feed `ratings.json["bt"]`.
- **League state and retention.** `league_state.json` for run-dir resume. Snapshot retention (`arena.retention`): top-k by LCB, pinned snapshots, `max_snapshots`, current-run final; `colosseum pin`.
- **Tournament.** `colosseum tournament` (`arena.tournament`):
  - players: run snapshots, `top:N`, `-a`, config agents;
  - schedules: balanced blocks, admissible rotations, FFA compositions, score layouts, keyed seed domains;
  - a worker engine;
  - statistics and modes: gauntlet / round-robin / halving / select, with `confirmed` from the 95 % CI;
  - a tournament dir with resume by whole blocks, snapshot links, journal writes and `--pin`.
- **Pulse.** An optional child process (`arena.pulse`) that evaluates new snapshots against a fixed field during training and writes to the journal only.
- **Report and dashboard.** Report data builders, `run_state.json`, a self-contained HTML report (`colosseum report`) and a stdlib live dashboard (`colosseum dashboard`).
- **Metric changes.** `rho_trunc_mass` / `c_trunc_mass` replace the clip fractions, plus monotonic loss counters.
- **Distributed guards.** The bench measures by counters.
- **Result batching (controller task FX1).** Training workers send match results to the main process in batches (`ResultBatcher`: up to 256 results or 0.1 s; `_RESULTS_QUEUE_SIZE` = 1000 batches; a full queue drops the whole batch, counted per result). Before FX1 the per-result queue lost 73–79 % of the results on the default tic-tac-toe config, on `main` too. Diagnostic: `.superpowers/sdd/2026-10-16-sp4-selection/drops-diag/report.md`; ruling in the ledger. The learner's metrics queue has the same drain pattern and was parked (pre-SP4).
- **Measurements and slow tests.** Ladder bots, measurement scripts with pre-fixed decision rules, measured defaults, and new slow tests (ladder on two games, pipeline with pulse + select, disk bound).
- **Docs.** `docs/ARENA_GUIDE.md` and updates to LEAGUE_GUIDE / README / CLAUDE.md.

The owner required backward compatibility with SP3 configs, run dirs and checkpoints.

## Authority and inputs
- **Spec** (binding authority, Russian): `docs/superpowers/specs/2026-10-16-sp4-selection-design.md`.
- **Plan overview:** `docs/superpowers/plans/2026-10-16-sp4-selection/00-overview.md`. It has the interface contract and the amendments A1–A33. Parts 01–03 sit next to it; read only what you need.
- **Constraints, pre-flight and controller rulings:** `.superpowers/sdd/2026-10-16-sp4-selection/constraints.md` (P1–P22, C1–C14; several amendments carry "(T… review:)" additions).
- **Controller ledger:** `.superpowers/sdd/2026-10-16-sp4-selection/progress.md`. It records every task completion, every `Ruling:`, every `minor (deferred)` and every `⚠️ carried forward`.
  - Rulings are decisions already taken. Do not re-raise them as findings unless the code contradicts the ruling or the ruling causes a real defect; then name the ruling.
  - Carried-forward items name the task that had to close them. Check that it did.
- **SP1–SP3 rulings** stay binding unless SP4's spec §8 changes them. They are at the ends of `docs/superpowers/reports/2026-10-08-sp1-acceptance.md` and `2026-10-09-sp2-acceptance.md`, and in «Решения контроллера во время SP3» of `2026-10-10-sp3-acceptance.md`.
- **Docs of the real state:** `README.md`, `CLAUDE.md`, `docs/ARENA_GUIDE.md`, `docs/LEAGUE_GUIDE.md`, `docs/ENV_GUIDE.md`, `docs/benchmarks.md`, `docs/GPU_CHECKS.md`.
- **Git range:** merge base with main `1366bec` .. `HEAD` of `sp4-selection`. Use `git diff 1366bec..HEAD -- <paths>`. Much of the diff is docs, plans, research and fixtures; focus on your area's source and tests.

## Machine
- **ROS:** prefix every `.venv/bin/python` command with `env -u PYTHONPATH`, because the machine's ROS PYTHONPATH breaks pytest.
- **Clock:** the wall clock jumps by about +9 s every few seconds. Durations must use `time.monotonic()`; ordering must use counters.

## Read-only
- Do not mutate the working tree, index, HEAD or branches. Write no files anywhere except your output file below.
- For experiments you may run focused scripts or single tests (`OMP_NUM_THREADS=1`, ≤ 2 worker processes).
- Put any scratch script and its temp dirs under `.superpowers/sdd/2026-10-16-sp4-selection/final-review/<AREA>/` (git-ignored). Run it with that directory as the cwd, NEVER the repo root: a test guard fails the suite on any new top-level entry in the repo root.
- Do not run the full suite, the slow tests, training runs longer than a few seconds, or benchmarks; acceptance does that.
- Do not dispatch subagents.

## What to look for (priority order)
1. **Correctness bugs that silently corrupt results.** Examples:
   - journal rows attributed to the wrong entity, version or seat;
   - lost or double-counted matches (compaction, import, resume, shutdown);
   - BT math errors: bias, anchors, groups or components, uncertainty that is too narrow or too wide;
   - the bootstrap not paired or not stratified as the spec says;
   - scores attributed by position;
   - retention evicting what it must keep;
   - a tournament reusing stale blocks, or seeds colliding across domains;
   - pulse results reaching PFSP / ELO;
   - a report showing a non-confirmed winner as confirmed;
   - SP3 configs, run dirs or checkpoints breaking.
2. **Contract violations** against the spec, overview or amendments; spec requirements implemented differently without a ruling.
3. **Robustness.**
   - Hangs, leaked processes or threads, swallowed errors, silent fallbacks the spec does not name.
   - Unbounded disk, memory or page growth ("disk is bounded by construction").
   - Errors without context.
   - SQLite rules: one connection per thread, short transactions, read-only readers.
   - Security of the dashboard and the page: XSS through embedded data or links, path traversal, binding.
4. **Test gaps** where a bug could hide: tests that cannot fail, statistics tests outside the "< 1e-3" rule, missing negative cases on critical paths.
5. **Leftovers:** dead code, stale names (`rho_clip_frac` / `c_clip_frac` outside the historical tables, ruling C10), duplication worth removing.
6. **Deferred minors** in the ledger for your area: say which, if any, should be fixed in the single final fix wave and why. The rest stay parked.

## Calibration
Categorize by real severity:
- **Critical:** wrong results, a crash or hang on a supported path, data loss.
- **Important:** a real defect on a supported but less common path, a missing spec requirement, a misleading doc of the real state.
- **Minor:** polish.

For every finding give file:line, what is wrong, why it matters and how to fix it. For every Critical or Important finding, also give a concrete failure scenario (inputs → wrong output).

Verify each Critical / Important by reading the code path end to end, or by a focused experiment, before reporting it. Drop what you cannot substantiate.

## Output
Write the full review to `.superpowers/sdd/2026-10-16-sp4-selection/final-review-<AREA>.md`. This is the only file you write. Sections:
- Strengths
- Issues: Critical / Important / Minor, each with file:line, failure scenario and fix
- Deferred minors to fix now (ledger references)
- Recommendations
- Assessment: Ready to merge? Yes | No | With fixes

Return only: counts per severity, one line per Critical / Important finding, and the assessment.

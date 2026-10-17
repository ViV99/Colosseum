<!-- Origin: .superpowers/sdd/2026-10-16-sp4-selection/fix-wave-brief.md in the git-ignored SDD workspace (deleted after the SP4 merge); copied verbatim on 2026-10-17. -->
<!-- Frozen record: never edit. -->

# SP4 final fix wave: shared brief

You implement part of the single fix wave that follows the SP4 final whole-branch review. You work on the Colosseum repo, branch `sp4-selection`.

Read and follow `/home/viv/dev/repos/Colosseum/.superpowers/sdd/2026-10-16-sp4-selection/implementer-instructions.md`. The overrides below take precedence over it.

## Inputs
- **Review files.** `.superpowers/sdd/2026-10-16-sp4-selection/final-review-A.md`, `final-review-B.md` and `final-review-C.md`. Each finding there carries file:line, a failure scenario and a suggested fix. Your dispatch names which finding ids are yours.
- **Spec.** `docs/superpowers/specs/2026-10-16-sp4-selection-design.md`. It is binding.
- **Contract and rulings.** The contract is `docs/superpowers/plans/2026-10-16-sp4-selection/00-overview.md`. The amendments and rulings are in `.superpowers/sdd/2026-10-16-sp4-selection/constraints.md` and in the ledger `progress.md`, which is read-only for you.
- **Controller rulings for this wave.** They are below. They are already decided; implement them as written.

## Controller rulings for this wave
- **A-M3 (startup race).** A training worker, local or distributed, never plays a trainable agent's `latest` seats before it has received that agent's first weights from the learner.
  - At startup the worker waits for the first weight payload of every trainable agent it serves.
  - The wait is in short slices and stops on `stop_event`.
  - It logs one INFO line if it waits longer than a few seconds.
  - Today the worker plays its own factory weights, which are random, as version 0 until the first sync (up to `rollout.weight_sync_interval_sec`). With `init.from` or a resume, those games are journaled as `v0` / the resumed version and mix random-weight games into that bucket's rating.
  - In-process loops and unit tests that never push weights must keep working. Use an explicit option, default off in `RolloutLoop`, on in the worker processes; or an equivalent design. Say which one you chose.
  - Needs a test showing that no match is played or journaled before the first weights arrive.
- **A-M7 (SP3 compatibility of the snapshot cap).** When `checkpoint.max_snapshots` is not set in the config and `keep_last` is greater than 100, the cap becomes `keep_last`, with one INFO line.
  - The case also covers an SP2 `pool_size` translated to `keep_last`.
  - An explicit `max_snapshots` below `keep_last` stays a `ConfigError`.
  - Needs a test with an SP3-form config that has `keep_last: 150`.
- **B-I1 (one-team games).** An empty field is allowed in two cases.
  - In `colosseum tournament`, when every chosen layout has one team. The modes are gauntlet, halving and select. Round-robin already refuses one-team layouts (C4).
  - In the pulse, when no planned layout has two or more teams.
  - The error text for a real two-team empty field stays as it is.
  - Needs one test per path, plus one ARENA_GUIDE sentence.
- **B-M8 (round-robin role groups).** Mixed role groups in one round-robin table are a `ConfigError` that lists the groups. This follows spec block 3 («Смешанные группы в одной таблице — ConfigError с перечнем групп»).
  - A participant with no playable pair gets `score: null`, not 0.0.

- **C-I1 (live dashboard filters).** In live mode the page keeps its own layout choice. It does not send `layout` in its refresh query and then rebuild the select from the answer. The matrix section gets an agent select that drives `api/matrix?agent=`, and drives the score cross-play too (T7.1 deferred).
  - Needs a live-page DOM test: headless Chrome, as in `_dom`. It must show that a second layout and another agent stay reachable after a refresh.
  - The ARENA_GUIDE line about the agent filter must describe the page as it is after the fix.
- **C-M4 (static report).** `colosseum report` isolates sections, as the dashboard does. A failing section gets a `{"notes": ["This section failed: …"]}` placeholder, the page is still written, and the command exits 1 with a message naming the failed sections. The page is useful, and the failure must not look like success.
- **C-M2 (report memory).** `MetricsTail` keeps bounded memory over a run of any length.
  - Thin each (kind, agent) series by its counter on ingest, using stride doubling: the order and counter keys stay as they are, and the page's own size limits stay.
  - Read the file in fixed-size chunks.
  - The docstring states the bound.
- **A-M4 (compact `best`).** The `ratings` record's `best` is the best stored snapshot, as spec block 2 says («лучшего снимка»): versions > 0 that are stored. If the live thread cannot know the stored set cheaply, restrict `best` to versions > 0 and call it "best version" in the docs. Say which.
- **B-M2 (dead tournament worker).** The engine checks `first_failure` once per loop iteration, not only on an empty queue. On a failure it sets `stop` first, and it keeps journaling the `done` blocks that arrive while it looks for the typed error, bounded by `STOP_GRACE_SEC`.
- **B-M6 (snapshot link dir).** In a new tournament, where nothing references `snapshots/` yet, the dir is emptied before linking. A reused link dir is never trusted by its path alone.

## Overrides
- **Where you work.** Your dispatch gives your worktree and branch. Commit there and do not push; the controller integrates. Run Python as `env -u PYTHONPATH PYTHONPATH=<worktree>/src OMP_NUM_THREADS=1 /home/viv/dev/repos/Colosseum/.venv/bin/python`, and check `colosseum.__file__` once.
- **TDD.** Write the failing test first wherever behaviour changes, and record RED / GREEN. Text-only fixes need no test.
- **Full fast suite.** Run it ONCE at the end in your worktree, with zero failures and zero warnings, plus ruff. Another wave implementer may be running its suite at the same time, so a load-dependent integration failure may be a flake. Re-run that single test alone; if it passes, say so in your report.
- **Docs.** When a fix changes behaviour that the docs describe (ARENA_GUIDE, LEAGUE_GUIDE, README, CLAUDE.md, benchmarks.md), update the docs in the same commit. Use Russian in the guides and benchmarks, and English in README / CLAUDE.md.
- **Commits.** Use one commit per finding, or small logical groups, each with a conventional prefix and no attribution lines. In each commit body, name the finding ids it fixes.

## Report
Write `.superpowers/sdd/2026-10-16-sp4-selection/fix-wave-<PART>-report.md`. If the write is refused, put the report in your final message. It contains:
- for each finding id: fixed / not fixed (why), the commit, and the test;
- RED / GREEN evidence;
- the suite result;
- deviations and proposed rulings, in the form «что — почему — цена ошибки».

Your final message: the status (`DONE | DONE_WITH_CONCERNS | BLOCKED`), the commits, and one line per finding id.

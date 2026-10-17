# SP4: final review, fix wave and FX1 records (frozen)

These are verbatim copies of SP4 records that lived only in the git-ignored SDD workspace `.superpowers/sdd/2026-10-16-sp4-selection/`, which is deleted after the SP4 merge. Each copy has a 2-line origin header and is otherwise unchanged. The records are frozen: never edit them. The summary, the acceptance measurements and every ruling are in the SP4 acceptance report in `docs/superpowers/reports/` (`*-sp4-acceptance.md`).

As in SP3, the final whole-branch review and its fix wave ran **before** the acceptance, so every acceptance measurement ran on the final code. One controller task, FX1, ran between the measurements of T8.2 and the slow tests of T8.3. Its finding came out of T8.2's throughput measurement: match results were dropped on fast games, on `main` too.

| File | What it is |
|---|---|
| [`fx1-drops-diagnostic.md`](fx1-drops-diagnostic.md) | The FX1 diagnostic. It measured `main` and the branch the same way. On the default tic-tac-toe config with 2 workers, `main` dropped 79 % of the match results and the branch 73 %; on the bench at 4 workers, 27–31 % and 7 %. The root cause is a once-per-pass drain of a 64 KiB pipe. Batching the results on the workers measured 0 drops at 99–102 % of `main`'s speed. |
| [`task-FX1-report.md`](task-FX1-report.md) | The FX1 fix (`cf49651..d479f90`). Workers send results in batches of up to 256 results or 0.1 s; the queue holds 1 000 batches; drops are still counted per result. The check measurement showed 0 lost on both workloads, at 101.8 % / 99.9 % of `main`. Includes its fix round 1. |
| [`final-review-brief.md`](final-review-brief.md) | The shared brief of the three area reviewers. |
| [`final-review-A.md`](final-review-A.md) | Area A: journal, ratings, live ratings, league state, retention, the result flow (with FX1), metrics. 0 Critical, 0 Important, 11 Minor. Among them: M3, workers played their factory weights as v0 until the first sync, and M7, an SP3 config with `keep_last` > 100 failed validation. |
| [`final-review-B.md`](final-review-B.md) | Area B: tournament, pulse, eval / players, distributed guards. 0 Critical, 1 Important (I1: on all-one-team games, i.e. solo or cooperative, tournament and pulse refused an empty field), 10 Minor. |
| [`final-review-C.md`](final-review-C.md) | Area C: report, dashboard, run state, measurements, slow tests (also T8.3's task review), docs. 0 Critical, 1 Important (I1: the live dashboard locked Ratings and Results matrix to one layout after the first refresh and had no agent control), 10 Minor. |
| [`fix-wave-brief.md`](fix-wave-brief.md) | The controller's brief for the single fix wave, with the controller rulings A-M3, A-M7, A-M4, B-I1, B-M8, B-M2, B-M6, C-I1, C-M4 and C-M2. The wave ran as three parallel parts in worktrees. |
| [`fix-wave-A-report.md`](fix-wave-A-report.md) | Part FW-A: 11 commits. Covering tests and RED / GREEN evidence; fast suite 2143 passed, 0 warnings. |
| [`fix-wave-B-report.md`](fix-wave-B-report.md) | Part FW-B: 10 commits; fast suite 2134 passed, 0 warnings. |
| [`fix-wave-C-report.md`](fix-wave-C-report.md) | Part FW-C: 11 commits; fast suite 2129 passed, 0 warnings; the disk slow test passed 2/2 alone. |
| [`fix-wave-2-report.md`](fix-wave-2-report.md) | FW-2: a follow-up with the five Minor findings of the scoped re-reviews (`7faf179..0abc4bd`); fast suite 2168 passed, 0 warnings. |

Integration: the controller cherry-picked the three parts onto `167d9cc`, which gave `7faf179`. There was one conflict, in CLAUDE.md «Parked during SP4», and the controller resolved it. The fast suite on the integrated tree: 2164 passed, 0 warnings.

The scoped re-reviews of the wave have no files of their own. One re-reviewer per part gave these verdicts; they are copied as ledger lines into the acceptance report:
- **FW-A:** all findings addressed, no new breakage. The M3 wait was probed end to end: a learner that dies before its first push, late weights, and SIGINT during the wait.
- **FW-B:** all findings addressed, no new Critical / Important. FW-2 fixed the two Minor findings.
- **FW-C:** all findings addressed, no new Critical / Important. FW-2 fixed the three Minor findings.

FW-2's diff was checked by the controller.

Some workspace files that these records mention were not kept: task briefs and reports, `review-*.diff` packages, `progress.md`, and the reviewers' scratch scripts under `final-review/`. They are reproducible from git: the diffs are the named commit ranges, and every `Ruling` and deferred-minor line of `progress.md` is copied verbatim into the acceptance report.

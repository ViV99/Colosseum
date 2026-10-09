# SP2: final review and fix records (frozen)

Verbatim copies of the SP2 records that lived only in the git-ignored SDD workspace `.superpowers/sdd/2026-10-08-sp2-game-model/`, which was deleted after the SP2 merge. Each copy has a 2-line origin header; absolute scratch paths are replaced with `<scratch>/`, nothing else changed. Frozen: never edit them. The summary and the rulings are in [`../2026-10-09-sp2-acceptance.md`](../2026-10-09-sp2-acceptance.md).

| File | What it is |
|---|---|
| [`final-review-A.md`](final-review-A.md) | Final whole-branch review, area A: game contract, data trees, models (`7430361..7d55974`). 0 Critical, 1 Important (`UnitsHead` keys), 11 Minor (M-1..M-11). |
| [`final-review-B.md`](final-review-B.md) | Area B: training data path (worker buffers, `MatchRunner`, V-trace, APPO, learner, BC/kickstart, IPC, serialization). 0 Critical, 0 Important, 6 Minor. |
| [`final-review-C.md`](final-review-C.md) | Area C: league, orchestration, CLI, docs and the acceptance report. 0 Critical, 1 Important (eval schedule for partly overlapping roles), 6 Minor. |
| [`final-fix-report.md`](final-fix-report.md) | The single final fix wave (`a6a2faa..b4d6f93`, 10 items) with RED/GREEN evidence. |
| [`final-rereview.md`](final-rereview.md) | Scoped re-review of the fix wave: all items addressed; new Minor N-1 (parked to SP4). |
| [`task-FIX-1-report.md`](task-FIX-1-report.md) | FIX-1 (`a4f1ad1`): flaky resume integration test; root cause = first run overshoots its budget. |
| [`task-FIX-2-report.md`](task-FIX-2-report.md) | FIX-2 (`b97dfec`): flaky SIGINT-during-startup test; CPython 3.12 unhandled-KeyboardInterrupt quirk and two more startup Ctrl-C paths. |
| [`readiness-review.md`](readiness-review.md) | Fresh-session readiness review of the handoff docs at `70e65ab`: answers a new SP3 session would get from the repo alone, gaps G1–G14 with suggested text, 7 inaccuracies, 32 verified claims, the check that every forwarded ledger minor was closed. Its gaps were fixed in the next docs commit. |

Workspace files these records mention but that were not kept (briefs, `review-*.diff` packages, `progress.md`) are reproducible from git: the diffs are the named commit ranges, and every `Ruling` and deferred-minor line of `progress.md` is copied verbatim into the acceptance report.

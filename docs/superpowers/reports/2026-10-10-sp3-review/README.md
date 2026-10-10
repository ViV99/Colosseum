# SP3: final review and fix-wave records (frozen)

Verbatim copies of the SP3 records that lived only in the git-ignored SDD workspace `.superpowers/sdd/2026-10-10-sp3-league/`, which is deleted after the SP3 merge. Each copy has a 2-line origin header; nothing else changed. Frozen: never edit them. The summary, the acceptance measurements and every ruling are in [`../2026-10-10-sp3-acceptance.md`](../2026-10-10-sp3-acceptance.md).

In SP3 the final whole-branch review and its single fix wave ran **before** the acceptance (controller ruling in the ledger), so every acceptance measurement ran once on the final code.

| File | What it is |
|---|---|
| [`final-review-A.md`](final-review-A.md) | Final whole-branch review, area A: players, config and execution (`c16fe5e..c31eda9`). 0 Critical, 1 Important (I-1: `record` / `bc` / `eval` validate warm-start sources they never use, so the pipeline cannot run from one config), 10 Minor (M-1..M-10). |
| [`final-review-B.md`](final-review-B.md) | Area B: league, ratings, metrics, distributed guards. 0 Critical, 2 Important (I1: evicted snapshots come back into the PFSP table through late results; I2: LEAGUE_GUIDE understates the SP2 behaviour changes), 13 Minor (M1..M13). |
| [`final-review-C.md`](final-review-C.md) | Area C: warm start, training data path, record/BC, tests, docs, benchmarks. 0 Critical, 3 Important (I-1 = A's I-1; I-2: the critic warm-up froze the critic's own `global_state` normalizer; I-3 = B's I2), 6 Minor. |
| [`fix-wave-brief.md`](fix-wave-brief.md) | The controller's brief for the single fix wave: items 1–18 (4 Important + 14 cheap minors chosen from the reviews' "fix now" lists). |
| [`fix-wave-report.md`](fix-wave-report.md) | The fix wave (`c31eda9..6fa1406`, 18 commits): per item what changed, covering tests, RED/GREEN evidence; full fast suite 1689 passed, 0 warnings. |

The scoped re-review of the fix wave has no separate file; its verdict is the ledger line copied into the acceptance report (section «Финальное ревью ветки»): 18/18 items addressed, no new Critical/Important; 3 documentation Minors handed to the acceptance task (fixed there).

Workspace files these records mention but that were not kept (task briefs and reports, `review-*.diff` packages, `progress.md`, the reviewers' scratch scripts under `final-review/`) are reproducible from git: the diffs are the named commit ranges, and every `Ruling` and deferred-minor line of `progress.md` is copied verbatim into the acceptance report.

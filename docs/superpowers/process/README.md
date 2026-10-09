# Sub-project execution process (as run in SP2)

Templates and the loop the controller (main session) used to execute the SP2 plan. The SDD workspace (`.superpowers/sdd/<plan-name>/`) is git-ignored scratch, deleted after the merge; this directory keeps the reusable parts. Copy them into the next workspace and adapt the plan, spec and workspace names.

| File | Use |
|---|---|
| [`implementer-instructions.md`](implementer-instructions.md) | Binding process contract for every task implementer (TDD, full suite once, commit + push, no subagents, report and status contract). |
| [`reviewer-instructions.md`](reviewer-instructions.md) | Per-task reviewer: spec compliance, then code quality, read-only, from a review package. |
| [`rereviewer-instructions.md`](rereviewer-instructions.md) | Scoped re-review of one fix round: a verdict on every finding plus new breakage in the fix diff. |
| [`final-review-brief.md`](final-review-brief.md) | Shared brief of the three area reviewers of the whole-branch review. |
| [`brief.sh`](brief.sh) | Extracts one task's brief (part preamble, task text, part contract notes) from the plan parts into the workspace. `WS=... PLAN=... brief.sh T1.1`. |

## The loop

1. **Before the first task.**
   - Write `constraints.md` into the workspace: global constraints, binding contract amendments and plan-stage rulings, extracted from the plan overview.
   - Run a pre-flight consistency scan of the plan: each task, each pair of tasks, and every plan-mandated defect. Rule on every finding (P-rulings, appended to `constraints.md`).
   - Start the ledger `progress.md`.
2. **Per task, in the plan's order.**
   1. `brief.sh <task>` writes the brief file.
   2. Dispatch one implementer (fresh subagent) with `implementer-instructions.md`, the brief and the applicable amendments. Its report goes to `task-<id>-report.md`; it replies with a short status: `DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT`.
   3. Build a review package with the superpowers `review-package` script (`skills/subagent-driven-development/scripts/review-package PLAN_FILE BASE HEAD [OUTFILE]`; BASE = the task's recorded base, not `HEAD~1`).
   4. Dispatch one reviewer with `reviewer-instructions.md`, the brief, the report and the package.
   5. On findings: fix rounds. Resume the same implementer with the findings, at most 5 rounds per task. After each round, a scoped re-review (`rereviewer-instructions.md`) on the `FIX_BASE..HEAD` package.
   6. The controller decides what is fixed now and what is deferred (Minor), and writes the ledger lines.
3. **Final whole-branch review.** Three area reviewers run in parallel and read-only on `merge-base..HEAD` (`final-review-brief.md`; areas in SP2: A game contract/data trees/models, B training data path, C league/orchestration/CLI/docs/acceptance report). Then ONE fix wave (one implementer, one brief listing the items) and one scoped re-review of it. Anything else stays parked.
4. **Acceptance** against the spec's §3 criteria, as a report in `docs/superpowers/reports/`. The owner must accept it explicitly.
5. **Handoff before the merge.** Move every durable item (rulings, deferred minors, review records) from the workspace into `docs/`. Then run a fresh-session readiness review: can a new session start the next sub-project from the repo docs alone? Fix its gaps.
6. **Merge** with `git merge --no-ff` after the owner's explicit "yes", then delete the workspace.

## Ledger line formats (`progress.md`)

- `T<id> BASE=<sha>` — the task starts from this commit; also `FIX_BASE=<sha>` for a fix round.
- `Task <id>: fix round N/5 (<k> addressed, <m> open — <what>; commits a..b)`
- `Task <id>: minor (deferred): <items>` — also `minor (deferred→T<n>)`, when a later task is to close it.
- `Task <id>: ⚠️ resolved by controller: <what>` and `Task <id>: ⚠️ carried forward: <what the next tasks must do>`
- `Task <id>: complete (commits a..b, review clean[ after fix round n])`
- `Ruling: <what> — <why> — cost if wrong: <cost>` — every decision taken on the owner's behalf; all of them go verbatim into the acceptance report.
- Phase lines (`Phase T3–T4 complete at <sha>.`), concerns, and resume notes before a context compaction.

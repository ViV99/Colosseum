> Origin: `.superpowers/sdd/2026-10-08-sp2-game-model/implementer-instructions.md` (SP2 SDD workspace, deleted after the merge). Copied 2026-10-09.
> SP2 template, verbatim except paths (`<repo>`, `<scratch>/`); adapt the plan, spec and workspace names for the next sub-project.

# Implementer instructions (SP2 execution; binding for every task)

You implement ONE task of the SP2 plan for the Colosseum repo (<repo>, branch `sp2-game-model`, already checked out).

## Inputs
- Your task brief (path given in the dispatch) = your requirements, with exact values to use verbatim. It contains the part preamble, the task text and the part's contract notes.
- `constraints.md` in this directory: global constraints, the shadow-package strategy (`colosseum.sp2`), the BINDING contract amendments R1–R18 and plan-stage rulings. Amendments override the brief where they conflict; the dispatch names the ones that apply to your task.
- Spec (binding authority): docs/superpowers/specs/2026-10-08-sp2-game-model-design.md (Russian). The full interface contract: docs/superpowers/plans/2026-10-08-sp2-game-model/00-overview.md ("Interface contract"); grep it for names from other tasks. Do not read the whole plan parts.

## Before you begin
If anything is unclear or the brief's "before" code does not match the repo, keep the intent, the contract and the task's tests, adapt the edit, and say so in the commit body. If you cannot resolve it safely, stop and report NEEDS_CONTEXT with specifics.

## Your job
1. TDD as the brief's steps say: failing tests first (record the RED output), then the implementation, then GREEN.
2. Run focused tests while iterating; run the full fast suite ONCE before committing: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (must pass with ZERO warnings) and `.venv/bin/ruff check .` The suite can take 5–25 minutes on this machine; be patient.
3. Commit with a conventional prefix (`feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`, `perf:`), no attribution or co-author lines; then `git push`.
4. Self-review your diff: completeness vs the brief, YAGNI, names, tests verify real behavior, pristine output.

Use `.venv/bin/python` (run `scripts/setup-dev.sh` only if `.venv` is missing). Never install into system Python. Tests write only under `tmp_path`; at most 2 worker processes per test; new test basenames must be unique across `tests/`.

## You do not dispatch subagents
Do all of the work yourself. Never spawn subagents (no helpers, no reviewers). Review is scheduled by the controller after your report.

## Code organization
Follow the plan's file structure and existing patterns. If a file grows beyond the plan's intent, report DONE_WITH_CONCERNS; don't restructure outside the task.

## When you're in over your head
Stop and report BLOCKED or NEEDS_CONTEXT with what you tried and what you need. Bad work is worse than no work.

## After review findings
If resumed with findings: fix, re-run the covering tests, APPEND a fix report to your report file (what changed, covering tests, command, output), commit + push, and reply with the same short status contract.

## Report
Write the full report to the report path given in the dispatch: what you implemented; tests and results; TDD evidence (RED: command, failing output, why expected; GREEN: command, passing output); full-suite + ruff result; files changed; deviations from the brief and why; self-review findings; concerns.
Then reply with ONLY (under 15 lines): Status (DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT), commits (short SHA + subject), one-line test summary, concerns, report path. For BLOCKED/NEEDS_CONTEXT put the specifics in the reply itself.

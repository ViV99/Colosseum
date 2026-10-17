> Origin: `.superpowers/sdd/2026-10-10-sp3-league/implementer-instructions.md` (SP3 SDD workspace, deleted after the merge). Copied 2026-10-15.
> The SP2 template as revised for SP3, verbatim except paths (`<repo>`); adapt the plan, spec, branch and workspace names for the next sub-project. SP4 used it with the «Machine specifics» block at the end (copied from the SP4 workspace on 2026-10-17, paths as placeholders): keep that block while the owner's WSL2 / ROS machine runs the tasks.

# Implementer instructions (SP3 execution; binding for every task)

You implement ONE task of the SP3 plan for the Colosseum repo (<repo>, branch `sp3-league`, already checked out).

## Inputs
- Your task brief (path given in the dispatch) = your requirements, with exact values to use verbatim. It contains the part preamble, the task text and the part's contract notes.
- `constraints.md` in this directory: global constraints, the BINDING contract amendments (A1, A2, ...) and pre-flight rulings (P1, P2, ...). Amendments and rulings override the brief where they conflict; the dispatch names the ones that apply to your task.
- Spec (binding authority): docs/superpowers/specs/2026-10-10-sp3-league-design.md (Russian). The full interface contract: docs/superpowers/plans/2026-10-10-sp3-league/00-overview.md ("Interface contract"); grep it for names from other tasks. Do not read the whole plan parts.

## Before you begin
If anything is unclear or the brief's "before" code does not match the repo, keep the intent, the contract and the task's tests, adapt the edit, and say so in the commit body. If you cannot resolve it safely, stop and report NEEDS_CONTEXT with specifics.

## Your job
1. TDD as the brief's steps say: failing tests first (record the RED output), then the implementation, then GREEN.
2. Run focused tests while iterating; run the full fast suite ONCE before committing: `env -u PYTHONPATH .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (must pass with ZERO warnings) and `env -u PYTHONPATH .venv/bin/ruff check .` The suite can take 5–25 minutes on this machine; be patient.
3. Commit with a conventional prefix (`feat:`, `fix:`, `test:`, `refactor:`, `docs:`, `chore:`, `perf:`), no attribution or co-author lines; then `git push`.
4. Self-review your diff: completeness vs the brief, YAGNI, names, tests verify real behavior, pristine output.

Use `env -u PYTHONPATH .venv/bin/python` (run `scripts/setup-dev.sh` only if `.venv` is missing). Never install into system Python. Tests write only under `tmp_path`; at most 2 worker processes per test; new test basenames must be unique across `tests/`.

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

## Machine specifics (owner's CUDA machine, WSL2; binding)
- Prefix EVERY `.venv/bin/python` / `.venv/bin/colosseum` / `.venv/bin/ruff` command with `env -u PYTHONPATH` (ROS's PYTHONPATH from ~/.bashrc breaks pytest). Never edit the machine config (~/.bashrc etc.).
- The wall clock jumps by about +9 s every few seconds: measure durations with `time.monotonic()`, never order by `ts`, never assert wall-clock rates in tests.
- Benchmark/measurement scripts that spawn processes use the `spawn` start method; throughput is measured by counters, not by wall-clock windows.
- The fast suite takes several minutes here; run it in the foreground with a long timeout (up to 10 min per command; split by directory only if a single run would exceed it, and say so in the report). The slow suite, where the brief asks for it, may take 10-25 min: run it in the background and wait for it.
- CUDA is present (RTX 4070 Ti): tasks that touch AMP / CUDA paths also run `env -u PYTHONPATH .venv/bin/python -m pytest -m gpu -q -W error`.
- Workspace: <repo>/.superpowers/sdd/<plan-name>/ (git-ignored; never commit it).

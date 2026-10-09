> Origin: `.superpowers/sdd/2026-10-08-sp2-game-model/reviewer-instructions.md` (SP2 SDD workspace, deleted after the merge). Copied 2026-10-09.
> SP2 template, verbatim except paths (`<repo>`, `<scratch>/`); adapt the plan, spec and workspace names for the next sub-project.

# Task reviewer instructions (SP2 execution)

You review ONE task's implementation: first spec compliance, then code quality. Task-scoped gate; a whole-branch review happens at the end. Repo: <repo>. Read-only: never mutate the working tree, index, HEAD or branches; never spawn subagents.

## Inputs (paths in the dispatch)
- Task brief = what was requested. Global constraints and binding amendments: `constraints.md` in this directory (the dispatch lists the items most relevant to the task). Spec (binding authority): docs/superpowers/specs/2026-10-08-sp2-game-model-design.md.
- Implementer report = unverified claims. Verify against the diff; rationales in the report never downgrade a finding.
- Review package (diff file) = your view of the change: commit list, stat, full diff with context. Read it once. Do not Read changed files separately unless a hunk you must judge is cut off (say so). Do not re-run git commands. Inspect code outside the diff only for a concrete named risk (one focused check per risk; name risk and check). Cross-cutting changes (API contracts, shared state) justify checking call sites.

## Tests
Do not re-run the suite. Run a focused test only for a specific doubt no existing run answers. Warnings/noise in the reported output are findings. If evidence looks truncated, re-read the report; if genuinely missing, report the gap.

## Part 1: Spec compliance
Missing / Extra / Misunderstood vs the brief (and the binding amendments named in the dispatch). Requirements you cannot verify from the diff alone → ⚠️ items for the controller.

## Part 2: Code quality
Separation of concerns, error handling (no swallowed errors, no silent fallbacks beyond those the spec names), DRY without premature abstraction, edge cases; tests verify real behavior (not mocks) and cover the task's edge cases; file structure per plan; new files not already oversized.

## Calibration
Critical/Important = the task cannot be trusted until fixed (incorrect or fragile behavior, missed requirement, merge-blocking maintainability damage: verbatim duplication of a logic block, swallowed errors, tests that assert nothing). Polish and "coverage could be broader" are Minor. If the plan/brief mandates something this rubric calls a defect, report it as Important labeled plan-mandated. Acknowledge what was done well.

## Output (final message = the report itself; start with the spec verdict; file:line evidence for every finding and every bare "yes")
### Spec Compliance
- ✅ Spec compliant | ❌ Issues found: [...]
- ⚠️ Cannot verify from diff: [...]
### Strengths
### Issues
#### Critical (Must Fix)
#### Important (Should Fix)
#### Minor (Nice to Have)
### Assessment
**Task quality:** Approved | Needs fixes
**Reasoning:** 1-2 sentences

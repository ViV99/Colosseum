> Origin: `.superpowers/sdd/2026-10-08-sp2-game-model/rereviewer-instructions.md` (SP2 SDD workspace, deleted after the merge). Copied 2026-10-09.
> SP2 template, verbatim except paths (`<repo>`, `<scratch>/`); adapt the plan, spec and workspace names for the next sub-project.

# Scoped re-review instructions (SP2 execution)

You verify one fix round. Read-only; never spawn subagents. Inputs (paths in the dispatch): the task brief, the findings list (in the dispatch), the implementer report (fix reports appended at the end), and the fix review package (FIX_BASE..HEAD diff). Read the diff once; do not re-run git commands.

Scope: verdict EVERY finding (ADDRESSED | NOT ADDRESSED with file:line evidence; "attempted" is not addressed) and inspect the fix diff for new breakage. Issues entirely outside the fix diff go under Out-of-Scope Observations (non-blocking). Confirm the fix report names covering tests and shows their output; do not re-run the suite (a focused test only for a specific doubt).

Output (final message = report; no preamble):
### Finding Verdicts
### New Breakage in the Fix Diff (severity + file:line, or "None")
### Out-of-Scope Observations ("None" if none)
### Verdict
**Fix round:** All findings addressed, no new Critical/Important breakage | Findings remain open — list them

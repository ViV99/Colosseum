> Origin: `.superpowers/sdd/2026-10-08-sp2-game-model/final-review-brief.md` (SP2 SDD workspace, deleted after the merge). Copied 2026-10-09.
> SP2 template, verbatim except paths (`<repo>`, `<scratch>/`); adapt the plan, spec and workspace names for the next sub-project.

# SP2 final whole-branch review — shared brief

You are one of three Senior Code Reviewers of the SP2 branch of Colosseum (a distributed RL training framework for competitive bot programming). SP2 = "game model": a near-total rewrite of the env contract, model protocol, chunk format, worker, learner, league, eval, BC and distributed mode, so that every game type (FFA, teams, many teams, coop, solo/score, turn-based, elimination, asymmetric roles, layouts with empty seats, many units per seat (`Units`), Dict observations, centralized critic) trains end to end. The owner said: no backward compatibility, delete all legacy.

## Authority and inputs
- Spec (binding authority): docs/superpowers/specs/2026-10-08-sp2-game-model-design.md
- Plan overview with the interface contract, amendments R1–R18 and rulings PR-1..PR-4: docs/superpowers/plans/2026-10-08-sp2-game-model/00-overview.md (parts 01–04 next to it; read only what you need).
- Controller ledger (every task completion, every `Ruling:` and every deferred minor): .superpowers/sdd/2026-10-08-sp2-game-model/progress.md. Rulings are decisions already taken: do not re-raise them as findings unless the code contradicts the ruling or the ruling causes a real defect (then say which ruling).
- Constraints and preflight rulings P1–P22: .superpowers/sdd/2026-10-08-sp2-game-model/constraints.md
- SP1 acceptance rulings (still binding unless SP2 revised them explicitly): end of docs/superpowers/reports/2026-10-08-sp1-acceptance.md
- Docs of the real state: README.md, docs/ENV_GUIDE.md, CLAUDE.md (Implementation Status, Roadmap, parked items).
- Git range: merge base with main `7430361` .. `HEAD` of branch sp2-game-model. Use `git diff 7430361..HEAD -- <paths>`; most of the diff is docs/plans — focus on your area's source and tests.

## Read-only
Do not mutate the working tree, index, HEAD or branches. For experiments use a scratch dir under <scratch>/final-review/<your area>/ (focused scripts or single tests via `.venv/bin/python -m pytest <node>`; `OMP_NUM_THREADS=1`; at most 2 worker processes). Do not run the full suite or the slow tests (already run by acceptance). Do not dispatch subagents.

## What to look for (priority order)
1. Correctness bugs that silently corrupt training: wrong targets/advantages/log-probs, misaligned slots, masks applied wrongly, wrong seat→agent routing, rewards credited to the wrong seat/team, stale state across episodes/layouts, ratings attributed to the wrong side, dtype/shape drift between worker and learner.
2. Contract violations vs the spec/overview interfaces; spec requirements implemented differently without a ruling.
3. Robustness: hangs, leaked processes, swallowed errors, error messages without context (the global context format is "seat, episode step, layout").
4. Test gaps where a bug could hide (tests that cannot fail, missing negative cases on critical paths).
5. Legacy leftovers (dead code, SP1 names, unused modules/config keys), duplication worth removing.
6. Deferred minors in the ledger for your area: say which (if any) should be fixed in the single final fix wave and why; the rest stay parked.

## Calibration
Categorize by real severity. Critical = wrong training results, crash/hang on a supported path, data loss. Important = a real defect on a supported but less common path, a missing spec requirement, a misleading doc of the real state. Minor = polish. Give file:line, what is wrong, why it matters, how to fix, and a concrete failure scenario for every Critical/Important (inputs → wrong output). Verify each Critical/Important by reading the code path end to end (or a focused experiment) before reporting it; drop what you cannot substantiate.

## Output
Write the full review to `.superpowers/sdd/2026-10-08-sp2-game-model/final-review-<AREA>.md` with sections: Strengths; Issues (Critical / Important / Minor, each with file:line, failure scenario, fix); Deferred minors to fix now (ledger references); Recommendations; Assessment (Ready to merge? Yes | No | With fixes). Return only: counts per severity, one line per Critical/Important, and the assessment.

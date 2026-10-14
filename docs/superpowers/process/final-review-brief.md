> Origin: `.superpowers/sdd/2026-10-10-sp3-league/final-review-brief.md` (SP3 SDD workspace, deleted after the merge). Copied 2026-10-15.
> The SP2 template as revised for SP3, verbatim except paths (`<repo>`); adapt the plan, spec, branch and workspace names for the next sub-project.

# SP3 final whole-branch review — shared brief

You are one of three Senior Code Reviewers of the SP3 branch `sp3-league` of Colosseum (a distributed RL training framework for competitive bot programming; repo `<repo>`). SP3 = "players, league, warm start": agent kinds (trainable / scripted / frozen) in one `agents:` section, scripted bots (`ScriptedBot`) and frozen players seated by `MatchRunner`, snapshot retention (`keep_last` / `keep_every` / final) with worker eviction and run-dir pool import, PFSP statistics per player, the `colosseum.league` matchmaker (opponent categories with shares, schedules, per-agent overrides, custom `BaseMatchmaker` classes), per-agent warm start (`init` strict/partial, critic warm-up, kickstart from a neural or scripted teacher = DAgger labels in chunks), `colosseum record` and multi-source `bc`, distributed-mode guards, demo bots, slow pipeline test, team_tag anchors, APPO Units defaults measurement, docs. The owner required backward compatibility with SP2 configs and checkpoints.

## Authority and inputs
- Spec (binding authority, Russian): `docs/superpowers/specs/2026-10-10-sp3-league-design.md`
- Plan overview with the interface contract and amendments A1–A33: `docs/superpowers/plans/2026-10-10-sp3-league/00-overview.md` (parts 01–03 next to it; read only what you need).
- Constraints and pre-flight rulings P1–P18: `.superpowers/sdd/2026-10-10-sp3-league/constraints.md`
- Controller ledger `.superpowers/sdd/2026-10-10-sp3-league/progress.md`: every task completion, every `Ruling:` and every deferred minor (`minor (deferred)`). Rulings are decisions already taken: do not re-raise them as findings unless the code contradicts the ruling or the ruling causes a real defect (then say which ruling).
- SP1 and SP2 rulings (still binding unless SP3's spec §8 «Изменения правил SP2» changes them): ends of `docs/superpowers/reports/2026-10-08-sp1-acceptance.md` and `docs/superpowers/reports/2026-10-09-sp2-acceptance.md`.
- Docs of the real state: `README.md`, `CLAUDE.md`, `docs/LEAGUE_GUIDE.md`, `docs/ENV_GUIDE.md`, `docs/benchmarks.md`, `docs/GPU_CHECKS.md`.
- Git range: merge base with main `c16fe5e` .. `HEAD` of `sp3-league`. Use `git diff c16fe5e..HEAD -- <paths>`; much of the diff is docs/plans/fixtures — focus on your area's source and tests.

## Read-only
Do not mutate the working tree, index, HEAD or branches, and write no files anywhere except your output file below. For experiments you may run focused scripts or single tests with `.venv/bin/python` (`OMP_NUM_THREADS=1`, ≤2 worker processes); put any scratch script under `.superpowers/sdd/2026-10-10-sp3-league/final-review/<AREA>/` (git-ignored). Do not run the full suite, the slow tests, training runs or benchmarks (acceptance does that). Do not dispatch subagents.

## What to look for (priority order)
1. Correctness bugs that silently corrupt training or results: wrong seat→player routing, a fixed player that collects, labels/teacher actions on the wrong slots, a loss term with the wrong denominator, PFSP statistics attributed to the wrong player, ratings mixing snapshots and anchors wrongly, schedules evaluated at the wrong step, evicted snapshots still scheduled, resume restoring the wrong state, init loading the wrong tensors, critic warm-up changing the policy, config translation changing SP2 semantics beyond what spec §8 lists, dtype/shape drift between worker and learner.
2. Contract violations vs the spec / overview / amendments; spec requirements implemented differently without a ruling.
3. Robustness: hangs, leaked processes, swallowed errors, errors without context (format "worker W, env E, seat P, episode step K, layout L" + agent for players), silent fallbacks the spec does not name.
4. Test gaps where a bug could hide (tests that cannot fail, missing negative cases on critical paths).
5. Leftovers: dead code, stale names (LineupMatchmaker, AgentPool, pool_size, training.kickstart_* outside the compat translation), duplication worth removing.
6. Deferred minors in the ledger for your area: say which (if any) should be fixed in the single final fix wave and why; the rest stay parked.

## Calibration
Categorize by real severity. Critical = wrong training results, crash/hang on a supported path, data loss. Important = a real defect on a supported but less common path, a missing spec requirement, a misleading doc of the real state. Minor = polish. Give file:line, what is wrong, why it matters, how to fix, and a concrete failure scenario for every Critical/Important (inputs → wrong output). Verify each Critical/Important by reading the code path end to end (or a focused experiment) before reporting it; drop what you cannot substantiate.

## Output
Write the full review to `.superpowers/sdd/2026-10-10-sp3-league/final-review-<AREA>.md` (the only file you write) with sections: Strengths; Issues (Critical / Important / Minor, each with file:line, failure scenario, fix); Deferred minors to fix now (ledger references); Recommendations; Assessment (Ready to merge? Yes | No | With fixes). Return only: counts per severity, one line per Critical/Important, and the assessment.

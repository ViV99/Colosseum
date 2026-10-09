> Origin: `.superpowers/sdd/2026-10-08-sp2-game-model/readiness-review.md` (SP2 SDD workspace, git-ignored, deleted after the merge). Copied 2026-10-09.
> Frozen record; do not edit. Only absolute scratch paths were replaced with `<scratch>/`. Line numbers refer to commit `70e65ab`; its gaps were fixed in HANDOFF round 2.

# SP3 readiness review: can a fresh session start from the repo docs alone? (HEAD 70e65ab)

Reviewer: read-only, nothing in the repo changed. Scratch: `<scratch>/readiness/`.
Commands run: `pytest --collect-only` for each marker, `ruff check .`, `colosseum <cmd> --help` for every command, a link and path checker over CLAUDE.md, the SP2 acceptance report, README, ENV_GUIDE, benchmarks, GPU_CHECKS and the review dir, plus greps against `src/` and `tests/`.

---

## 1. Answers (simulated fresh session, docs only)

### 1.1 What is implemented, by game type
- **Common base:** `GameSpec` + `MultiAgentEnv` + `StepResult`, with the `EpisodeTracker` contract checks (CLAUDE.md:267-271). APPO/V-trace runs over `act`/`boot`/`pad` slots, and the learner computes every value (CLAUDE.md:272-273). One `MatchRunner` serves training and eval (CLAUDE.md:272, 276).
- **Solo (score):** `coin_grid` (CLAUDE.md:278; ENV_GUIDE.md:11). The outcome kind comes from the team count: 1 team → `score` (CLAUDE.md:269).
- **1v1, turn-based:** `tic_tac_toe`, which also has attention and multi configs. Simultaneous moves: `unit_harvest` and `tron` (acceptance report:188).
- **One bot with many units:** `Units(max_units, per_unit, only_if)`, `ratio_mode`, `unit_trace`, `entropy_reduction` (CLAUDE.md:271, 273). Demo: `unit_harvest`. Reference: `space_miners` (Box2D).
- **Team vs team:** `team_tag` 2v2 with `global_state` and a centralized critic (CLAUDE.md:273, 278). League teams have a core agent plus `teammates: self | mixed` (CLAUDE.md:115, 275).
- **FFA with elimination, 2–4 players in one run:** `tron` (empty seats, `rank` outcome).
- **1 vs N, asymmetric roles:** `predator_prey` (`agents.<id>.roles`, one agent per role). An agent cannot play roles with different spaces (CLAUDE.md:292).
- **Cooperative:** `coop_buttons` (`score`, cross-play).
- **Single machine only.** Distributed mode covers only games where every agent plays every role, on latest weights, with no league (CLAUDE.md:285).
- **Not implemented:** scripted, frozen and external players; PFSP over snapshots; per-agent warm start and kickstart teacher; top-k snapshots (all SP3, CLAUDE.md:291). Also missing: OpenSkill/BT, tournament and match log (SP4); GPU inference and vectorized fast path (SP6); distributed league (SP5). The "never" list is at CLAUDE.md:292.

### 1.2 The env contract a user implements
- **Interface:** a `MultiAgentEnv` subclass with `spec: GameSpec` and two methods, `reset(seed, layout) -> StepResult` and `step(actions) -> StepResult` (ENV_GUIDE.md:21-60).
- **`GameSpec`:** roles, layouts and teams (ENV_GUIDE.md §2:61).
- **`StepResult`:** `acting`, rewards, `terminated`, `truncated` + `final_obs`, `episode_over`, the seat lifecycle and rewards to waiting seats (ENV_GUIDE.md §3:111-188).
- **Spaces:** observation trees with `uint8` and the Dict key-order caveat (§4:190-220). `Outcome` (§5:221). `global_state` (§6:239). Actions, `Units` and masks: `dtype=bool`, `only_if` only inside `Units` (§7:258-326).
- **Model and checks:** model (§8:328), `colosseum validate` (§9:352).
- **Authority:** the full rules are in the SP2 spec, blocks 1–3 (ENV_GUIDE.md:5).
- **Code:** `src/colosseum/envs/{game,spaces,contract}.py`.
- **Short summary:** CLAUDE.md:94-97, 229-233.

### 1.3 Decisions and rulings SP3 must respect
- **SP1:** 58 rulings at the end of `docs/superpowers/reports/2026-10-08-sp1-acceptance.md:411-473`, plus residuals at :474.
- **SP2:** PR-1..PR-5 (acceptance report:399-405), P1–P22 (:407-432), the 42 ledger rulings (:434-477) and the T8.3–T8.4 rulings (:479-511). The pointer is CLAUDE.md:364.
- **SP2 spec §8:** the brainstorm decisions, the SP1 rule changes and the risks (spec:510-562).
- **Contract amendments R1–R18:** `docs/superpowers/plans/2026-10-08-sp2-game-model/00-overview.md` (CLAUDE.md:255).
- **Standing rules repeated in CLAUDE.md:**
  - agent-id regex and reserved ids (:280);
  - strict resume (:281);
  - `SHUTDOWN_GRACE_SEC = 7` (:281);
  - exactly `batch_chunks` chunks per update (:272);
  - the bench `_make_config` rule (:371).

### 1.4 Open owner questions
CLAUDE.md:351-355 and acceptance report:529-533:
- team_tag best-of-two;
- `ratio_mode: auto` = `per_unit`;
- `unit_trace: auto` = `joint`;
- GPU checks (18 tests) not yet run.

The defaults stand, and the questions are to be decided at the SP3 brainstorm.

### 1.5 Parked items by sub-project
- **SP3:**
  - Roadmap row (CLAUDE.md:305): players (also as BC data recorders and DAgger experts), PFSP over snapshots (including opponent checkpoints in asymmetric games, balanced `x(1-x)` weighting), anti-passivity, per-agent `init`, a kickstart teacher per agent, critic warm-up, top-k storage.
  - team_tag single-run requirement (:336).
  - `AgentPool` legacy and `units_component_valid` (:343).
- **SP4:**
  - BT checkpoint selection (:312);
  - `dropped_reward_episodes` monotonic counter (:341);
  - full eval/MatchRunner unification is done; nothing else is listed.
- **SP5:** CLAUDE.md:314-320, 339-340, 344, and the roadmap row at :307.
- **SP6:** CLAUDE.md:322-325, 346, and the row at :308.
- **Unscheduled:** CLAUDE.md:327-334, 347.
- **Full minor lists:** acceptance report:567 and the appendix at :571-613.

### 1.6 Development process
- **Order and branch:** brainstorm → spec → plan (overview + parts) → branch `spN-<name>` → acceptance report → merge (CLAUDE.md:359).
- **Merge rule:** `--no-ff` only after the owner's explicit acceptance; push regularly (:360).
- **Subagents:** one per task (TDD and commit), a spec+quality review after each task, scoped re-reviews, then a whole-branch review with one fix wave, then acceptance (:361).
- **Models:** SP2 used Opus/high for implementers and reviewers and three area reviewers (:362).
- **Ledger:** the SDD workspace `.superpowers/sdd/<plan>/` holds briefs and the `progress.md` ledger. It is scratch, deleted after the merge, and anything durable goes to `docs/` (:363).
- **Rulings format:** "what — why — cost if wrong" (:364).
- **Reports:** `docs/superpowers/reports/` (:359).

### 1.7 Tests and benchmark
- **Setup:** `scripts/setup-dev.sh`.
- **CI suite:** `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` with zero warnings, plus `-m slow` and `.venv/bin/ruff check .` (CLAUDE.md:367-368).
- **Conventions:** CLAUDE.md:369-371.
- **README:** README.md:367-372 (slow suite 11–16 min).
- **Benchmark:** `.venv/bin/python scripts/bench_throughput.py --workers 1 2 4 --duration 60 --warmup 15` (docs/benchmarks.md:40). Load average must be < 1 (benchmarks.md:74, 139).
- **Other measurements:** units experiment (benchmarks.md:161), `measure_global_state.py` (:196).

### 1.8 What the SP2 spec says SP3 should do
- spec:37 (§2): scripted, frozen and external players; PFSP over snapshots, including opponent checkpoints in asymmetric games; per-agent warm start (`init`, critic warm-up); top-k snapshot storage. BC and kickstart were only ported in SP2.
- spec:313: scripted players are SP3; until then any `PolicyModel` plays in-process.
- spec:385: a kickstart teacher per agent is SP3.
- spec:516: full league with teams and asymmetry belongs to SP3; SP2 shipped only the minimum.
- Added by the acceptance report: anti-passivity / PFSP over snapshots so team_tag passes with one run (acceptance report:502, :519).

### 1.9 Where the docs did not answer, or answered ambiguously
1. **Where SP3 starts.** Nothing points to the SP3 design input. For SP2 the inputs were `review/code/06_game_types.md` §3 and `review/research/D` (spec:8). The SP3 equivalents are `review/README.md` §6.3, §6.5 (and §6.4 for snapshot storage), `review/code/04_league.md` «Proposed design» and `review/research/B_league_rating_warmstart.md`. CLAUDE.md never names them, and calls `review/` only a "frozen snapshot".
2. **What "external player" means.** It is defined only in review/README.md:301 ("External — чужой `.pt` + описание сети").
3. **Owner priorities** (heterogeneous machines, warm starts, an arena with scripted bots, best-agent selection, all game types). They exist only in user memory (`colosseum-owner-priorities.md`). CLAUDE.md "Project Vision" lacks them.
4. **Communication rules.** The rules are: reply in Russian; long results go on a private, phone-readable artifact page with a short Russian summary; autonomous runs continue without asking and stop only for merges or destructive actions; the owner approves the spec section by section. These are in memory only (`colosseum-dev-workflow.md`, `user-delegation-style.md`).
5. **Models for SP3.** CLAUDE.md:362 records only that SP2 used Opus/high "(owner's instruction)". Web research on Sonnet/high appears only in spec §7 and the plan overview, not in CLAUDE.md. It is unclear whether this carries over to SP3.
6. **Commit and language conventions.** These are: conventional prefixes, **no attribution or co-author lines**, push after every task; code and README/CLAUDE.md in English; specs, reports and new docs in Russian. They exist only in the frozen plan overviews (sp2 00-overview.md:47-50, sp1 00-overview.md:46-52) and in the scratch `constraints.md`. A fresh session would likely add `Co-Authored-By` lines by default.
7. **SDD mechanics.** These exist only in `.superpowers/` or the job scratch dir: the implementer, reviewer and re-reviewer instruction files, `final-review-brief.md`, `brief.sh`, the review-package script, the ledger line formats, the 5-round fix cap, and the status contract (DONE / DONE_WITH_CONCERNS / BLOCKED / NEEDS_CONTEXT). They are partly reconstructible from the superpowers skills but not from the repo.
8. **Backward compatibility in SP3.** "No backward compatibility" was the owner's instruction for SP2 (spec:11, spec:§8). Whether SP3 may break SP2 configs or checkpoints (`meta.json`, `players:` config) is not stated.
9. **Scheduling contradictions:**
   - The shared-memory weight store and dynamic add/remove of agents and machines are "(SP5)" at CLAUDE.md:297 but "Minor, unscheduled" at :331-332.
   - Tier 2 algorithms are labelled "(next)" (CLAUDE.md:50), while the roadmap puts "new algorithms" in SP6 (:308).
   - Off-policy/SAC/MuZero (:295) have no sub-project.
10. **AgentPool rotation order.** Owner rotation depends on `AgentPool.list_trainable()` keeping config order. This was a ledger carry-forward from T5.3 and is relevant when SP3 adds frozen or scripted entries to `AgentPool`. It is not recorded anywhere in the repo (coordinator.py:98-106 only implies it).
11. **Closure of routed minors.** The acceptance report appendix says only that the "deferred→T…" lines were routed, not that they were closed (acceptance report:575). I verified them (section 4).
12. **README vs CLAUDE.md roadmap.** The README roadmap (README.md:346-353) is shorter than CLAUDE.md:303-308: SP3 lacks anti-passivity, balanced PFSP and recorders/DAgger; SP5 lacks the distributed league; SP6 lacks the vectorized path. It is unclear which one is authoritative.
13. **Research list.** CLAUDE.md "Research Documents" omits `research/MULTI_AGENT_RL_RESEARCH.md` (1397 lines) and `research/dialogue.txt`, the owner's original brief in Russian.
14. **Slow suite.** CLAUDE.md:368 says "`-m slow` for the learning test", singular. There are 9 slow tests (7 demo learning tests and 2 torch.compile tests), 11–16 min. It also does not say when the slow suite must run (after each task, or at acceptance only as in SP2).
15. **Memory outside the repo is stale.** `memory/colosseum-sp2-plan.md` and its MEMORY.md index line say SP2 is "awaiting owner's merge yes / Not merged". A fresh session loads this alongside CLAUDE.md and gets conflicting status.

---

## 2. Gaps with suggested exact fixes

| # | File / location | Suggested text |
|---|---|---|
| G1 | CLAUDE.md «Next step», after line 350 | `- Design inputs for SP3 (pre-SP1, re-check every finding against current code): review/README.md §6.3 (players: Trainable / Snapshot / Scripted / External = a foreign .pt + network description; policy pool in the worker; matchmaker over snapshots), §6.4 (snapshot storage: keep_last / keep_every / top-k / pinned), §6.5 (warm start: per-agent init, critic_warmup_steps, kickstart from any Policy, colosseum record), review/code/04_league.md «Proposed design», review/research/B_league_rating_warmstart.md.` |
| G2 | CLAUDE.md «Project Vision», after line 7 | `Owner priorities (2026-10-07; judge every design against them): 1) workers and learners on different machines with different hardware, machines and agents added/removed dynamically; 2) flexible, simple deployment and scenario config; 3) several warm-start options (offline BC, kickstarting, resume); 4) a final arena where bots with different weights/architectures and scripted bots play, with convenient best-agent selection and full monitoring; 5) every game type: solo, 1v1 (turn-based and simultaneous), one bot with many units, team vs team, 1 vs N (FFA or asymmetric).` |
| G3 | CLAUDE.md «Development Workflow», new bullet after line 360 | `- **Working with the owner:** the owner writes in Russian; reply in Russian. Specs are approved section by section in the brainstorm. When left alone, keep working without asking, decide conflicts as rulings, and stop only for merges and destructive, security-sensitive or outward actions. Long results (acceptance, reviews) also go on a private artifact page readable on a phone, with a short Russian summary in chat.` |
| G4 | CLAUDE.md line 362, append | `Default for SP3 unless the owner says otherwise: Opus/high implementer and Opus/high reviewer per task (implementers never dispatch subagents), Sonnet/high for web research, the main session orchestrates and takes rulings.` |
| G5 | CLAUDE.md «Development Workflow», new bullet | `- **Conventions:** commits use conventional prefixes (feat/fix/refactor/test/docs/chore/perf) and carry no attribution or co-author lines; push after every task. Code, comments, docstrings, logs, README.md and CLAUDE.md are in English; specs, plans' prose, reports and new project docs (ENV_GUIDE, benchmarks, GPU_CHECKS) are in Russian.` |
| G6 | Before deleting `.superpowers/`: copy `implementer-instructions.md`, `reviewer-instructions.md`, `rereviewer-instructions.md`, `final-review-brief.md` to `docs/superpowers/process/` (header: "templates from SP2; adapt names and paths"). Then add to CLAUDE.md:363 | `Templates for briefs and review instructions: docs/superpowers/process/. Ledger lines: "Task X: complete (commits A..B, review clean[ after fix round n])", "Ruling: what — why — cost if wrong", "Task X: minor (deferred[→Tn]): …"; at most 5 fix rounds per task.` |
| G7 | CLAUDE.md:343 (SP3 minors), append | `the owner rotation (coordinator.generate_lineups) relies on AgentPool.list_trainable() returning trainable agents in config order (T5.3); keep that when adding frozen/scripted entries;` |
| G8 | CLAUDE.md:351-354 (open owner questions), add | `- backward compatibility in SP3: none, as in SP2 (configs/checkpoints may break; SP2 checkpoints would need a re-train), or keep SP2 checkpoints resumable / as frozen players?` |
| G9 | CLAUDE.md:297 and :331-332; :50 | Decide one place. Either drop "Shared-memory weight store …; adding or removing agents and machines during a run" from :297 and leave them unscheduled, or delete :331-332 and add both to the SP5 row (:307). At :50 replace "**Tier 2 (next):**" with "**Tier 2 (SP6 or later; see Roadmap):**". At :295 add "(SP6 or later)". |
| G10 | CLAUDE.md «Research Documents», after line 15 | ``- `research/MULTI_AGENT_RL_RESEARCH.md` — survey of multi-agent / multiplayer RL systems (landmark systems, methods)`` and ``- `research/dialogue.txt` (Russian) — the owner's original brief for the project`` |
| G11 | acceptance report:575, append a sentence | `Строки, направленные «deferred→T…», закрыты в названных задачах (проверено при ревью готовности): 4 (T0.1→T8.6, список остатков в CLAUDE.md), 24 (тест kickstart-учителя в test_sp2_learner_entry.py и тесты жизненного цикла), 30 (ключи рейтингов по вариантам — test_sp2_metrics_outputs.py:30-33; консольный скрипт — test_sp2_lifecycle.py::test_readme_quickstart_from_repo_root), 31 (комментарии conftest.py:25,48 и core/errors.py:15), 35 («2 of 9» в test_demo_learning_slow.py:88; в отчёте одна версия цены, :501).` (The report is frozen after acceptance only by convention; if it may not be edited, put this in CLAUDE.md:342.) |
| G12 | README.md:350-353 | Copy the SP3/SP5/SP6 row text from CLAUDE.md:305-308, or add under the table "Full scope per sub-project: CLAUDE.md «Roadmap»." |
| G13 | CLAUDE.md:368 | `plus .venv/bin/python -m pytest -m slow -v (9 tests: the 7 demo-game learning tests and 2 torch.compile checks, 11–16 min; run at acceptance and after changes to the algorithm, model or demo games) and .venv/bin/ruff check .` |
| G14 | Not a repo file: `~/.claude/projects/-home-viv-dev-repos-Colosseum/memory/colosseum-sp2-plan.md` + MEMORY.md line | After the merge, update the status to "merged into main <date>; SDD workspace deleted; next: SP3", or the fresh session reads "not merged". |

---

## 3. Inaccuracies (claim → reality)

1. **CLAUDE.md:273.**
   - Claim: "with one decider every mode collapses to `joint`".
   - Reality: an explicit `unit_trace: none` stays `none` with one decider (`src/colosseum/algorithms/appo.py:66-70`; ruling PR-1, acceptance report:401).
   - Fix: "…collapses to `joint` (except an explicit `unit_trace: none`)".
2. **CLAUDE.md:297 vs :331-332.**
   - The same two items (the shared-memory weight store; adding or removing agents and machines during a run) are marked SP5 in one place and "Minor, unscheduled" in the other.
3. **CLAUDE.md:368.**
   - Claim: "`-m slow` for the learning test".
   - Reality: 9 slow tests, of which 7 are demo learning tests (`tests/learning/test_demo_learning_slow.py`) and 2 are torch.compile checks (`tests/unit/test_sp2_performance.py`).
4. **CLAUDE.md:127 vs :124.**
   - :127 says "`--num-matches` is per pair"; :124 says "per pair (or composition) and layout".
   - The two lines are internally inconsistent (minor).
5. **CLAUDE.md:340.**
   - "identical per-env seeds … (ruling PR-3)": PR-3 is "layouts fixed per worker env" (00-overview.md:975).
   - The identical seeds are a consequence of seeding by `training.seed + worker_id` with the same rotation (`distributed.py:22-24`).
   - Suggest "(a consequence of ruling PR-3)" (wording only).
6. **CLAUDE.md:254 and acceptance report:6.**
   - "merged into `main` on 2026-10-09" is not yet true at HEAD 70e65ab (`main` is at 7430361).
   - It is correct only if the merge happens today; otherwise fix the date.
7. **Code, not docs (parked as C M-x, acceptance report:567).**
   - The docstrings still say "Unused in SP1": `src/colosseum/core/config.py:454,460,464`, `transport/local.py:3,23`, `weight_store/shared_memory.py:8,54`.
   - CLAUDE.md:287 correctly says "Unused, kept for SP5". Listed only so that SP3 does not trust the code wording.

No other claim I checked was wrong.

### Claims verified correct (32)
1. Test counts 1248 / 9 / 18: `1248/1275`, `9/1275`, `18/1275` collected.
2. `ruff check .` clean.
3. GPU list in `docs/GPU_CHECKS.md` equals `pytest -m gpu --collect-only` (diff empty).
4. CLI flags (CLAUDE.md:259-263, all `--help`):
   - eval: `-a`, `--layout`, `-n/--num-matches`, `--num-envs`, `--deterministic`, `--seed`, `-o`;
   - bc: `-a/--agent`, `-d`, `-o`, `--epochs`, `--batch-size`, `--lr`, `--seq-len`;
   - serve-weight-store: `--port`, `--max-message-mb`;
   - run-learner: `-c`, `-a`, `--traj-port`, `--weight-store`, `--set`;
   - run-workers: `-c`, `--weight-store`, `-l`, `--set`.
5. `rollout.weight_sync_interval_sec` (config.py:263, default 5.0).
6. `algorithm.vtrace_lambda` (:116).
7. `ratio_mode` auto/joint/per_unit and `unit_trace` auto/joint/geo_mean/none, `entropy_reduction` (:152-164).
8. `resolve_modes` gives auto → per_unit with Units, `unit_trace` auto → joint (appo.py:57-80).
9. `networks.critic_encoder_class` (:217).
10. `training.kickstart_teacher`, `kickstart_kl` default forward (:334, 346).
11. `rollout.chunk_length` default 256 (:256).
12. `rollout.vec_env: sync | subprocess` (:268).
13. `matchmaking.teammates: self | mixed`, `pfsp_exponent`, `mode: self_play | league`, `shuffle_seats` (:356-386).
14. One global `checkpoint.interval`, `pool_size` (:400-407).
15. `env.max_idle_steps` (:179).
16. `env.num_players`, `training.phase` and `self_play.*` are gone (only in the docstring "Changes from SP1").
17. Per-agent overrides `roles`/`networks`/`algorithm`/`learner` (`AgentOverride`, :478-498).
18. `RESERVED_AGENT_IDS` = ratings/system/episodes/train (:58).
19. Run name default `<config_stem>-<YYYYmmdd-HHMMSS>`, `run.dir` default `runs` (:432-441).
20. `transport.mode` / `grpc_port` exist (:458-464).
21. `SHUTDOWN_GRACE_SEC = 7.0` (utils/process.py:27).
22. `serve-weight-store` Ctrl-C → 130 (cli.py:340-352).
23. `meta.json` has `policy_version`, `env_steps`, `roles`, `role_signature` (checkpoint_manager.py:99-115). Resume without a signature is a `ConfigError` "SP1 checkpoints cannot be resumed" (:370-384).
24. `ratings.json` = `{"env_steps", "layouts"}` (metrics/hub.py:142).
25. `system` record has `dropped_reward_episodes` (metrics/jsonl.py:31, aggregator.py:171).
26. Wire format: JSON skeleton + `np.savez`, optional lz4, no pickle (transport/serialization.py:1-21). Client streaming `SendChunks(stream …)` (proto/colosseum.proto:48).
27. Distributed checkpoint queue is 16 slots (distributed.py:255).
28. Symbols exist:
    - `EpisodeTracker`, `EnvContractError`, `GameSpec`, `MultiAgentEnv`, `StepResult`, `Outcome`, `Units(max_units, per_unit, only_if)`;
    - `UnitsHead`, `gridnet_to_units`, `NormalizeObs`, `EncoderOutput`, `BaseCriticEncoder`;
    - `play_lineups`, `schedule_lineups`, `ScoreTracker`, `Lineup`, `SeatAssignment`, `LineupMatchmaker`, `MatchRunner`, `RolloutLoop`;
    - `units_component_valid`, `distributed_setup`, `setup_run`, `GRPCTransport`, `LocalTransport`, `SharedMemoryWeightStore`;
    - the diagnostics `clip_fraction_joint` / `ess` / `log_rho_abs_p95` / `deciders_valid_mean`.
29. `AgentPool.register_frozen` / `register_scripted` and `AgentHandle.elo` / `win_rates` have no callers (only `register_trainable` / `list_trainable` are used, coordinator.py:46-102).
30. The shadow package `src/colosseum/sp2` is gone, and no `colosseum.sp2` references remain in src/tests/scripts/configs/examples.
31. `tests/game_helpers.py`, `tests/cli_runner.py` and `tests/contract/game_harness.py` exist. `scripts/bench_throughput.py::_make_config` is guarded by `tests/unit/test_bench_throughput.py:117`.
32. Demo configs: all 11 in `configs/examples/`, 9 example dirs. Throughput 4848 / 9298 / 13702 matches docs/benchmarks.md:145-147.

### "(target; see Roadmap)" markings
Every marked target has a roadmap or "Not implemented" entry:
- :25 and :77 → shared-memory weight store;
- :70 → SP5 control plane;
- :78 → trajectory ring buffer;
- :101 and :204 → SP3 players;
- :117 → SP3 balanced PFSP and snapshots;
- :118 → SP4 OpenSkill;
- :119 → dynamic add/remove;
- :137 and :138 → SP3;
- :146 → SP5;
- :183 → SP5;
- :219 → SP4 dashboard.

The marking is correct, but :25, :77 and :119 point to items whose sub-project is contradictory (Inaccuracy 2). Unmarked targets that are still plain rationale or vision are fine: :31 "dynamic add/remove", and :72 "gRPC or shared memory" (shared memory is marked at :25).

### Paths and links
- 170 backticked repo paths and relative links in CLAUDE.md and the acceptance report resolve.
- The only misses are expected:
  - run-dir-relative names (`logs/`, `checkpoints/…`);
  - `/tmp/…` and `/eval.json` (`.gitignore` text);
  - `dist/` (ruling text);
  - `tests/integration/test_bc_cli.py`, which the report correctly describes as moved in T7.3.
- The links in README, ENV_GUIDE, benchmarks, GPU_CHECKS and the review dir resolve. The non-path misses are metric names.
- No tracked doc depends on `.superpowers/` beyond origin headers and hygiene lines.

---

## 4. Forwarded-items check (acceptance report appendix)

| Ledger line | Forwarded to | Status at HEAD | Evidence |
|---|---|---|---|
| 4 [T0.1] CLAUDE.md residuals list stale | T8.6 | **Fixed** | No "Residuals from the SP1 final review" section remains. The 4 items: second Ctrl+C, bc unreadable file and the `core/ipc` move are closed (acceptance report §3.10:248-269). The distributed final-checkpoint item is in the SP5 list (CLAUDE.md:319). `registry._reset_mask_row` is gone. |
| 30 [T7.1] per-layout ratings keys in metrics.jsonl end to end | T7.3 | **Fixed** | `tests/integration/test_sp2_metrics_outputs.py:30-33` asserts `ratings.json` = `{env_steps, layouts}` and the last `ratings` record's `layouts["2p"]` keys. Also `test_sp2_game_runs.py:35-36`. |
| 30 [T7.1] console-script check | T7.3 | **Fixed** | `tests/integration/test_sp2_lifecycle.py:219-229` (`test_readme_quickstart_from_repo_root`) asserts `.venv/bin/colosseum` exists and trains. The entry point is `colosseum = "colosseum.cli:main"` (pyproject.toml:35-36). |
| 31 [T7.3] stale comments `tests/conftest.py:25,47-48` | T8.6 | **Fixed** | conftest.py:25 names `game_helpers`, `cli_runner`. :48 names `game_harness`, which exists as `tests/contract/game_harness.py`. Fixed in cbc0d4c. |
| 31 [T7.3] `core/errors.py:15` "BaseEnv contract" | T8.6 | **Fixed** | errors.py:14-16 now reads "violated the `MultiAgentEnv` contract … `EpisodeTracker`". |
| 35 [T8.3] team_tag docstring "one run in seven" → "2 of 9" | T8.6/T8.7 | **Fixed** | `tests/learning/test_demo_learning_slow.py:88`: "about 2 of 9 single runs". |
| 35 [T8.3] cost numbers consistent (0.86→0.43/67% vs 0.8→0.4/64%) | T8.6/T8.7 | **Fixed** | Only one version is left in the repo: acceptance report:501 "с ~0.8 до ~0.4 … ~64%". No 0.86/0.43/67 anywhere in docs. Ruling 33 (:468) quotes "~1/7 … ~98%" verbatim, and :500 reconciles it (98% at 6/7, 95% at 2/9). |
| (extra) 24 [T5.4] CARRY to T7.1: kickstart-teacher branch in `_learner_main`; sp2 lifecycle coverage | T7.1 | **Fixed** | `tests/unit/test_sp2_learner_entry.py:202` (`test_learner_target_builds_and_uses_the_kickstart_teacher`); `tests/integration/test_sp2_lifecycle.py`, `tests/unit/test_sp2_launcher_lifecycle.py`. |
| (extra) ledger nits outside the appendix: units_experiment.py `DEFAULT_TRACE` comment (T8.4 nit); eval.py:31,34 "unattributed" docstring (final re-review) | T8.6 / docs commit | **Fixed** | scripts/units_experiment.py:53 says "the current default is joint". eval.py:29-35 says `schedule_lineups` no longer produces mixed lineups. |
| (extra) T5.3 carry: rotation relies on `list_trainable()` config order | not routed | **Not in docs** | See Gap G7. |

Still open, as the appendix says: line 2, T0.1 duplicate `cancel_join_thread` blocks (launcher.py:875, 907 vs ipc.py:349).

---

## 5. Verdict

**With fixes.** The repo answers almost every question a fresh SP3 session needs:
- status by game type;
- the env contract;
- every ruling list and its location;
- open owner questions with their defaults;
- parked items by sub-project;
- test and benchmark commands;
- what the SP2 spec hands to SP3.

The CLAUDE.md claims I checked match the code, and all forwarded ledger items were really closed.

Before the merge, and before `.superpowers/` is deleted, do these:
- **G1:** point to the SP3 design inputs in `review/`.
- **G5:** commit and language conventions, especially "no co-author lines".
- **G6:** preserve the SDD instruction templates.
- **G3/G4:** owner communication rules and SP3 subagent models; today they are only in user memory.
- **G9 and Inaccuracies 1–3:** contradictory scheduling and small wording errors.
- **G14:** update the stale memory file after the merge.

Everything else is optional polish.

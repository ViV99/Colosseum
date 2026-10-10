> Origin: `.superpowers/sdd/2026-10-10-sp3-league/final-review-A.md` (SP3 SDD workspace, git-ignored, deleted after the merge). Copied 2026-10-10.
> Frozen record; do not edit. Copied verbatim (no absolute scratch paths to replace).

# SP3 final whole-branch review — Area A (players, config and execution)

Reviewer: area A. Range `c16fe5e..c31eda9` (`sp3-league`). Read in full: `core/config.py`, `players/registry.py`,
`players/scripted.py`, `worker/match_runner.py`, `worker/rollout_loop.py`, `worker/rollout_worker.py`, the
`launcher.py` diff and launch path, `coordinator/checkpoint_manager.py` (diff + read/resume paths),
`coordinator/coordinator.py`, `league/lineups.check_lineup`, `eval.py` (engine, `load_player`, `evaluate`),
`core/validation.py` diff, `cli.py` diff, the SP2 compatibility fixtures and tests. Ledger T0.1, T1.1–T1.5,
T2.1–T2.2 and the rulings P1–P18 checked against the code.

Experiments (scratch under `.superpowers/sdd/2026-10-10-sp3-league/final-review/A/`):
- `sp2_resolved.py`: every SP2 fixture config was loaded with SP2's own `config.py` (from `git show c16fe5e`),
  dumped as SP2 would write `config.resolved.yaml` (every knob explicit: `mode`, `self_play_ratio`,
  `latest_prob`, `pfsp_exponent`, `pool_size`, `training.kickstart_*`, `agents.<id>` with null overrides), and
  loaded with SP3: all 11 load; translations as the spec says (e.g. `tic_tac_toe_multi` →
  `rivals: 1.0`), one warning per knob family.
- `alias_probe.py`: `lambda_` / `from_` next to their aliases in per-agent or top-level sections are rejected
  (no silent precedence between alias and field name).
- A one-config pipeline (`unit_harvest_league.yaml` + `agents.main.init.from: bc.pt` +
  frozen `bc_net: {path: bc.pt}` before `bc.pt` exists): `colosseum record` exits 1 (Important 1).
- Focused tests of the area: 185 passed (agent kinds, players, match-runner players, eval players, validate
  players, retention, SP2 compat, fixed-player wiring, eviction, matchmaking/warm-start config).

## Strengths

- **One legality gate, honoured everywhere.** `check_bot_action` (structure, kind, shape, `space.contains`,
  `first_illegal_action`) is the only path for scripted players, DAgger teachers and `validate`; bots get copies of
  obs and mask (T1.3 ruling), so in-place edits can neither bypass the gate nor corrupt records.
- **"Only latest of a trainable agent collects" is enforced at three layers**: `league.lineups.check_lineup`
  (every matchmaker lineup, built-in or custom), `MatchRunner._check_lineup` (any lineup the runner sees, including
  a `ScriptedPlayer` served under `latest`), and `MatchRunner._resolve` (the SP1 missing-snapshot fallback refuses
  to turn a scripted player into a collecting seat). I found no path that seats a fixed player with `collect=True`.
- **Seat → player routing is keyed by `(agent_id, network_id)` consistently**: `RolloutLoop.get` serves fixed
  players only under `FIXED_NETWORK_ID` and trainable models only by their own ids, so a trainable agent at
  `fixed` or a frozen agent at `latest`/`ckpt_v*` fails loudly in `_check_lineup`. Frozen models of other
  architectures are batched under `(agent, "fixed")`, bots are instantiated per `(agent, env, seat)` and reset
  with `bot_rng(episode seed, seat, agent)`.
- **Eviction is ordered correctly end to end.** `CheckpointManager._retain` updates the index before
  `on_evict`; the coordinator forgets PFSP stats and buffers the id; `_refresh_worker_matches` takes evictions
  *before* generating lineups (so a command never pairs a lineup with the eviction of a snapshot it uses), retries
  per worker after a full queue, and only forgets `sent` after a successful put; `_drain_commands` merges both
  `new_checkpoints` and `evict`; the worker unloads only when no current or staged lineup uses the id. I traced
  the merged-command, failed-put and replaced-staged-lineup cases and found no stale model or premature unload.
- **Retention and pool import** match spec block 4: newest `keep_last` (with `trainer_state.pt`),
  `version % (interval * keep_every) == 0`, `final: true`, the just-saved snapshot; import hard-links only
  `model.pt` / `meta.json`, reads the source strictly and read-only, checks every snapshot's role signature,
  and never edits linked files in place (only `trainer_state.pt`, never linked, is unlinked). The learner's
  `pv % interval == 0` trigger keeps `keep_every` aligned across resumes.
- **SP2 compatibility is real, not only for the example copies**: SP2-written `config.resolved.yaml` files
  load too (experiment above); translations are pure functions over the raw dict, rounded (P3), rejected next to
  their replacements, never stored, and `--set` reaches them through `LEGACY_OVERRIDE_KEYS`.
- **Errors carry the SP2 context plus the agent** (`"worker W, env E, seat P, episode step K, layout L: agent 'X'"`)
  for bot `act`, `reset` and construction failures; `cli._config_errors` turns `PlayerError` into one line (P17).
- Test coverage of the area is broad and mostly sharp (negative cases for collect, missing players, bot
  failures, illegal Units actions, eviction retry, raising eviction callbacks, signature mismatches on import).

## Issues

### Critical

None found.

### Important

**I-1. `record`, `bc` and `eval` validate warm-start sources and fixed agents they never use, so the
documented competition pipeline cannot run from one config.**
- Where: `src/colosseum/cli.py:271` (`record_cmd`: `validate_config(cfg)`), `cli.py:329-330` (`bc`), `cli.py:200-201`
  (`eval_cmd`); `src/colosseum/core/validation.py:567` (`load_fixed_players` reads every frozen agent's path) and
  `validation.py:592-599` (`resolve_init` for every trainable agent), `validation.py:591`
  (`_check_kickstart_teachers` resolves every teacher). `record` has no `--set` to blank them.
- What: spec §1 promises the pipeline "настраивается в одном конфиге" and block 6's recipe is
  `record` → `bc` → `init: {from: bc.pt}` → `kickstart: {teacher: bc_net}` with `bc_net: {kind: frozen, path: bc.pt}`
  (spec block 1 shows exactly this config). Steps 1 and 2 run before `bc.pt` exists, but both commands run the full
  `validate_config`, which loads every frozen agent and resolves every `init.from` / teacher.
- Failure scenario (reproduced): `unit_harvest_league.yaml` + `agents.main.init.from: bc.pt` + frozen
  `bc_net: {path: bc.pt}` → `colosseum record -c one.yaml --player greedy ...` → `Config error: agents.bc_net.path=
  '.../bc.pt': expected a checkpoint dir ... or a .pt state_dict file`, exit 1. The same config fails `bc`.
  After training, `eval` of the run with the same config fails as soon as `bc.pt` (or any `init.from` source) is
  moved. The README/LEAGUE_GUIDE recipe works only because it injects `init.from` / `kickstart.teacher` with
  `train --set` and never declares `bc_net` (the pipeline test writes two configs, `tests/pipeline_kit.py:201,215`).
- Why it matters: the headline SP3 flow breaks for a user who follows spec block 1/6 literally; the error points at
  the frozen agent, not at "this command does not need it". Spec principle 1 ("удобный конфиг — первый критерий").
- Fix (small): give `validate_config` a scope, e.g. `validate_config(cfg, warm_start=True, fixed=None)`;
  `record` / `bc` / `eval` pass `warm_start=False` (skip `resolve_init`, `_check_kickstart_teachers`,
  `_check_critic_warmup`) and load only the fixed agents the command seats (`--player` / `--against` / `-a` names);
  `train` and `validate` keep the full check. Alternatively (or additionally) add `--set` to `record`, `bc` and
  `eval`. Add a CLI test: one config with a frozen agent and `init.from` pointing at a not-yet-existing `bc.pt`
  runs `record` and `bc`; LEAGUE_GUIDE §8 then shows the one-config form.

### Minor

- **M-1 `play_lineups` collect trap still open** (`src/colosseum/eval.py:243-296`): lineups are played as given,
  so `SeatAssignment("bot")` (default `collect=True`) for a `ScriptedPlayer` raises `ValueError` in
  `MatchRunner._check_lineup`. The T6.2 ruling schedules "force `collect=False` on every seat" for this fix wave;
  `tests/learning/test_sp3_fast_learning.py:70,77` still pass `collect=False` by hand. Fix: rebuild each lineup with
  `replace(seat, collect=False)` in `play_lineups` and add a test with a default `SeatAssignment` for a bot.
- **M-2 Unused frozen agents are built on every worker** (`launcher.py:621` passes `setup.fixed` whole;
  `rollout_loop.py:177-178`): a frozen agent used only as kickstart teacher or `init.from` (or with
  `anchors: []`) is still deserialized and built in every worker process. Cost = memory × workers for large teachers.
  Fix later: pass only fixed agents that can be seated (anchors with positive weight at some schedule point, plus a
  custom matchmaker's whole set).
- **M-3 `load_player` applies a frozen entry's `.pt` overrides to a user-given path**
  (`eval.py:663-672` → `load_frozen`): `-a bc_net=runs/x/ckpt_v100` where `bc_net` is a frozen `.pt` agent with
  `networks:` set fails with "remove agents.bc_net.networks / roles (they are for .pt files)", which is about the
  config entry, not the CLI argument. Edge case; either ignore the entry for `name=path` or say "the name collides
  with a frozen agent of the config".
- **M-4 Dead SP2 aliases**: `core/config.py:798` `AgentOverride = TrainableAgent` and
  `worker/match_runner.py:94` `ModelPool = PlayerPool` (+ `__all__`) have no caller in `src/`, `tests/` or
  `scripts/` (tests use their own `DictModelPool`). The SP3 compatibility promise covers configs and checkpoints,
  not these names; delete them.
- **M-5 Stale wording in `core/config.py`**: module docstring header still says "(config v2, SP2)";
  `get_agent_config` (`config.py:1000-1005`) says "deep-merged" (matchmaking uses `merge_matchmaking`) and its
  ConfigError lists only "networks / algorithm / learner"; the `agents` field description (`config.py:893-897`)
  omits `matchmaking`, `init`, `kickstart` (ledger T3.1 deferred).
- **M-6 `--set` into an anchor map or a schedule** (`config.py:1116`): `--set matchmaking.anchors.random=0.2` or
  `...opponents.latest.0=0.5` gives "'matchmaking.anchors' is not a section" without saying the whole-value form
  works (`--set 'matchmaking.anchors={random: 0.2}'`). One-line hint (ledger T3.1 deferred).
- **M-7 Unbounded `Coordinator._evicted` with `match_refresh_interval_sec: 0`** (`coordinator.py:140-145`, only
  drained in `launcher._refresh_worker_matches`): grows by one id per eviction for the whole run (ledger T2.2).
  Fix: the coordinator buffers only when the launcher will drain (a flag), or the monitor loop calls
  `take_evictions()` and drops the result when refresh is off.
- **M-8 `_notify_evicted` drops later callback errors silently** (`checkpoint_manager.py:313-325`): only the first
  error is raised; the others are not logged (ledger T2.2). Log each with `logger.exception`.
- **M-9 Import log counts skipped ids** (`checkpoint_manager.py:371`): "Imported N snapshots" uses `len(infos)`,
  including ids already present (ledger T2.1). Count the linked ones.
- **M-10 `check_bot_action` accepts `bool` for integer components** (`players/scripted.py:88`, `"iub"`): `True`
  is cast to 1 silently (ledger T1.2). Harmless for legality (the gate still runs), but a bot bug goes unnoticed.

## Deferred minors to fix now (ledger references)

Fix in the single final fix wave (cheap, and two of them are already promised to this wave by rulings):
1. **T6.2 ruling** — `play_lineups` forces `collect=False` (M-1). Promised to this wave.
2. **T1.5 ruling (Important, deferred to the final wave)** — extract `_reset_layout` / `_step_env` helpers shared by
   `_exercise_env` and `_check_scripted_players` (`core/validation.py:139-163` vs `:377-439`); same error wording.
3. **T3.1** — the `--set` hint for anchor maps / schedules (M-6) and the stale `get_agent_config` / `agents`
   field wording (M-5).
4. **T2.2** — log every `on_evict` callback error (M-8); stop `Coordinator._evicted` growing without refresh (M-7).
5. **T2.1** — "Imported N snapshots" counts only linked snapshots (M-9); optionally skip the
   "snapshot pool carried over: []" line for agents absent from the old run.
6. Not a ledger item, same wave: delete `AgentOverride` / `ModelPool` (M-4).

Keep parked (performance or test polish, no behaviour risk): T1.1 (`CoreConfig.model_config`, StrictModel
docstring, weak `match="agents"`), T1.2 (double meta read, `FrozenSpec` eq, missing negative tests; bool cast M-10
can stay parked), T1.3 (uint8 bot test, infos by reference already documented), T1.4/T1.5 (roles resolved twice,
frozen agents loaded twice by eval, `ObsSpec` per bot decision in validate, validate plays bots only in enabled
layouts), T0.1 (fixture helper polish), T2.1 (hard-link proof test, console-only `pool_size` warning).

## Recommendations

- Fix I-1 by scoping validation per command rather than by adding `--set` alone: the pipeline should work with the
  config the spec shows. Re-run `tests/integration/test_sp3_pipeline_smoke.py` and `test_sp3_bc_record.py` after.
- Add one regression test for the run-dir resume + eviction interplay at the launcher level: import a pool larger
  than `keep_last`, check that the first refresh's `evict` lists only ids never sent and that no lineup draws an
  evicted id (today covered piecewise by unit tests).
- LEAGUE_GUIDE could note that every run's final snapshot is kept forever and is imported by every run-dir
  resume, so frequent stop/resume cycles grow the snapshot pool (and the `snapshots` candidates) by one per cycle —
  spec-conformant, but worth knowing for preemptible machines.

## Assessment

**With fixes.** No Critical issue in area A: seat routing, the collect rule, eviction, retention, pool import,
resume and the SP2 translations are correct on every path I traced, and SP2 configs (including SP2-written
resolved configs) and the SP2 checkpoint work. I-1 is a real usability defect on the headline pipeline (one config
for record → bc → train); fix it together with the ruling-promised items M-1 and the T1.5 helper extraction in the
final wave.

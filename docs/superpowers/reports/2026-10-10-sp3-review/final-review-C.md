> Origin: `.superpowers/sdd/2026-10-10-sp3-league/final-review-C.md` (SP3 SDD workspace, git-ignored, deleted after the merge). Copied 2026-10-10.
> Frozen record; do not edit. Copied verbatim (no absolute scratch paths to replace).

# SP3 final review — area C (warm start, training data path, record/BC, tests/docs/benchmarks)

Reviewer: area C, branch `sp3-league` at `c31eda9` (range `c16fe5e..HEAD`). Ledger tasks T4.1–T4.5, T5.1–T5.2, T6.2, T6.4–T6.6.

What was read end to end: `learner/factory.py`, the `learner/learner.py` diff, `algorithms/appo.py` (warm-up, label loss, `teacher_active`), `algorithms/base.py`, `bc/kickstart.py`, `networks/model.py` + `networks/composed.py` (`value_parameters`), `core/types.py` (chunk teacher fields, payloads, `WeightPayload`), `worker/buffers.py`, the DAgger part of `worker/rollout_loop.py` (plus the `MatchRunner` hooks it relies on), `transport/serialization.py`, `record.py`, `bc/offline_bc.py::bc_data_sources`, the `record` / `bc` CLI commands, the launcher's warm-start wiring (`_learner_main`, `_resolve_init`, teacher resolution, worker teachers), `validation.py` warm-start checks, both demo bots, `tests/pipeline_kit.py`, `tests/learning/*` (SP3), the T4.x/T5.x contract and unit tests, `scripts/units_experiment.py`, `scripts/pipeline_vs_scratch.py`, and the area's docs.

Experiments (scratch under `.superpowers/sdd/2026-10-10-sp3-league/final-review/C/`):
- `dagger_units.py`: APPO's DAgger label loss with `Units` (K = 4) equals `per_sample_nll` over the labeled ACT slots (1.21522 vs 1.21522), `kickstart_label_frac` = 0.5 with every other slot labeled, finite gradients. The K>1 path works; only its test is missing.
- `warmup_norm.py`: a critic `NormalizeObs(source="global_state")` during a 2-step warm-up (Issue I-2).
- `colosseum record` with a config whose `init.from` names the not-yet-written BC file (Issue I-1).
- Focused tests (86, the warm-start / DAgger / init / factory / neural-teacher / demo-bot / fast-learning files) pass; test counts in CLAUDE.md match collection (1662 fast, 10 slow, 22 GPU).

## Strengths

- **Warm start is cleanly layered.** The main process resolves `init` and teachers into numpy dataclasses (`InitState`, `TeacherSpec`), and the learner builds the torch objects (`build_algorithm`, `apply_init`). The local and distributed learners share one factory, and nothing torch crosses a process boundary. `resolve_init` is careful: strict uses `check_model_state` with a hint, partial loads only tensors whose name and shape match, reports missing / mismatched / unexpected tensors, and errors when nothing matches. The student's role signature is checked the same way whether the source is a path or a frozen-agent name.
- **The critic warm-up is correct by construction.** Frozen parameters get `requires_grad_(False)` for the step and are restored in a `finally`, so they get no gradient and no Adam state. Normalizers are frozen, kickstart is off and its decay waits, the counter lives in the trainer state, and the LR schedule runs as usual. The tests check bit-identity of every non-value tensor, normalizer buffers included, and the Adam state set, on CPU, with GPU twins in `test_sp3_gpu_warmstart.py`.
- **The DAgger path is consistent from end to end:**
  - `BufferSpec.teacher` makes the label fields always present for such agents, and BOOT/PAD slots get zero labels.
  - `validate_slot_structure` rejects a label off an ACT slot.
  - The payload round-trip is backward compatible (`payload.get`).
  - `_stack_labels` fills in chunks that have no labels.
  - `_label_actions` substitutes the recorded (legal) action on unlabeled rows, so every log-prob is finite.
  - The denominator counts only labeled ACT slots.
  - The teacher flag travels with the weights.
  - Teacher errors are `PlayerError` with the SP2 context.
  - The contract tests check per-seat instances, resets per episode, `infos`, rng reproducibility, the flag switching off, and learner/worker log-prob reproduction with labels in the chunks.
- **`record` is well built.** Each (env, seat) has its own buffer that is flushed whole at the episode end, so seat-episodes stay contiguous and `dones` are right. Writes are atomic. Discarded episodes are dropped. `record.json` carries the eval summary. `bc_data_sources` reads record dirs with a clear `DataError`.
- **The measurements are honest.** The units-defaults decision table checks out arithmetically (K=128: mean 0.187 → 0.223, SE 0.0459, 2·SE 0.092). Pipeline vs scratch is reported as "no gain on this game" rather than dressed up. Thresholds follow the pre-fixed rule, and the flake risk is recorded as a ruling.
- **The status docs are thorough.** CLAUDE.md test counts and command lines match the code, and README's `metrics.jsonl` table has `critic_warmup`, `kickstart_label_frac` and `opponents`. GPU_CHECKS lists all 22 GPU node ids.

## Issues

### Critical

None.

### Important

**I-1. `record` and `bc` validate the whole training config, so the spec's "one config for the whole pipeline" cannot be used before `bc.pt` exists.**
`src/colosseum/cli.py:271` (record) and `:330` (bc) call `validate_config(cfg)`. That resolves every agent's `init.from` and kickstart teacher (`validation.py:590-600`) and loads every frozen agent. Neither command has `--set`.
- **Scenario:** a config written as in spec block 1 / block 6 / LEAGUE_GUIDE §4:
  - `agents.main.init.from: bc.pt` (or `kickstart.teacher: bc_net`);
  - `bc_net: {kind: frozen, path: bc.pt}`.

  Step 1 of the pipeline, `colosseum record -c cfg.yaml --player greedy ...`, exits 1 with `Config error: agent 'main': init.from='…/bc.pt' is neither a frozen agent of the config nor a .pt file, …`. This is reproduced. `colosseum bc` fails the same way, and it is the very command that would create `bc.pt`.
- **Why it matters:**
  - Spec §1 says the pipeline "настраивается в одном конфиге", and the spec's own example config shows `init.from` and a frozen `bc_net` next to the bot.
  - The documented recipe (README, LEAGUE_GUIDE §8) works only because it adds `init` / `kickstart` with `--set` at `train`.
  - The tests use two configs (`pipeline_kit.run_pipeline` writes `record.yaml` without `bc_net`), so nothing exercises the one-config form.
  - The error is loud, but it blocks the headline workflow and nothing in the docs explains it.
- **Fix (any of these):**
  - (a) Give `validate_config` a scope, so that `record` / `bc` skip `init` / kickstart resolution and frozen agents they do not seat. `record` needs only the players named on the command line; `bc` needs only the `--agent`'s model.
  - (b) Add `--set` to `record` and `bc`, which also helps `--seed`-style overrides.
  - (c) At minimum, document in LEAGUE_GUIDE §8 / §15 and README that `init.from` / `kickstart.teacher` / a frozen agent pointing at the BC output must be added only for `train` (`--set` or a second config).

  (a) + (c) is the smallest change that honours the spec.

**I-2. The critic warm-up freezes the critic's own `global_state` normalizer, so the warmed-up critic is invalidated on the first step after the warm-up.**
`src/colosseum/algorithms/appo.py:536` skips `_update_normalizers` for the whole step, including `NormalizeObs(source="global_state")` modules. Those live only on the value path (the contract: `global_state` reaches only the value path), so updating them cannot change the policy.
- **Scenario (`warmup_norm.py`):** `ComposedModel` with a critic encoder that normalizes `global_state` (values ≈ 3 ± 0.4), `critic_warmup_steps: 2`.
  - During the warm-up the critic statistics stay at the defaults (mean 0, var 1, count 1e-4), so the critic learns V from inputs ≈ 3.
  - The first non-warm-up step updates them to mean ≈ 2.9, var ≈ 0.2 (count 28), so the same inputs reach the critic as ≈ 0 ± 1.
  - The value function the warm-up trained is discarded at exactly the step where the policy loss turns on, and V-trace advantages from that critic drive the first policy updates of the BC policy. That is the failure the warm-up exists to prevent.
  - Typical case: BC trains `obs` normalizers but never sees `global_state`, so after `bc` the critic's normalizer is always at its defaults.
- **Status:** spec §8 note 7 says "не обновляется и статистика нормализаторов", but its rationale (bit-identical policy) covers only the policy path. This needs a short owner ruling rather than a silent change.
- **Fix:** during the warm-up call `self._model.update_normalizers(None, gs)` (with `obs=None`, every `obs`-source normalizer is skipped by `PolicyModel.update_normalizers`), i.e. update only the value-path normalizers. Then:
  - extend `test_warmup_trains_only_the_value_path_and_keeps_the_workers_policy_bit_identical` to expect `critic_encoder.norm.rms.*` to change while every `obs`-path buffer stays bit-identical;
  - update the `InitConfig.critic_warmup_steps` description, README and LEAGUE_GUIDE ("статистика нормализаторов пути политики").

  Normalizer `count` is a buffer, not a value parameter, so the existing `keys` check needs the critic-normalizer buffers added.

**I-3. LEAGUE_GUIDE says translated SP2 configs behave the same, but spec §8 lists behaviour changes.**
- **Where:**
  - `docs/LEAGUE_GUIDE.md:34`: "Примеры из `configs/examples/` … это точный перевод их формы SP2 …, поведение не изменилось. Два отличия от SP2 — в разделе 16."
  - `:290` §16: "с двумя отличиями", which names only the single-agent `mode: league` case and the `.pth` teacher.
- **What changes**, per spec §8 «Уточнения при написании spec», note 2, and §8 «Изменения правил SP2»:
  - (a) SP2 configs without explicit knobs now get the 0.7 / 0.2 / 0 / 0.1 mix instead of 0.5 / 0.5;
  - (b) in translated configs snapshots are drawn by PFSP, not uniformly (every example with `snapshots > 0`, e.g. `tic_tac_toe.yaml`, `predator_prey.yaml`);
  - (c) in asymmetric games the opponent can now be a snapshot of the other agent: `predator_prey.yaml` translates to `snapshots: 0.2`, where in SP2 the prey was always latest;
  - (d) multi-agent `mode: self_play` now meets other agents' snapshots;
  - (e) `keep_every: 10` by default keeps every 10th snapshot forever (disk grows; example configs set no `keep_every`).
- **Why it matters:** this misleads about the real state (criterion 3.8). A user comparing an SP2 run with an SP3 run of the same config is told nothing changed, and the SP2 slow tests' dynamics did change (spec risk table, last row).
- **Fix:** rewrite the sentence at :34 as "the config translation is exact; the behaviour differs as listed in §16", and list (a)–(e) in §16 next to the two existing items.

### Minor

1. **`kickstart.kl: reverse` with a scripted teacher is silently ignored** (`learner/factory.py:162`; the field description mentions it). Spec block 6 says `kl` is for a neural teacher only, and spec §4 says errors are loud. Make it a `ConfigError` (or one warning) in `resolve_teacher` when `kind == "scripted"` and `kl != "forward"`.
2. **A resume from a snapshot without `trainer_state.pt` repeats the critic warm-up and restarts kickstart λ.** This covers a `keep_every` or imported snapshot, an SP2 checkpoint, or a `.pt`. `learner.py:105-108` loads a fresh `algorithm.state_dict()` and `appo.py:622` defaults `critic_warmup_done` to 0. The resumed, already trained policy is then frozen for another N steps. This is intentional per the code comment, and the kickstart part is SP2 behaviour, but it is not documented: add one line to LEAGUE_GUIDE §8 (resume) and the README resume paragraph.
3. **README.md:78 timing:** "BC about 17 s" contradicts `docs/benchmarks.md` and `pipeline_kit.py` (bc 9.1–9.2 s; record 17 s). It is probably a copy of the record figure. Fix the number or say what was measured.
4. **Wasted DAgger queries during the warm-up.** `APPO.teacher_active` is True while warming up (λ > 0), so workers query the scripted teacher for labels the learner ignores (`appo.py:430`, `not warmup`). This costs time only, but on slow Python bots it is the whole warm-up's worth of bot calls. Optional: `teacher_active = … and not self.critic_warming_up`, because chunks without labels are already handled. The label fraction then rises one weight-sync later, which is harmless. This also makes the deferred "`kickstart_label_frac` reads 0 during warm-up" moot.
5. **`bc_data_sources` checks nothing from `record.json` beyond `roles`.** It does not compare `game` / `env_kwargs` with the `bc` config. Data recorded with other `env.kwargs` but the same spaces (e.g. another map size with a fixed observation) trains silently. A one-line WARNING when `record.json["game"]` or `["env_kwargs"]` differs from the config would make this visible.
6. **`pipeline_kit.py` has three blank lines before `def evaluate`** (also deferred in T6.2), and `demo_learning.py` the same. Cosmetic.

## Deferred minors to fix now (ledger references)

Recommended for the single fix wave (cheap, and on paths this area owns):
- **T6.4 / T6.2 "`colosseum bc` has no `--seed`"**: seed `torch` / the minibatch order from `--seed`. It makes the pipeline reproducible end to end (record and train already take a seed). It also pairs naturally with I-1 if `--set` is added to `bc`.
- **T4.5 "APPO-level DAgger test with K>1 missing"**: the path works (experiment above), but the `Units` + `_label_actions` tree substitution + `reduction="mean_valid"` combination has no test through `APPO.compute_loss`. The scratch script `final-review/C/dagger_units.py` is almost the test: assert `kickstart_loss == per_sample_nll[labeled].mean()` and finite gradients.
- **T4.5 "dead fallback `_episode_layout[env] or lineup(env).layout`"** (`rollout_loop.py:438`): `on_episode_start` always runs first, also for the constructor's initial reset, so the fallback only hides a broken hook. Replace it with the plain `self._episode_layout[env]` (or assert it is not None).
- **T4.3 "`_check_critic_warmup` lets `import_class` errors escape raw"**: wrap them as `ConfigError` (one `try`), in line with "every config error is a ConfigError".
- **T6.2 ruling "`play_lineups` forces `collect=False`"**: already scheduled for this wave. `record` is not affected (`schedule_lineups` builds `collect=False` seats), but in-process callers are.
- **T6.6 "record.py module docstring less precise than LEAGUE_GUIDE about skipped layouts"**: one sentence (P8: default layouts the player cannot fill alone are skipped with one INFO line).

Keep parked: T4.1 (test name in `test_sp2_validate.py:159`), T4.2 (duplicate checkpoint discovery; teachers/init resolved twice), T4.4 (spy for the recurrent-teacher `state0`; loose "KL" match), T5.1 (`_parse_player` duplication; partial part files after a failure; `_schedule` called privately; no neural `Units` recording test), T5.2 (rounding predicate duplication), T6.5 (n=2 decisions, raw `--cells` tracebacks, timing line), the rest of T4.5 (`validate_chunk_payload` trust class → SP5; RolloutLoop/MatchRunner bot-handling duplication).

## Recommendations

- I-1 and I-2 touch user-facing semantics. Record the chosen fixes as rulings: I-1 "record/bc validate only what they use", I-2 "the warm-up freezes only policy-path normalizer statistics, revising spec §8 note 7". Re-run `test_sp3_pipeline_smoke.py` and the critic warm-up tests afterwards.
- For I-1, add a fast test: one config with `init.from` / a frozen `bc_net` pointing at a missing `bc.pt`; then `record` → `bc` → `validate` succeeds once `bc.pt` exists.
- The slow `unit_harvest` pipeline test cannot tell a working warm start from none (ruling T6.4 addendum). In SP4/SP6, consider a harder calibration game where scratch training does not reach the bot within the budget. Until then the warm start's value rests on the fast and contract tests, which are solid.

## Assessment

**Ready to merge: With fixes.** No critical defects: the warm-start, DAgger and record/BC data paths are correct and well tested. Fix I-1 (the pipeline's one-config form) and I-3 (the compatibility doc) in the fix wave. I-2 needs a one-line owner ruling and a one-line code change.

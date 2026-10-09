> Origin: `.superpowers/sdd/2026-10-08-sp2-game-model/final-review-B.md` (SP2 SDD workspace, git-ignored, deleted after the merge). Copied 2026-10-09.
> Frozen record; do not edit. Only absolute scratch paths were replaced with `<scratch>/`.

# SP2 final whole-branch review — AREA B: the training data path

Reviewer B. Branch `sp2-game-model` at `7d55974`, merge base `7430361`.
Scope: `worker/{buffers,match_runner,rollout_loop,rollout_worker}.py`, `algorithms/{vtrace,appo,base}.py`,
`learner/learner.py`, `bc/{kickstart,offline_bc}.py`, `core/ipc.py`, `transport/serialization.py`, and their
unit / contract / integration / learning tests.

Method: read every file in scope end to end against spec blocks 1, 4, 5, 6, overview contract (T3.2–T4.4),
amendments R9/R10/R15/R16, PR-1 and the ledger rulings. Traced every event order
(`_infer → on_act → vec_env.step → on_rewards → on_terminated → on_episode_end → on_lineup_applied → reset`)
by hand on the cases in the brief. Experiments (scratch dir `<scratch>/final-review/B/`):

1. `cmp_vtrace.py`: `compute_vtrace_slots` on ACT-only chunks ending in BOOT (or PAD after a terminal ACT),
   with random terminals, ρ̄, c̄, λ, γ and off-policy log-ratios, against SP1's IMPALA `compute_vtrace` from
   the merge base (dones = terminal, bootstrap = V(BOOT)). 200 random cases: **max |Δ| = 0.0** for `vs` and
   for `td · clipped_rho` vs SP1's advantages.
2. `park_repro.py`: 3 envs, 2 seats, random lineup commands every 3 steps that toggle `collect` per seat (so
   buffers are parked and re-acquired across envs; 58 of 296 chunks mix envs), turn-based waiting rewards,
   truncation; LSTM / GRU / attention cores. Learner re-evaluation of every chunk after the payload round
   trip reproduces the worker's log-probs: max |Δ| ≤ 3e-7 for all three cores.
3. `torch.where` backward check (see Minor 1).
4. The area's fast tests (`test_slot_buffers`, `test_vtrace_slots`, `test_appo_v2`, `test_match_runner`,
   `test_learner_v2`, `test_kickstart_v2`, `test_chunk_v2_types`, `test_chunk_v2_rules`,
   `test_learner_reproduces_worker`, `test_rollout_loop_lineups`, `test_sp2_bc_trainer`): 258 passed.

## Strengths

- **Slot writer is exactly spec block 4.** `RolloutLoop.on_act` (rollout_loop.py:276-292) implements
  rules 1–5 literally: BOOT + seal + ACT into the *same* reset buffer when an open ACT meets one free slot,
  PAD + seal otherwise, ACT never in the last slot. The chunk-boundary BOOT carries the next ACT's
  observation and `global_state`, and `begin(record.pre_state, …)` gives the next chunk the state before
  that ACT, so `V(BOOT)` in chunk *n* and `V(ACT)` in chunk *n+1* are the same quantity under the
  learner's network. Parking is only possible at an episode boundary (`BufferPool.park` refuses
  otherwise), and a continuing episode never lands in a parked buffer.
- **Rewards land on the right ACT.** `on_rewards` runs after `vec_env.step` and before the next `on_act`,
  so `r_t` of the step after a decision goes to that decision; rewards of waiting seats accumulate on
  their open ACT; rewards before the first ACT of an episode go into it as `pending`; the elimination
  step's reward is applied before `mark_terminal` (MatchRunner orders `on_rewards` before
  `on_terminated`); a seat eliminated in the truncation step gets `terminal`, not a BOOT (R15); a dead
  teammate keeps its open ACT and every later reward until the end (terminal, or BOOT(final_obs) on
  truncation). The contract test `test_reward_accounting_matches_the_env_except_documented_drops` pins
  the full reward balance.
- **V-trace over slots is the IMPALA math.** Non-ACT slots have ρ = c = 0 and `vs = V`, traces stop at
  BOOT, terminal ACTs bootstrap 0, all selection is `torch.where`; experiment 1 shows bit-identical
  results to SP1 on ACT-only chunks, and `test_zero_lag_equals_gae_over_act_slots` pins GAE(λ) at zero lag.
- **Every value is the learner's.** The worker no longer computes values; `APPO._evaluate` unrolls the
  current network over every slot (BOOTs included) and V-trace consumes `ev["values"].detach()`.
  `test_bootstrap_uses_the_learners_current_values` and `test_value_targets_bootstrap_from_the_learners_boot_value`
  pin it.
- **APPO v2 modes match the spec and R9.** `ratio_mode`, `unit_trace`, `entropy_reduction` resolve as
  specified (auto `unit_trace` = `joint` per the T8.4 ruling); `per_unit` applies PPO-clip per decider on
  the shared advantage without the ρ factor, mean over valid deciders, then over ACT slots; `joint`
  multiplies by the clipped scalar ρ (SP1 parity); the K == 1 collapse is exactly R9/PR-1. Every loss
  and diagnostic reduces over ACT slots (`_act_mean`, `valid = unit_valid & is_act`). Per-decider
  behaviour log-probs (`behavior_unit_logp [S, K]`) are recorded from the same distribution and action
  as the learner's `unit_log_prob`, and the contract test reproduces them under random unit masks for
  all four cores.
- **`validate_slot_structure` is complete for what V-trace relies on**: ACT never last, open ACT followed
  only by ACT or BOOT, BOOT only after an open ACT, PAD only after an episode end, non-reset BOOT only
  last, and flag/kind consistency. Every grammar violation that could corrupt targets or recurrent
  resets is rejected with the agent, version and the slot letters.
- **State handling does not leak.** Per-seat model states are re-initialised for the new lineup at every
  episode end (after `on_lineup_applied`), the observation rows are copied before inference, the batched
  state is rebuilt with `cat_batch` (copies) every step, and `begin()` clones the pre-state. Experiment 2
  confirms continuity across parking for every stateful core.
- Strong tests: the slot-rule contract tests decode every slot back to `(env, episode, step, seat, final)`,
  so they cannot pass vacuously; the reproduce test asserts that T, B, R and P slots, an eliminated seat's
  terminal ACT and a dead teammate's truncation BOOT are all present before comparing.

## Issues

### Critical

None.

### Important

None.

### Minor

1. **`torch.where` protects the forward pass only** — appo.py:117-119 (`_act_mean`), appo.py:362
   (`value_loss`), appo.py:359 and 363. If the model ever outputs NaN on a non-ACT slot, the forward
   losses stay finite but the backward is NaN: `where(is_act, (v - vs)**2, 0)` gives the unselected
   branch a zero upstream gradient, and `0 · d/dv (v - vs)**2 = 0 · NaN = NaN` reaches the parameters
   (checked: `w.grad = [1., nan]`). Spec block 4 only promises "NaN does not reach the targets" and
   relies on PAD copying the previous observation, so no supported path produces NaN today (BOOT masks
   are "all allowed", PAD repeats a real observation). No fix needed; if a NaN-safe backward is ever
   wanted, sanitise `ev["values"]` / per-decider tensors once with `torch.where(is_act, x, 0)` *before*
   any arithmetic.

2. **Rewards of a seat that never acts in an episode are invisible at the default log level** —
   rollout_loop.py:321-327. See the dedicated judgement below. Additional details: the counter counts
   seat-episodes, not episodes (a 4p episode with two such seats adds 2), despite its name; a seat whose
   dropped rewards sum to exactly 0.0 is not counted. The counter already reaches the main process in
   every `worker_stats` message (rollout_worker.py:60-64, 154), but `SystemStats` keeps only
   `parked_buffers` (metrics/aggregator.py:170), so it never appears in `metrics.jsonl`.
   Fix (final wave, ~10 lines): log a WARNING the first time it happens per worker with the global context
   (`"worker i, env e, seat s, layout L: dropped reward R of a seat that never acted in the episode
   (further drops only counted)"`); optionally sum `dropped_reward_episodes` into the `system` record next
   to `parked_buffers` and update ENV_GUIDE.md:146-150 accordingly.

3. **BC docstring overstates skipping** — offline_bc.py:18-19 says "Decisions without a valid decider are
   skipped", but `per_sample_nll` (offline_bc.py:83-85) marks every K == 1 decision valid. With
   `Units(1, …)` (K = 1) a decision whose single unit is absent gets NLL 0 and weight 1: no gradient,
   but it dilutes the reported `bc_loss`. Fix: for K == 1 use `dist.unit_valid(actions)[:, 0]` as the
   validity, or reword the docstring ("with K > 1").

4. **The learner does not check that a chunk belongs to its agent** — learner.py:356-358. Local mode routes
   by `chunk.agent_id` per queue, so it is correct by construction; in distributed mode a misconfigured
   `run-workers -l A=host:port` (pointing agent A at B's learner) would silently train B on A's data when
   the spaces match. Fix (cheap, or SP5): `collect_batch(..., agent_id=...)` raising `ValueError` on a
   foreign `chunk.agent_id`.

5. **MatchRunner trusts that every seat of a lineup is a role its agent plays** — match_runner.py:264-276,
   203-224. The observation batch of an inference group is allocated from the first seat's role; a seat
   of another role with a smaller observation would be broadcast silently by `put_row` (e.g. shape `(1,)`
   into `(5,)`). The matchmaker, eval and distributed mode never build such lineups (their invariants are
   tested elsewhere), so this is defence in depth only. Fix (optional): `RolloutLoop` knows
   `agent_roles`; check `spec.role_of(layout, seat) in agent_roles[agent]` in `_start_collecting` /
   initial lineups, or give `MatchRunner` an optional roles map.

6. **`validate_slot_structure` ignores payload values on the slots it validates** — types.py:176-232. A
   non-zero reward on a BOOT/PAD would be dropped silently by V-trace (`where`), and a NaN
   `behavior_logp` on an ACT would make the whole minibatch NaN. The local worker cannot produce either;
   relevant only to a hostile/broken distributed peer (SP5 wire-format work). Park with SP5.

## Judgement: rewards of a seat that never acts in an episode (`dropped_reward_episodes`)

**Not a correctness problem for any supported game type; no code change is required for correctness.**

- Training credit: a reward can only be credited to a decision of the same seat in the same episode.
  A seat with no ACT in the episode has no decision to credit; carrying the reward into the next
  episode's first ACT would be a cross-episode bias (worse than dropping). Lineups only change at
  episode ends, and the chunk-boundary BOOT is always followed in the same `on_act` by the seat's next
  ACT, so no other path leaves a seat without an open ACT while it is alive and has acted.
- Ratings and outcomes: `MatchResult` returns and the default team score come from `EpisodeTracker`,
  which counts every reward, including dropped ones, so eval, ELO and `ScoreTracker` are unaffected.
- Cases: a seat eliminated before its first turn (turn-based FFA) loses only its death penalty, which no
  decision of that episode caused; a coop seat that never acts has no policy decisions to improve, and
  its teammates receive their own per-seat rewards. None of the seven demo games has such seats
  (simultaneous games act from the reset step; `tic_tac_toe`'s second player always acts before the
  game can end).
- The real risk is an env bug (a seat missing from `acting`) going unnoticed; ENV_GUIDE documents the
  trap honestly. A warn-once (Minor 2) is the right, cheap mitigation and is recommended for the final fix
  wave (it is the candidate the T8.6 ruling parked), but it is not a merge blocker.

## Deferred minors to fix now (ledger references)

- **T8.6 ruling (dropped-reward visibility, parked as a final-review candidate)** — fix now as Minor 2
  (warn-once with context; optional `system` metric). Cheap, and it turns the only silent data loss in
  the pipeline into a visible signal.
- Everything else in the ledger for this area stays parked:
  - T3.1 (payload key count not pinned; `_apply` on `None` leaves; `_map_optional`) — no observable defect.
  - T3.3 "MatchRunner `__init__` does not close the vec env on reset failure" — both callers
    (`RolloutLoop.__init__`, `eval.play_matches`) already close it in their `except`, so there is no leak.
  - T3.3 ActRecord dtype cast — `put_row` casts on write; no effect on training.
  - T3.4 `_add_checkpoint` ignores unknown agents — the coordinator only sends known agents.
  - T4.1 shape guard / docstring; T4.2 clip metrics vs configured bars under `unit_trace: none`
    (diagnostic only); module-level R9 once flag; R12 GPU tests (need a CUDA machine: `docs/GPU_CHECKS.md`).
  - T6.3 (torch import in `specs.py`, BC traceback on bad device/output, absent-unit range checks).

## Recommendations

1. Final fix wave: Minor 2 (warn-once, optionally the `system` metric plus the ENV_GUIDE sentence) and the
   one-line docstring fix of Minor 3. Optionally Minor 4 (one `if` in `collect_batch`).
2. SP5 (wire format): Minor 6 (finite `behavior_logp`, zero non-ACT rewards, agent check at the server).
3. SP6 (already parked): normalizer update after the epochs; consider sanitising non-ACT outputs before
   arithmetic (Minor 1) if NaN-safe backward becomes a requirement.

## Assessment

**Ready to merge: Yes** for area B (training data path). No Critical or Important findings; the six Minors
are polish or defence in depth, and Minor 2 is recommended (not required) for the single final fix wave.

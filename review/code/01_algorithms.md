# R1 — Algorithms / Networks / Behavioral Cloning review

Scope: `algorithms/{base,vtrace,appo}.py`, `networks/{base,distributions,actor_critic,normalization}.py`, `bc/{offline_bc,kickstart}.py`, `core/action_spec.py`, `core/types.py`, `core/registry.py`, plus the learner/worker code that calls them.
Repro scripts: `scratchpad/r1_algos/t1_vtrace_ref.py`, `t2_dists.py`, `t3_recurrent_e2e.py`, `t4_misc.py` (torch 2.14.1+cpu).
The in-scope test files (`test_vtrace/appo/recurrent/action_masking/composite_actions/distributions/bc`) all pass (87 passed). They miss the critical bug in R1-01.

## Findings

### R1-01 — CRITICAL — bug — recurrent training crashes on chunks built by the real worker
- **Location**: `algorithms/appo.py:159-165` (stacks `lstm_hidden` with `dim=1`); `worker/rollout_worker.py:143-157, 449-451, 220` (hidden stored as `[L,1,H]`); `core/types.py:39-42` (docs say `[num_layers, hidden_size]`).
- **What is wrong**: the worker keeps per-slot hidden state as `new_hidden[0][:, j:j+1, :]`, which has shape `[L,1,H]`. It stores that unchanged as `chunk.lstm_hidden`. APPO calls `torch.stack(..., dim=1)`, which gives `[L,B,1,H]`, and nn.LSTM/GRU reject that.
- **Evidence (VERIFIED-BY-RUNNING, t3)**: I built the chunk with the real `_run_inference_group` + `RolloutBuffer.set_lstm_init` + `_build_chunk` and passed it to `APPO.train_step`:
  ```
  [lstm] worker-built chunk lstm_hidden[0].shape = (1, 1, 32)
  [lstm] APPO.train_step ... FAILED: RuntimeError: For batched 3-D input, hx and cx should also be 3-D but got (4-D, 4-D) tensors
  [gru]  ... FAILED: RuntimeError: For batched 3-D input, hx should also be 3-D but got 4-D tensor
  ```
  The tests do not catch this because they build chunks by hand with `h.squeeze(1)` (`tests/test_recurrent.py:231-233`). No test feeds worker output into a recurrent learner.
- **Impact**: every LSTM/GRU training run crashes the learner on its first batch, in both local and gRPC modes (serialization keeps the shape). The "LSTM/GRU support" feature does not work end to end.
- **Fix**:
  - Pick one layout. Either `set_lstm_init(h.squeeze(1), c.squeeze(1))` in the worker, or in APPO `torch.cat([c.lstm_hidden[0].reshape(L,1,H) ...], dim=1)`.
  - Assert the shape in `TrajectoryChunk` or the learner.
  - Add an integration test: worker loop, then chunk, then `APPO.train_step` with `recurrent_type: lstm`.
  - Use a `recurrent_hidden_size` different from `latent_dim` in the test helpers (see R1-02).

### R1-02 — HIGH — bug — `ActorCriticNetwork.forward` silently skips the RNN when `hidden is None`
- **Location**: `networks/actor_critic.py:69` (`if self.recurrent is not None and hidden is not None`). Callers that pass no hidden state: `eval.py:245` (`net.act(obs, ...)`), `bc/kickstart.py:84-90` and `bc/offline_bc.py:160-161` (call `encoder` then `policy` directly), and `appo.py:146-149` (takes the stateless path whenever `chunks[0].lstm_hidden is None`, even for a recurrent net).
- **Evidence (VERIFIED-BY-RUNNING, t3)**:
  ```
  kickstart on recurrent(latent!=hidden) FAILED: mat1 and mat2 shapes cannot be multiplied (5x16 and 32x4)
  kickstart on recurrent(latent==hidden): silently runs, RNN bypassed, loss = 0.038
  BC on recurrent(latent!=hidden) FAILED: ... (32x16 and 32x4)
  act() without hidden on recurrent net (eval.py path) FAILED: ... (3x16 and 32x4)
  ```
- **Impact**:
  - Eval crashes for recurrent agents, which blocks best-agent selection.
  - Kickstarting and offline BC crash for recurrent agents. When `latent_dim == recurrent_hidden_size` they instead run silently with the RNN removed, so the KL is computed on a different function than the one being trained.
  - The test helpers use `HIDDEN_SIZE == latent_dim`, which hides this.
  - It also limits warm starting: BC weights can't be trained for a recurrent policy.
- **Fix**:
  - In `forward`, if the net is recurrent and `hidden is None`, either raise or use `initial_hidden(B)` (the latter only for documented single-step use).
  - Kickstart: return the distribution from `evaluate_actions(_recurrent)` and compute the KL on it. Unroll the teacher over the same `[T,B]` sequence with its own hidden state.
  - BC: accept sequence data (episodes or chunks) and unroll.
  - eval.py: track hidden state per (env, slot) and reset it on done, like the worker does.

### R1-03 — HIGH — correctness — truncation is treated as termination (no bootstrap at time limits)
- **Location**: `worker/rollout_worker.py:490` (`done = terminated or truncated`) and `:531` (`bootstrap_val = 0.0 if done`); `algorithms/vtrace.py:47-48, 68` (the `not_done` mask removes `V(s_{t+1})`); `core/types.py` has only one `dones` field. `VectorEnv` already keeps `terminal_observation` (`envs/vec_env.py:139`), but nothing uses it.
- **Evidence (VERIFIED-BY-RUNNING, t1)**: reward 1 per step, γ=0.99, true value ≈100, episode truncated at the last step: `vs at last step = 1.0; adv = -99.0`.
- **Impact**: any env that reports a step limit as `truncated` (the gymnasium TimeLimit convention, and VectorEnv sets truncated if *any* player truncated) gets a biased value function and wrong advantages near the time limit. Episodes that end at a scored final turn are fine if the env reports `terminated`.
- **Fix**: store `terminated` and `truncated` separately, or store a per-step `discount`.
  - At a truncated step, compute `V(terminal_observation)` and add `γ·V(s_T)` to the reward (the SB3 approach).
  - Alternatively, ship a per-step bootstrap value and keep cutting the trace at the episode boundary.
  - V-trace then uses `discount_t = γ(1-terminated_t)` for the TD term and `(1-done_t)` for trace cutting.

### R1-04 — MEDIUM — correctness/performance — the chunk-end bootstrap comes from the stale behavior net; all other values come from the learner
- **Location**: `worker/rollout_worker.py:520-531` (an extra batch-1 forward per chunk on the worker's latest net); `appo.py:200` (`bootstrap_value=batch["bootstrap_values"]`) next to `values=new_values` from the learner.
- **What is wrong**: V-trace mixes `V_learner(s_0..s_{T-1})` with `V_behavior(s_T)`. The behavior value can be many updates old and uses the worker's obs-normalization stats and RNN state. Because of the c-trace, its error spreads into all `vs` of the chunk.
- **Evidence**: BY-READING. Sample Factory recomputes all values in the learner, as does seed_rl's IMPALA, which carries the `T+1`-th observation.
- **Impact**: biased value targets under policy lag, plus an extra unbatched forward pass per chunk on the CPU worker.
- **Fix**: add `bootstrap_obs` (and its mask/done) to `TrajectoryChunk`. The learner then evaluates `V(s_T)` in the same forward pass (for RNNs, one more unroll step), and the worker no longer does that forward.

### R1-05 — MEDIUM — bug/doc-mismatch — offline BC ignores action masks despite its docstring; Box actions are silently cast to `.long()`
- **Location**: `bc/offline_bc.py:35-37` (docstring promises mask support), `:72-97` (`add_data`/`load_data` ignore an `action_masks` key), `:156-174`; `cli.py:53` (`--action-type` defaults to `discrete`).
- **Evidence (VERIFIED-BY-RUNNING, t4)**:
  ```
  BC trainer stored masks? False ; add_data signature: ('self','observations','actions')
  BC Box actions with default action_type: loss 1.877 (continuous actions in [0,0.9) were .long()'d to 0)
  ```
  The continuous branch uses MSE on `mode()` (log_std is never trained). The composite branch uses NLL, so the two are inconsistent.
- **Impact**:
  - BC policies put probability on illegal actions, and that mass is renormalized differently once RL applies masks.
  - Box-action BC with the default CLI flag trains on garbage without any error.
- **Fix**:
  - Load and apply `action_masks` with `dist.apply_mask(mask)`.
  - Drop `action_type` and choose the loss from the distribution class: NLL everywhere, with MSE optional.
  - Raise if float actions are passed in discrete mode.
  - Report accuracy and a validation split.

### R1-06 — MEDIUM — design — BC leaves the value head random; RL then starts with garbage advantages
- **Location**: `bc/offline_bc.py:156-174` (only encoder+policy are trained); the BC data format is only `{observations, actions}`.
- **Impact**: after `resume_from: bc_weights`, the first APPO updates use advantages from a random critic. With `normalize_advantages=True` these are pure noise at unit scale, and they quickly erase the BC policy. Kickstarting reduces the damage but costs a full teacher forward.
- **Evidence**: BY-READING.
- **Fix**:
  - Allow `rewards`/`dones` in the BC data and fit the value head on Monte Carlo returns.
  - And/or add an APPO `critic_warmup_steps` option (policy loss weight 0 while the value head adapts).

### R1-07 — MEDIUM — bug — resume resets the LR schedule and the kickstart lambda; the scaler isn't saved
- **Location**: `learner/learner.py:66-77` (restores the net, the optimizer through the private `_optimizer`, and `_policy_version`, then calls `setup_lr_schedule` from step 0); `appo.py:62-73`; `bc/kickstart.py:53-62` (`_current_step` isn't saved); `checkpoint_manager` stores only model and optimizer.
- **Evidence (VERIFIED-BY-RUNNING, t4)**:
  ```
  before 'checkpoint': lr = 0.0005 kickstart lambda = 0.5
  after resume (learner.py path): lr = 0.001 kickstart lambda = 1.0
  ```
- **Impact**: when you resume a long league run, the LR jumps back to its initial value and the teacher KL comes back at full strength. That undoes the end of training, which matters for a warm-start workflow.
- **Fix**:
  - Add `BaseAlgorithm.state_dict()/load_state_dict()` covering optimizer, scheduler, GradScaler, kickstart step and policy_version, and save it in checkpoints.
  - Alternatively, compute LR and lambda as pure functions of `policy_version` or consumed samples.

### R1-08 — MEDIUM — correctness — obs-normalization stats are updated on every training forward (epochs × minibatches, plus kickstart), and train/eval behavior differs
- **Location**: `networks/normalization.py:73-76` (updates whenever `self.training`); the learner network is never switched out of `train()` mode (`network.training=True`); `appo.py:187-193` (forward per minibatch per epoch), `:231-233` (kickstart does a second student forward).
- **Evidence (VERIFIED-BY-RUNNING, t4)**, with 64 unique samples per train_step:
  ```
  epochs=1 mb=0: rms.count=64 | epochs=4 mb=1: rms.count=256 | kickstart: rms.count=128
  ```
- **What is wrong**:
  - The same data is counted several times.
  - The stats move *before* the current minibatch is normalized, so the learner's `π(a|s)` uses different normalization than the worker that produced `μ(a|s)`. That adds ratio noise and clipping.
  - User encoders with Dropout or BatchNorm also behave differently on the learner (train mode) than on workers (eval mode), which biases the ratio.
  - Positive point: the stats are buffers, so they ship with the weights and are saved in checkpoints.
- **Fix**:
  - Give the algorithm an explicit `update_obs_stats(batch_obs)` hook, called once per `train_step` on fresh data.
  - Run all loss forwards with frozen normalization.
  - Document that Dropout and BatchNorm are unsupported, or run loss forwards in eval mode.

### R1-09 — MEDIUM — performance — the recurrent cuDNN fast path is dead code; training always uses a per-timestep Python loop
- **Location**: `appo.py:174` always passes `dones_seq`, so `actor_critic.py:118-145` always runs `T` separate single-step RNN calls.
- **Evidence (VERIFIED-BY-RUNNING, t4, CPU)**: T=256, B=32, H=64, forward+backward: single pass 2191 ms, per-step loop 5792 ms (2.6×). On GPU the gap is usually much larger: 2×T small kernel launches against one fused cuDNN kernel.
- **Fix**:
  - Use the fast path when `not dones.any()`.
  - Otherwise split sequences at done boundaries into sub-episodes and run `pack_padded_sequence` (what Sample Factory does), or unroll only segments between the union of done indices.

### R1-10 — MEDIUM — design — policy lag is never measured or bounded
- **Location**: `behavior_policy_version` is only carried around (it appears nowhere in algorithms or learner); `appo.py:209-218` (PPO ratio against the behavior μ).
- **What is wrong**:
  - The PPO clip window is centred on the behavior policy. With lag, many samples start outside `[1-ε,1+ε]` and contribute no gradient.
  - `approx_kl` and `clip_fraction` mix lag with the update itself, so they can't be read.
  - Nothing drops chunks that are too stale (Sample Factory has `max_policy_lag`).
  - This matters most in the multi-machine setup the owner wants, where lag is largest.
- **Evidence**: BY-READING. On-policy sanity check (t4): `clip_fraction 0.0, approx_kl 0.0`, as expected.
- **Fix**:
  - Log `policy_lag = policy_version - behavior_policy_version` (mean and max) and `mean(ρ)` and `frac(ρ > ρ̄)`.
  - Add `max_policy_lag` filtering.
  - Optionally report KL against the learner-at-start-of-update policy.

### R1-11 — MEDIUM — correctness — variable batch size breaks schedules and the training-length budget
- **Location**: `learner/learner.py:166-193` (`_collect_chunks` returns whatever is queued, possibly one chunk); `:88` (stops at `total_train_steps`); `distributed.py:191-192` (`total_train_steps = total_timesteps // (chunk_length*batch_chunks)` assumes full batches).
- **Impact**: when the learner is faster than the workers (the usual case with a GPU learner and few CPU workers):
  - every update uses tiny, noisy batches;
  - the LR schedule and kickstart decay advance per update rather than per sample, so the LR reaches 0 early;
  - the learner stops after using only a fraction of `total_timesteps`.
- **Evidence**: BY-READING.
- **Fix**:
  - Wait for `min_batch_chunks` (with a timeout).
  - Drive the schedules and the stop condition by consumed env steps, passed into `train_step`.

### R1-12 — LOW/MEDIUM — doc-mismatch — "GAE" is advertised, but `gae_lambda` is dead config
- **Location**: `core/config.py:61`, README table at line 463, CLAUDE.md ("GAE for advantage estimation"), `core/types.py:33`. `gae_lambda` is referenced nowhere in `algorithms/`.
- **What the code does**: pure V-trace (λ=1 with c̄-truncated traces). The V-trace math itself is correct: it matches a slow reference implementation of eq. 1 in Espeholt et al. to within 4e-7 on random data with dones (t1). Bootstrapping at chunk end and masking at episode boundaries are both right.
- **Fix**: implement V-trace(λ) (`c_t = λ·min(c̄, ρ_t)`, per the IMPALA appendix) and rename the knob to `vtrace_lambda`, or delete the knob and the GAE claims.

### R1-13 — MEDIUM — bug — masked-categorical edge cases: an all-false mask raises; KL with masks gives NaN/inf; kickstart ignores masks
- **Location**: `networks/distributions.py:55-57, 79-84`; `bc/kickstart.py:84-93`.
- **Evidence (VERIFIED-BY-RUNNING, t2)**:
  ```
  CategoricalDist with all-false row raised: ValueError Expected parameter logits ... Real()
  KL(a||b) [both masked] = tensor([nan, nan]);  KL(unmasked||masked) = tensor([inf, inf])
  ```
  Normal masked log_prob, entropy and gradients are finite, and the log-prob of a masked action is `-inf` with a finite gradient.
- **Impact**:
  - Turn-based envs that give the waiting player an all-zero mask, or terminal states with no legal move, crash the worker.
  - Any KL on masked distributions is NaN.
  - Kickstart computes the KL on *unmasked* distributions, so the student is pulled toward the teacher's preferences over illegal actions.
- **Fix**:
  - For rows with no valid action, fall back to unmasked or a no-op index, or fail with a clear message at the env boundary.
  - Make the KL NaN-safe: `torch.where(p>0, p*(logp-logq), 0)`.
  - Pass the batch action masks into `KickstartLoss.compute` and apply them to both student and teacher.

### R1-14 — LOW/MEDIUM — design — kickstarting: KL direction, same-architecture teacher, no scripted or DAgger expert, extra cost
- **Location**: `bc/kickstart.py:67-93`; `launcher.py:163-174` and `distributed.py:178-188` (the teacher is built from the *student's* config).
- **What is wrong**:
  - The code uses reverse KL(student‖teacher). Schmitt et al. 2018 use the teacher-to-student cross-entropy, which is forward KL and mode-covering. Reverse KL is infinite or unstable for near-deterministic teachers, such as a BC model fit to a scripted bot.
  - The teacher must have exactly the student's architecture, which works against "warm start from any bot".
  - CLAUDE.md "Mode B" (online expert labels from a scripted bot) is not implemented.
  - Each step runs one extra student encoder forward plus a teacher forward over all T×B steps, outside autocast.
- **Evidence**: BY-READING.
- **Fix**:
  - Make the KL direction configurable, defaulting to forward.
  - Give the teacher its own network config.
  - Reuse the student distribution from the main forward.
  - Support expert-action labels stored in the chunk (worker queries a `ScriptedAgent`) with a CE loss.

### R1-15 — LOW/MEDIUM — correctness — continuous actions are never clipped or squashed; log_std is unbounded
- **Location**: `networks/distributions.py:93-97`; `core/action_spec.py:186-218` (decode passes raw values through); `examples/space_miners/env.py:234-238` sends raw `accel` (Box[-1,1]) into the game.
- **Evidence (VERIFIED-BY-RUNNING, t2)**: `log_std=30 → sample 2.2e13`. The entropy bonus keeps pushing log_std up with no bound.
- **Fix**:
  - Clip to `space.low/high` in `ActionSpec.decode`, while storing the unclipped action for log_prob.
  - Clamp log_std (for example to [-5, 2]).
  - Optionally offer a TanhGaussian.

### R1-16 — LOW — bug — `DiagGaussianDist` with a multi-dimensional Box gives the wrong log_prob shape
- **Location**: `networks/distributions.py:100-106`. `ActionSpec` accepts any Box shape (`action_shape=space.shape`).
- **Evidence (VERIFIED-BY-RUNNING, t2)**: mean `[5,2,3]` gives `log_prob shape (5, 2)` (expected `(5,)`) and `action_dim: 3`.
- **Fix**: sum over all event dims, for example with `Independent(Normal, len(event_shape))`, or reject non-1-D Box spaces.

### R1-17 — LOW — bug — ActionSpec edge cases
- **Location**: `core/action_spec.py`.
  - `Discrete(n, start=k)` and `MultiDiscrete(start=...)` ignore `start`.
  - A size-1 Box inside a Dict decodes to a Python `float` instead of a `(1,)` array (`:217`).
  - The `_MAX_DISCRETE` limit also applies to simple Discrete, which is stored as int64 and doesn't need it.
- **Evidence**: BY-READING.
- **Fix**: add and subtract `start`, and keep array shape for Box components.

### R1-18 — LOW — performance — repeated stacking and transfers, many syncs
- **Location**: `appo.py:87-118` (`_prepare_batch` re-stacks and re-copies all fields every minibatch of every epoch; `pin_memory()` on every call); `:304` (`.item()` per metric per minibatch, about 8+ syncs); `vtrace.py:69-70` (a T-step Python loop of tiny GPU kernels; under `torch.compile` it is fully unrolled and recompiles for new B).
- **Evidence**: BY-READING.
- **Fix**:
  - Stack and transfer once per `train_step` and index minibatches on the device.
  - Accumulate metrics as tensors and move them to CPU once.
  - Run V-trace on CPU, or vectorize it with a reverse cumulative-product/scan.

### R1-19 — LOW — ux — config knobs that do nothing
- **Location**: `config.py:65` and `appo.py:289-297`.
  - `max_grad_norm` has `gt=0`, so `if cfg.max_grad_norm > 0` is always true and clipping can't be turned off.
  - GradScaler is created for bf16 too, where it isn't needed.
- **Missing knobs** that APPO users expect: value-loss clipping, value or reward normalization (PopArt), Adam eps, separate policy and value LR.
- **Evidence**: BY-READING.

### R1-20 — MEDIUM — design — `BaseAlgorithm` is effectively actor-critic-only; adding R2D2, DQN or MCTS needs changes across the system
- **Location**: `algorithms/base.py`.
  - `network` is typed `ActorCriticNetwork`.
  - The worker hard-codes `net.act()`, which returns `(actions, log_probs, values, hidden)`.
  - `TrajectoryChunk` has fixed fields with no `extras`.
  - The replay hook is minimal: `create_replay_buffer(queue_size*4)`, `sample()` returns chunks, and there's no priority-update API.
  - There's no `state_dict`; the learner uses `hasattr` and private fields (`_optimizer`, `_policy_version`, `setup_lr_schedule`).
- **Impact**:
  - DQN/R2D2 need ε-greedy acting, Q-values or priorities stored per step, and burn-in.
  - MCTS needs search at the actor and visit-count targets.
  - None of this fits without touching the worker, the chunk format, the learner and serialization.
- **Fix**:
  - The algorithm provides an `Actor`: `act(obs, mask, state) -> (actions, extras: dict[str,Tensor], new_state)`.
  - Chunks carry `extras`.
  - Add `state_dict()` and `load_state_dict()`.
  - Add `add_chunks()`, `ready()`, `train_step()`.
  - Add `wants_bootstrap_obs` and `sequence_requirements`.

### R1-21 — MEDIUM — test-gap
- No worker-to-learner recurrent integration test (R1-01). The helpers' `HIDDEN_SIZE == latent_dim` hides R1-02.
- No tests for:
  - truncation;
  - all-false masks or masked KL;
  - NormalizeObs inside APPO;
  - schedule or kickstart state across resume;
  - kickstart or BC with recurrent nets or masks;
  - multi-dim Box.
- No "does it learn" test (a bandit or short chain MDP where APPO must reach a known return). The current APPO tests only check that losses are finite and keys exist. So a sign error in advantages, or a wrong ratio direction, would still pass the suite.

### R1-22 — HIGH — correctness (cross-scope, in worker; affects algorithm correctness) — turn-based `info["active"]` drops rewards and dones for inactive slots and desyncs RNN state
- **Location**: `worker/rollout_worker.py:496-517` (only active slots append) and `:465-480` (inference, which advances hidden state, runs for every slot every step).
- **What is wrong**:
  - Rewards received while a slot is inactive are dropped. In a typical turn-based game, the loser's −1 arrives on the winner's move.
  - If the episode ends on another player's move, the inactive slot's buffer never gets `done=1`. V-trace then bootstraps across episodes, and that player never sees the loss.
  - For RNN policies, the hidden state advances on unrecorded steps, so the learner's unroll from `h_init` no longer matches the behavior policy.
- **Evidence**: BY-READING. No shipped example sets `active`, so this is latent. It would show up immediately in turn-based competitions.
- **Fix**:
  - Keep a pending reward per slot and add it to the slot's last recorded transition.
  - On episode end, set `done=1` (and add the reward) on each slot's last recorded transition.
  - Don't update RNN state for inactive slots, or record their steps with a "no-op/ignored" flag.

### Notes (not defects)
- **Double importance weighting**: the PPO surrogate uses `ratio = π/μ` times a V-trace advantage that already contains `min(ρ̄, ρ)`. Sample Factory and RLlib APPO do the same, so it is acceptable. Documenting it would help.
- **Stale stored RNN state**: `h_init` comes from the behavior policy at chunk start, with no R2D2-style burn-in. That is fine for APPO with small lag. Add burn-in when R2D2 lands.
- **Correct parts**:
  - The V-trace recursion and the PG advantage `ρ_t(r_t + γ v_{t+1} − V(x_t))` are right.
  - Trace cutting at episode boundaries is right.
  - V-trace is recomputed per minibatch with the current values, so it isn't stale when `num_epochs>1`.
  - AMP does unscale-before-clip and uses `scaler.step`.
  - Advantages are normalized per minibatch.
  - Masked log_prob, entropy and gradients are finite.
  - The in-chunk RNN reset matches the worker's reset (on-policy recompute matches worker log-probs to 0.0 and values to 6e-9, once the hidden shape is fixed, t3).
  - CompositeDist sums log_prob and entropy correctly over sorted keys.

## Subsystem assessment

**What is good.**
- The numerical core is sound and readable:
  - V-trace matches the paper (verified against a brute-force reference, including dones and ρ̄≠c̄).
  - The APPO loss matches Sample Factory-style APPO.
  - The AMP order is right.
  - Masked categoricals are numerically safe on the usual path.
- The composite action codec (ActionSpec + CompositeDist) is a nice abstraction with zero overhead for simple spaces.
- Keeping obs-normalization stats in buffers is the right call for syncing with workers and saving checkpoints.
- The code is small and easy to follow.

**What is structurally weak.**
1. **Interfaces between layers are untested end to end.**
   - The worker and the learner disagree on the hidden-state layout (R1-01).
   - The network's single-step and sequence entry points behave differently, and the RNN is silently bypassed (R1-02).
   - The BC and kickstart code reach into `encoder`/`policy` directly instead of going through the network's public API.
2. **Episode semantics are thin.**
   - There is one `dones` flag: no terminated/truncated split, no bootstrap observation, and turn-based reward and done aggregation is lost.
   - That limits correctness in exactly the competitive, turn-based, time-limited games the project targets (R1-03, R1-04, R1-22).
3. **Training state is scattered.**
   - Schedule, kickstart step, scaler and normalization updates live in different places.
   - None of them are part of a resumable algorithm state (R1-07, R1-08, R1-11).
4. **Off-policy/lag awareness is missing.**
   - There are no lag metrics or filtering, even though async multi-machine training is the main deployment goal (R1-10).
5. **The algorithm interface is APPO-shaped** (R1-20), so the Tier-2/3 algorithms will need a cross-cutting refactor.
6. **The warm-start path is fragile.** BC ignores masks, leaves the critic random, has no recurrent support, and casts Box actions to integers. Kickstart is same-architecture only, uses reverse KL and ignores masks (R1-05, R1-06, R1-13, R1-14).

**Recommended changes, in priority order.**
1. Fix the hidden-state layout (R1-01). Add a worker-to-learner recurrent integration test and a small "APPO solves a bandit/chain" test (R1-21).
2. Make the RNN impossible to bypass silently (R1-02). Route kickstart and BC through the network's sequence API, and give eval.py hidden-state tracking.
3. Episode semantics:
   - split terminated/truncated and bootstrap on truncation with `terminal_observation`;
   - ship `bootstrap_obs` so the learner computes `V(s_T)` itself;
   - fix turn-based reward/done aggregation and RNN state for inactive slots (R1-03, R1-04, R1-22).
4. Add `BaseAlgorithm.state_dict()` (optimizer, scheduler, scaler, kickstart step, version), and drive schedules by consumed samples. Add a minimum learner batch (R1-07, R1-11).
5. Update normalization stats once per train_step, and use frozen stats in loss forwards (R1-08).
6. Add policy-lag metrics and `max_policy_lag` filtering (R1-10).
7. BC/kickstart:
   - masks;
   - loss chosen from the distribution type;
   - value pretraining or critic warmup;
   - forward-KL option;
   - separate teacher config;
   - scripted-expert labels (R1-05, R1-06, R1-13, R1-14).
8. Recurrent performance: a cuDNN path with episode segmentation (R1-09). Batch prep and metric-sync cleanup (R1-18).
9. Generalize the algorithm interface (Actor + chunk `extras` + replay API) before starting R2D2/DQN (R1-20).
10. Small items: Box clipping and log_std bounds, multi-dim Box, ActionSpec `start`, dead config knobs, and the GAE doc claims (R1-12, R1-15, R1-16, R1-17, R1-19).

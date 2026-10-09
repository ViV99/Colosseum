> Origin: `.superpowers/sdd/2026-10-08-sp2-game-model/final-review-A.md` (SP2 SDD workspace, git-ignored, deleted after the merge). Copied 2026-10-09.
> Frozen record; do not edit. Only absolute scratch paths were replaced with `<scratch>/`.

# SP2 final whole-branch review — AREA A: game contract, data trees and models

Reviewer: area A (read-only). Range `7430361..7d55974` (branch `sp2-game-model`).
Scope read in full: `envs/{game,spaces,contract,vector}.py`, `core/{tree,specs,outcomes,types,roles,validation,errors}.py`,
`networks/{base,model,composed,heads,normalization,cores}.py`, `networks/dist/*`, every `examples/*/game.py` and
`examples/*/models.py`, `tests/contract/game_harness.py`, the relevant parts of `tests/game_helpers.py`, and the unit
tests of these modules. Focused run: 404 tests of this area (`tests/contract` + 16 unit files), all green in 9.8 s.
Experiments (scratch `<scratch>/final-review/A/exp{1,2,3}.py`) are cited per finding.

## Strengths

- **Seat lifecycle matches spec block 1 rule by rule.** `EpisodeTracker.on_step` (`envs/contract.py:138-184`) checks
  rewards and `terminated` against the phases *before* applying this step's eliminations. So a reward in the
  elimination step is allowed, later rewards, acts and re-terminations are `EnvContractError`, and EMPTY seats
  get nothing (`_check_empty_seat_data`, also on the final step). `episode_over` with a non-empty `acting` is
  rejected, `truncated`/`outcome` without `episode_over` are rejected, `final_obs` (and the final `global_state` when
  the role declares one) is required only for live seats not terminated in the same step (`_check_truncation`), and
  idle ticks are bounded by `max_idle_steps`, with the reset counting as the first idle tick. Every check I listed
  against the spec's "Проверки контракта" paragraph exists and has a test in `tests/unit/test_episode_tracker.py`.
- **Outcome resolution is correct and general.** `core/outcomes.py` uses the mean of seat returns (not the sum), ranks
  as `1 + #strictly better`, `team_rank`-only keeps the mean score, keys must be exactly the layout's teams, and
  non-finite values are rejected. The outcome kind comes from the team count (`GameSpec.outcome_kind`). There are
  no `num_players == 2` branches.
- **The distribution layer is careful.** Every invalid position is selected with `torch.where` (`UnitsDist._sum_valid`,
  `categorical_entropy`, `categorical_kl`). `masked_log_softmax` uses `finfo.min` instead of `-inf`, so all-illegal
  rows stay finite (checked in fp16 and bf16 in exp1: finite values and finite gradients). The `only_if` gate has
  exactly one implementation (`units_component_valid`), shared by `UnitsDist` and `ActionSpec.first_illegal_action`,
  and it gates log-prob, entropy, KL and validity on the *recorded* parent value. `log_prob` is by construction the
  sum of `unit_log_prob` over valid deciders. Decider layout (flat decider 0, then `max_units` per units group in
  spec order) is the same in `ActionSpec.num_deciders` and `TreeDist`. Unit tests pin exact values, including dead
  units, empty rows, MultiDiscrete/Box components and gradient isolation at finite parameters.
- **`global_state` stays on the value path.** `ComposedModel.unroll` (`networks/composed.py:50-67`) sends
  `global_state` only to `critic_encoder`, whose output is concatenated before the value head. The encoder and
  policy never see it. `with_value=False` needs no `global_state` and returns `value=None`. The contract test
  `test_global_state_changes_values_but_not_log_probs` proves this end to end for all four cores: real
  RolloutLoop → payload → APPO; perturbing `global_state` changes every ACT/BOOT value and leaves the log-probs
  bit-identical.
- **Dtypes are preserved.** `ObsSpec.allocate`, `tree_to_numpy`/`tree_to_torch` and `numpy_to_tensor` keep leaf dtypes.
  `test_uint8_observations_reach_the_model_on_both_sides` checks that `uint8` reaches the worker's `step`, the
  chunk and the learner's `unroll`. Actions are int64 for discrete parts and float32 for Box parts; masks are bool.
- **Strong contract tests.** The learner reproduces the worker's joint and per-unit log-probs at zero lag for 4 cores
  on chunks with BOOT/PAD/truncation/elimination, including mutation detection. The per_unit loss and mean_valid
  entropy are recomputed by hand under random unit masks with 0..3 valid deciders.
- **Examples work as reference code.** All nine games follow the contract cleanly: mirrored perspectives with the
  actions mapped back, the dead-teammate rule in `team_tag`, elimination with shared fractional ranks in `tron`,
  `truncated` + `final_obs` only in `coin_grid`, and `Dict` spaces built from lists so the component order is
  explicit. Constructor arguments are checked.

## Issues

### Critical

None found.

### Important

**I-1. `UnitsHead` crashes for natural `Units` component names (`"type"`, `"to"`, `"float"`, `"train"`, names
with `.`).** `networks/heads.py:27-32` uses the component names as keys of `nn.ModuleDict` / `nn.ParameterDict`.
`add_module` rejects any key that is an attribute of `nn.Module`, and any key that contains `.`.
- Scenario (exp2): `Units(4, Dict([("type", Discrete(3)), ("amount", Box(-1, 1, (1,)))]))` →
  `UnitsHead(group, 8)` raises `KeyError: "attribute 'type' already exists"`. The same happens for `"float"`, `"to"`,
  and `"target.x"` (`module name can't contain "."`). `colosseum validate` reports it as
  `ConfigError: agent 'a': failed to build the model from networks: KeyError: ...`.
- Why it matters: "type" is the most natural name for an action-type component. The spec's own `only_if`
  example ("цель sap учитывается, только если тип действия = sap") and Lux-style action spaces lead users to it.
  `UnitsHead` is the spec's head helper (block 2), and `Units` accepts these names without complaint. The failure
  is loud, but the message does not point at the cause.
- Fix: key the dicts by position (`f"c{i}"`) or by an escaped name (`"c_" + name.replace(".", "_")`), and keep a
  `names` list for the output dict. `state_dict` keys change, which is fine with no backward compatibility. Add a
  unit test with `"type"` and a dotted name.

### Minor

**M-1. `validate` breaks the global context format "seat, episode step, layout"** (ledger T6.2, a final-wave
candidate). `core/validation.py:122,133` and `_check_spaces` (`:98-106`) produce
`"validate, layout 2p, episode step 3, seat 1: observation is not in ..."`, while the tracker inside the same
command produces `"validate, seat 1, episode step 3, layout 2p: ..."`. Fix: build the `where` from seat, episode step
and layout in that order (reuse `EpisodeTracker._where` or a shared helper).

**M-2. `resolve_outcome` turns a malformed key set into a bare `TypeError`** (ledger T1.4).
`core/outcomes.py:26` calls `sorted(values)`. Exp3: `Outcome(team_rank={0: 1.0, "b": 2.0})` →
`TypeError: '<' not supported between instances of 'str' and 'int'`, raised instead of the intended
`EnvContractError` with context. Fix: `sorted(values, key=repr)`.

**M-3. `ObsSpec.check` lets a ragged observation escape as a bare `ValueError`, and accepts `None` at scalar leaves**
(ledger T1.3). `core/specs.py:141-142` calls `np.shape(value)` unguarded. Exp3: a ragged list raises
`ValueError: setting an array element with a sequence...` with no seat/step/layout context. For a `Discrete`
observation leaf, `None` is accepted, and the worker's cast fails later. Fix: wrap `np.shape` in `try/except` and
reject `None`/dict at array leaves, raising `EnvContractError(where...)`.

**M-4. A dead `SubprocessVectorEnv` child surfaces as a bare `EOFError`** (ledger T1.6). `envs/vector.py:228` calls
`self._conns[w].recv()` unguarded. If an env segfaults (Box2D in `space_miners`) or is OOM-killed, the worker log
shows `EOFError` with no env range or exit code. Fix: catch `EOFError`/`OSError` in `_round` and raise
`RuntimeError(f"env child {w} (envs {a}..{b-1}) died, exit code {proc.exitcode}")`.

**M-5. The "no NaN, no gradient" promise holds only for finite parameters** (`dist/units.py:10`, `dist/base.py:5-6`).
`torch.where` zeroes the forward pass, but the backward pass still multiplies 0 by the local derivative of the
discarded branch. Exp1: NaN logits and NaN means on an absent unit give a finite forward and finite gradients on
valid units, but NaN gradients on the absent unit's parameters, which then flow into the head weights.
Realistic trigger: an attention encoder over a unit set that is empty on an ACT slot, e.g. a `unit_harvest` side
whose workers all died. `docs/ENV_GUIDE.md` already tells encoder authors not to produce NaN, so this is the
user's responsibility. Fix: soften the docstrings ("for finite parameters"), or sanitize parameters at invalid
positions before `log_softmax`/the Gaussian terms (the double-`where` pattern).

**M-6. Gaussian log-prob and entropy run in fp16 under CUDA autocast.** `dist/leaf.py:50-53`, `:262` and
`dist/units.py:188` cast the recorded float32 actions to `mean.dtype`. Under AMP fp16 that loses about 1e-3
relative precision, adds noise to the per-decider log-ratios, and can overflow `(x-mean)^2/(2var)` for small std.
`UnitsDist.unit_kl` already upcasts; `log_prob`/`entropy` do not. This matches SP1 (`Normal(mean_fp16, ...)`) and
the path is GPU-only and has never run. Fix: compute `gaussian_log_prob`/`gaussian_entropy` in float32. Add the
point to `docs/GPU_CHECKS.md`.

**M-7. `global_state` is silently ignored without a critic encoder** (ledger T2.3). `composed.py:64`: if a role
declares `global_state_space` and the agent has no `critic_encoder_class`, every chunk still carries
`global_state` (payload ×1.61 on `team_tag`), and the model never reads it. Fix: a one-line `validate` warning
("role declares global_state_space but agent X has no critic_encoder_class; the global state is stored and sent
but unused").

**M-8. `role_signature` ignores the `n`/`nvec` of `Discrete`/`MultiDiscrete` observation leaves**
(`core/roles.py:18`, `ObsSpec` stores shape and dtype only, `specs.py:87-89`). Two roles whose observations are
`Discrete(5)` and `Discrete(9)` count as "same spaces", so one agent may play both with a network built for the
first. An embedding then crashes at runtime with an index error, which is loud, not silent. Fix: include `n`/`nvec`
in the leaf signature, or park.

**M-9. All seats eliminated without `episode_over` is reported only after `max_idle_steps` idle ticks**
(`contract.py:275-279`). After the last live seat is terminated, the tracker could fail at once ("no live seats
left and the episode is not over") instead of stepping the env 1000 more times with `step({})`. Not a hang.

**M-10. Duplicate work on the hot path.** `networks/model.py:101`: `act()` computes `unit_log_prob` and then
`log_prob`, which recomputes it. APPO does the same in `_evaluate`. `UnitsDist` recomputes the gate table in each
of `unit_log_prob`/`unit_entropy`/`unit_valid` (ledger T2.2). Criterion 5 passes, so park; a cheap win is
`log_probs = unit_log_probs.sum(-1)`.

**M-11. Test gap: the end-to-end learner-reproduces-worker test uses `Units(3, Discrete(3))` only.** There is no
`only_if`, no Box component inside `Units`, and no flat decider 0 next to units on the chunk/payload/APPO path
(`tests/contract/test_learner_reproduces_worker.py:59`). The same cases are covered one level down:
`tests/unit/test_policy_model_v2.py::test_unroll_reproduces_step_per_decider` uses `UnitsGame`
(base + units + `only_if`), and the dist unit tests cover Box components. So the risk is low. A
`space_miners`-shaped role (`Units(Dict(accel=Box, push=Discrete))` with `only_if`) in the contract test would close
the gap cheaply.

## Deferred minors to fix now (ledger references)

Recommended for the single final fix wave. All are cheap and local, and each improves an error message on a path env
authors hit first (spec principle "Ошибки контракта громкие ... с контекстом"):

1. **T6.2**: validate context order (M-1). It violates the brief's global format.
2. **T1.4**: `resolve_outcome` `sorted(values)` `TypeError` (M-2). One line.
3. **T1.3**: `ObsSpec.check` ragged/`None` leaves → `EnvContractError` with context (M-3).
4. **T1.6**: `SubprocessVectorEnv._round` dead child → error naming the child, its env range and exit code (M-4).

Together with **I-1** (not in the ledger), these five items are the area-A part of the fix wave. All other area-A
ledger minors stay parked: T1.1 (`tree_map` message key order, `tree_assign` half-write), T1.2 (`Units.contains` NaN,
`sample` zeros), T1.5 (stale phases after a failed reset), T2.1 (`_assemble`/`_fmt` copies, `mode()` grad,
`nvec` writable), T2.2 (component error wording, clamping, gate rebuild), T2.3 (`NormalizeObs` `path=()` message,
`RandomPolicy` checks; see M-7 for the `global_state` one) and T2.4 (test gaps). None of them can corrupt training.

## Recommendations

- If the M-5 docstring stays as is, add one sentence to `docs/ENV_GUIDE.md` §4 (attention encoders): "NaN unit
  embeddings on ACT slots with zero units poison gradients even though the unit is masked."
- When the GPU checks run (`docs/GPU_CHECKS.md`), include a Box-in-Units AMP step and compare the zero-lag
  per-decider log-ratios with fp32 (M-6).
- SP3 can reuse `units_component_valid` for scripted bots' legality checks: it is already the single gate
  implementation.

## Assessment

**Ready to merge: With fixes.** I found no Critical issue in area A. The contract, outcome rules, distributions,
model protocol and dtype handling match the spec, and the tests show it end to end. Before merge: fix I-1
(`UnitsHead` key collision), plus the four cheap error-context items above (M-1 to M-4). The remaining Minors can be
parked.

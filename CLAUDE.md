# Colosseum — Distributed RL Training Framework for Competitive Bot Programming

## Project Vision

Reusable framework for competitive bot programming competitions (Lux AI, Neural MMO, CodeCraft, etc.).
The core problem: competitions are short, no time to rebuild RL infrastructure each time.
Colosseum provides the full pipeline: BC → RL → Self-Play → PFSP/League, distributed, modular, with dynamic agent and machine management.

## Research Documents

- `research/01_distributed_rl_architectures.md` — Distributed RL: GORILA, A3C, IMPALA, Ape-X, R2D2, SEED RL, Podracer, Sample Factory, PureJaxRL, PufferLib, Cleanba, GPU envs, V-trace math
- `research/02_competitive_rl_framework_research.md` — AlphaStar, OpenAI Five, AlphaZero/MuZero, Cicero, Pluribus, HoK; matchmaking (PFSP, PSRO, PBT, ELO/TrueSkill); BC/DAgger/GAIL/IRL/kickstarting; N-player games; OpenSpiel, PettingZoo, JaxMARL, Mava, EPyMARL; MAPPO/QMIX/MADDPG/HAPPO
- `rl_frameworks_survey.md` — 30+ frameworks: RLlib, SB3, CleanRL, TorchRL, Tianshou, Acme, Sample Factory, EnvPool, JaxMARL, Mava, EPyMARL, MARLlib, PettingZoo, OpenSpiel, Gymnax, Pgx, Brax, PufferLib

---

## AGREED Architecture Decisions

### Core Distributed Architecture: IMPALA-style Async Actor-Learner

- **CPU Workers** (N machines): run environments + local inference (forward pass on CPU), collect trajectories in fixed-length chunks (T=128-512 steps), send to learners async
- **Learners** (M machines, usually with GPU but not required): receive trajectory chunks, run backward pass + optimization, push updated weights to Weight Store
- **Weight Store**: shared storage for latest model weights per agent. Single-machine: shared memory. Multi-machine: dedicated service or Redis
- Workers periodically pull fresh weights from Weight Store (not every step — e.g., every K episodes or every few seconds)
- V-trace off-policy correction handles policy lag between worker's policy and learner's current policy
- No synchronization barriers — workers and learners run independently at their own pace

### Why IMPALA-style:
1. Workers fully async → dynamic add/remove of workers and agents without stopping
2. V-trace handles policy lag gracefully → quality
3. GPU utilization maximized — learner always has incoming data
4. PPO objective can be used WITH V-trace correction (= APPO, as in Sample Factory)
5. Cleanba paper confirms: IMPALA doesn't lose data efficiency from async, PPO does

### Algorithms (Priority Order)

**Tier 1 (implemented):**
- **APPO** (Async PPO with V-trace) — primary algorithm
  - PPO clipped surrogate loss + V-trace importance weights
  - V-trace(λ) targets and advantages (`algorithm.vtrace_lambda`; GAE is not used)
  - LSTM/GRU support for partial observability
  - Action masking for constrained action spaces
  - Many units per seat (`Units`): `ratio_mode` (`joint` / `per_unit`), `unit_trace` (scalar ρ for V-trace), `entropy_reduction`; per-decider diagnostics
  - Centralized critic: `global_state` reaches only the value path (`networks.critic_encoder_class`)
  - AMP (mixed precision) training support
  - This is what Sample Factory, PufferLib use; similar to OpenAI Five's approach

**Tier 2 (next):**
- **Rainbow DQN / R2D2** — off-policy, discrete actions
  - Prioritized replay buffer (Ape-X style distribution)
  - R2D2 = Rainbow + LSTM for partial observability
  - Good for turn-based games

**Tier 3 (later):**
- SAC (continuous), AlphaZero/MuZero (MCTS)

**Behavioral Cloning:**
- Offline BC: supervised learning from recorded trajectories (`-log_prob` of the recorded action for every distribution type, masks applied)
- Online BC (kickstarting): `loss = RL_loss + λ * KL(BC_teacher || policy)` (forward KL by default, `training.kickstart_kl`), λ decays over training
- Both approaches available; BC phase runs before RL phase

### Framework: PyTorch
- JAX can be added later but not initial priority

### No Ray — Custom Distribution Layer

**Transport: gRPC**
- Control plane (coordinator ↔ workers/learners): gRPC unary RPCs (RegisterWorker, RequestMatch, ReportResult, etc.) (target; see Roadmap)
- Data plane (workers → learners): gRPC client streaming for trajectory chunks
- Weight sync: Weight Store service (gRPC or shared memory)
- Serialization: protobuf for metadata, raw bytes + lz4 compression for tensors

**Single-machine fallback:**
- multiprocessing.Queue for control
- multiprocessing.shared_memory for weights (zero-copy) (target; see Roadmap — today: newest-wins mp.Queue of numpy payloads)
- Shared memory ring buffer for trajectories (Sample Factory style) (target; see Roadmap — today: mp.Queue of numpy chunk payloads)
- Same interfaces, different transport backend

### Trajectory Handling

- Fixed-length rollout chunks of T steps (T=128-512, configurable)
- Chunk slots are `act` (a decision), `boot` (an observation for the learner's bootstrap value) or `pad`; episode boundaries are handled within chunks (traces stop at `terminal` and `boot`), and the learner computes every value with its current network
- For LSTM: hidden state carried across chunks within episode, reset at done. Stored state approach: h_init saved at chunk start, replayed during training.
- One chunk ≈ 300KB per agent per chunk (for obs_dim=256, T=256)
- At 100 workers × 10 chunks/sec = 300 MB/sec total throughput — manageable for 10Gbit or shared mem

---

## AGREED Training Pipeline (Competition Lifecycle)

### Phase 1: Setup
- User writes: `MyGame(MultiAgentEnv)` with a `GameSpec` (roles, layouts, teams) — see `docs/ENV_GUIDE.md`
- User writes: `MyEncoder(BaseEncoder)` — game state → tensor
- User writes: `MyPolicy(BasePolicy)` + `MyValue(BaseValue)` — neural network heads
- User selects: algorithm (APPO, DQN, etc.) + hyperparameters via config

### Phase 2: Behavioral Cloning (optional)
- **Offline BC**: load trajectories from disk (recorded games from other players, etc.)
- **Online BC**: scripted agent generates trajectories, neural agent learns from them
- Training: supervised `-log_prob` loss (masked; sequences for stateful models)
- Result: initial policy that plays at basic level

### Phase 3: Self-Play
- Agent plays against pool of its own checkpoints + current self
- Each N-player match: random mix of latest (collect trajectories) + historical checkpoints (don't collect)
- Checkpoint pool: FIFO, save every K training steps
- Purpose: stable improvement without being crushed by much stronger opponents

### Phase 4: PFSP / League (longest phase)
- For each match, random decision:
  - **Solo match** (prob = self_play_ratio): same as self-play phase (own checkpoints)
  - **Arena match** (prob = 1 - self_play_ratio): all slots filled with TRAINABLE agents from the pool
- Every team gets a core agent (the owner, its checkpoints, or another trainable agent by PFSP); `matchmaking.teammates: self | mixed` fills the team's other seats; a role the core does not play goes to an agent that plays it
- ALL trainable agents in arena match collect trajectories → one env step feeds multiple learners
- PFSP priority function: `f(win_rate) = (1 - win_rate)^p` (focus on hard opponents) or `f(x) = x*(1-x)` (balanced)
- ELO/TrueSkill tracking for all agents (ELO implemented; TrueSkill-type ratings are a target; see Roadmap)
- Dynamic: add/remove agents and machines at any time without stopping (target; see Roadmap)

### Eval Mode
- Inference-only matchups between any set of checkpoints / `.pt` state dicts (`colosseum.eval`); no training
- Same engine as training rollouts (`MatchRunner`): one model `State` per (env, seat), `StepResult.acting` and action masks
- Lineups via `schedule_lineups` (per layout): for a pair (a, b), match m gives team i to the core `(a, b)[(i + m) % 2]` where the roles allow, teams filled homogeneously; one agent fills every team; one-team layouts play each agent's homogeneous team plus mixed compositions (cross-play); `--num-matches` is per pair (or composition) and layout
- Reports per layout and role: `wdl` — W/D/L, win rate and score with 95% Wilson intervals, per side; `rank` — mean rank with CI, first-place share, pairwise matrix; `score` — mean score with CI, cross-play table
- JSON output with `--output`; architecture of each checkpoint rebuilt from its `meta.json` (`.pt` files from `--config`)
- Command: `colosseum eval -c config.yaml -a A=runs/x/checkpoints/agent_0/ckpt_v100 -a B=runs/x/checkpoints/agent_0/ckpt_v200 --num-matches 1000 --output result.json` (`--num-matches` is per pair; the architecture comes from each checkpoint's `meta.json`)

---

## AGREED Detailed Design Decisions

### Agent Pool / Registry

Agents in pool can be:
- **TrainableAgent**: has its own learner, actively training, generates checkpoints. Can have completely different architecture, encoder, algorithm, hyperparameters from other agents.
- **FrozenCheckpoint**: static weights loaded from disk, used as sparring partner, no training
- **ScriptedAgent**: rule-based bot, no neural network

Multiple trainable agents can coexist with different architectures, different RL algorithms, different encoders — they are fully independent. This allows comparing approaches side by side.

### Learner-Agent Mapping

- One learner process = one trainable agent
- Multiple learner processes can share one physical machine (GPU time-sharing for small models)
- Load balancing: if 10 agents and 2 GPU machines → distribute ~5 learner processes per machine
- Learner machines don't require GPU (modern AI-focused CPUs work too) — "learner" just means "machine that does backward pass + optimization"

### Worker Parallelism (INTRA-machine)

Each CPU worker machine runs:
- Multiple worker PROCESSES, each managing multiple ENVs
- Each process does: step multiple envs, run inference (batch forward pass), collect trajectories
- Vectorized env execution within each process (like Gymnasium's VectorEnv)
- This maximizes CPU utilization on each physical machine

### Multi-Agent Worker Architecture

Workers are shared across all agents:
- Each worker receives all agents' network factories, trajectory queues, and weight queues
- Seats mapped to agents per env by a `Lineup` (layout + one `SeatAssignment(agent_id, network_id, collect)` per seat); an agent's roles come from `agents.<id>.roles`
- Inference grouped by `(agent_id, network_id)` for batching efficiency
- Trajectory chunks routed to the correct agent's learner queue
- Single-agent training is a special case (one agent in the agents dict)

### Learner Parallelism (INTRA-machine)

Each learner machine runs:
- Multiple learner processes (one per agent being trained)
- Each process owns its GPU memory slice (or time-shares GPU)
- Incoming trajectory queue per learner process
- Batch incoming chunks → compute loss → backward → optimize

### Checkpoint Management

- Save checkpoints every K training steps (configurable per agent)
- FIFO pool per agent (keep last N checkpoints, configurable)
- Checkpoints stored on shared filesystem or object store
- Checkpoint = state_dict + optimizer_state + training_step + metrics snapshot

### Orchestration

- **Docker containers** for all components (coordinator, workers, learners, weight store)
- **Kubernetes** for cluster orchestration:
  - Workers as a scalable Deployment/StatefulSet (scale up/down dynamically)
  - Learners as pods with GPU resource requests
  - Coordinator as a single-replica Deployment
  - Weight Store as a StatefulSet with persistent volume
  - HPA (Horizontal Pod Autoscaler) for workers based on queue depth
- For single-machine dev: docker-compose or direct process launch
- The system must work both locally (processes) and distributed (k8s) with same code

### BC Architecture Decision

Two modes for behavioral cloning:

**Mode A — Offline BC (simpler, implement first):**
1. Scripted agent plays games, trajectories saved to disk
2. OR: download external replay files
3. Neural agent trains supervised on these trajectories
4. Standard supervised learning loop, no env interaction needed

**Mode B — Online BC / DAgger-like:**
1. Neural agent plays (self-play or vs scripted)
2. At each state, scripted expert provides "what I would do"
3. BC loss added to RL loss: `total_loss = rl_loss + λ_bc * bc_loss`
4. λ_bc decays over training (kickstarting)
5. Requires expert to provide actions at inference time (more complex but better quality)

Both should be supported. Mode A is simpler and covers the case of external replays.

### Metrics & Logging

- WandB integration for all metrics
- Per-agent: reward curves, loss components, ELO over time, learning rate, entropy
- Per-match: outcomes logged to matchmaking engine
- Win rate matrix: all pairwise win rates between agents
- Dashboard: real-time view of training progress, agent pool, active workers/learners

---

## Modularity Requirements

Everything inheritable/pluggable:

```python
# User implements these per competition:
class MyGame(MultiAgentEnv): ...    # GameSpec + reset/step -> StepResult (docs/ENV_GUIDE.md)
class MyEncoder(BaseEncoder): ...   # obs → latent vector
class MyPolicy(BasePolicy): ...     # latent → action distribution
class MyValue(BaseValue): ...       # latent → scalar value

# Framework provides these (swappable):
class APPO(BaseAlgorithm): ...      # training logic
class PPO(BaseAlgorithm): ...
class R2D2(BaseAlgorithm): ...

class PFSPMatchmaker(BaseMatchmaker): ...
class SelfPlayMatchmaker(BaseMatchmaker): ...

class GRPCTransport(BaseTransport): ...
class SharedMemTransport(BaseTransport): ...
```

User writes env + encoder + networks → selects algorithm + matchmaking via config → launches training.

---

## Implementation Status

State after SP2 (game model; developed on branch `sp2-game-model` on 2026-10-09, awaiting the owner's acceptance of the acceptance report; merged into `main` only after that). Test suite: **1248** fast tests, **9** slow, **18** GPU-only (`pytest -m "not gpu and not slow"` is the CI suite). The SP2 spec is `docs/superpowers/specs/2026-10-08-sp2-game-model-design.md`, the plan `docs/superpowers/plans/2026-10-08-sp2-game-model/`, the acceptance report `docs/superpowers/reports/2026-10-09-sp2-acceptance.md`. SP1: `docs/superpowers/reports/2026-10-08-sp1-acceptance.md`; the pre-SP1 review `review/README.md` is a frozen snapshot (never edit it).

Commands (run from the repo root after `scripts/setup-dev.sh`; the cwd is put on `sys.path`, also for spawned children):
- `colosseum validate -c cfg.yaml [--set k=v ...]`: `GameSpec` and config checks (roles, matchmaking), `reset` of every enabled layout and random legal steps with full contract checks, `step` and `unroll` of every agent's model, the kickstart teacher.
- `colosseum train -c cfg.yaml [--set k=v ...]`: single-machine training into `runs/<name>/`.
- `colosseum eval -c cfg.yaml -a name=path [-a ...] [--layout L ...] --num-matches N --output result.json [--num-envs E] [--deterministic] [--seed S]`.
- `colosseum bc -c cfg.yaml --agent A --data <file-or-dir> --output bc.pt [--epochs E] [--batch-size B] [--lr LR] [--seq-len L]`.
- Distributed (limited): `colosseum serve-weight-store --port P`, `colosseum run-learner -c cfg.yaml --agent A --traj-port P --weight-store host:port`, `colosseum run-workers -c cfg.yaml --weight-store host:port -l A=host:port`.
- Measurements: `scripts/bench_throughput.py`, `scripts/units_experiment.py`, `scripts/measure_global_state.py` (results in `docs/benchmarks.md`).

### Works (single machine)
- **Game model (SP2):**
  - env contract `GameSpec` (roles, layouts = lists of seats with roles and teams) + `MultiAgentEnv` + `StepResult` (`acting`, `terminated`, `truncated` + `final_obs`, `global_state`, `Outcome`); `EpisodeTracker` checks every rule and raises `EnvContractError` with "worker, env, seat, episode step, layout";
  - outcome kind from the number of teams (1 → `score`, 2 → `wdl`, ≥3 → `rank`); default outcome = mean of a team's seat returns;
  - seat lifecycle: empty / live / eliminated; a dead teammate is removed from `acting` but keeps team rewards; steps without decisions bounded by `env.max_idle_steps`;
  - observation, action, mask and `global_state` trees with preserved dtypes (`uint8` reaches the model); bool masks only; entity lists with masks; `Units(max_units, per_unit, only_if)` with `Discrete`/`MultiDiscrete`/`Box`/`Dict` components; deciders (each unit + the non-unit part).
- **Training loop:** IMPALA-style; one `MatchRunner` core for training (`RolloutLoop`) and eval. Workers infer only the policy; chunk v2 with `act`/`boot`/`pad` slots; the learner computes every value (bootstrap from `boot` slots and `final_obs`) with its current network. Only numpy payloads cross process boundaries. Every learner update uses exactly `learner.batch_chunks` chunks; `training.total_timesteps` is a global env-step budget.
- **Algorithm:** APPO with V-trace over slots; `ratio_mode` (`joint` / `per_unit`; `auto` = `per_unit` with `Units`), `unit_trace` (`joint` / `geo_mean` / `none`; `auto` = `joint` for every action, decided by the units experiment), `entropy_reduction`; with one decider every mode collapses to `joint`; every loss and metric reduces over `act` slots; per-unit diagnostics (`clip_fraction_joint`, `ess`, `log_rho_abs_p95`, `deciders_valid_mean`, ...). Centralized critic: `networks.critic_encoder_class` over `global_state`, value path only.
- **Models:** `PolicyModel` protocol (`step` for the policy path, `unroll` for the learner, `with_value=False` for BC and the kickstart teacher); `ComposedModel(encoder, core, policy, value, critic_encoder)` with `EncoderOutput(latent, aux)` for per-unit features; cores `NoCore`, `LSTMCore`, `GRUCore`, `WindowAttentionCore`; `UnitsHead`, `gridnet_to_units`; `NormalizeObs` on a leaf path. The learner reproduces the worker's joint and per-unit log-probs for all four cores (contract tests).
- **League:** lineups (layout + seat assignments); owner rotation; layout weights; self-play / arena with PFSP per layout; teams with a core and `teammates: self | mixed`; roles per agent (`agents.<id>.roles`), asymmetric games with one agent per role; seat permutation within equal role compositions. Ratings per layout (`ratings.json` = `{"env_steps", "layouts": {...}}`): team-pair ELO, win-rate matrix, `wr_vs_past`, `ScoreTracker` and cross-play for one-team layouts. FIFO checkpoint pool with roles and role signature in `meta.json`; strict resume.
- **Eval:** `MatchRunner`-based; CLI lineups per spec block 8; reports for `wdl` (Wilson intervals, per side), `rank` (mean rank, first places, pairwise matrix), `score` (mean, cross-play), by role; in-process `play_lineups` with any `PolicyModel`.
- **BC / kickstart:** BC on observation/action trees (`-log_prob`, mean over valid deciders with `Units`; an illegal masked expert action is a `DataError`); one global kickstart teacher with a per-decider KL.
- **Demo games** (`examples/`, configs in `configs/examples/`): `coin_grid` (solo), `tic_tac_toe` (1v1 turn-based; also `tic_tac_toe_attention.yaml`, `tic_tac_toe_multi.yaml`), `unit_harvest` (units), `team_tag` (2v2, `global_state`), `tron` (FFA 2–4, elimination), `predator_prey` (asymmetric roles), `coop_buttons` (cooperative); reference examples `space_miners` (Box2D) and `composite_action` (chase). Each demo game has a slow learning test against random players (`tests/learning/test_demo_learning_slow.py`; team_tag takes the better of two runs); fast learning tests cover a Units bandit and a cooperative bandit. `docs/ENV_GUIDE.md` (Russian) explains how to write an env for every game type.
- **Observability, config, lifecycle:** as in SP1 (run dir, `metrics.jsonl`, `ratings.json`, console, optional WandB, strict pydantic config with `--set`, exit codes 0/1/130/143, `SHUTDOWN_GRACE_SEC = 7`, no child alive after 10 s), with per-layout and per-role breakdowns.
- **Throughput:** 4848 / 9298 / 13702 env steps/s with 1 / 2 / 4 workers on tic-tac-toe (104 % / 98 % / 128 % of SP1; `docs/benchmarks.md`).

### Partial
- **Distributed mode** (`serve-weight-store`, `run-learner`, `run-workers`) runs on chunk v2 for games where every agent plays every role, latest weights only: no coordinator, league, ratings, `metrics.jsonl` or WandB; per-worker budgets; `run-learner` ignores `training.resume_from`.
- **`deployment/`** (Docker, K8s) is not tested.
- **Unused, kept for SP5:** `transport/local.py::LocalTransport`, `weight_store/shared_memory.py::SharedMemoryWeightStore`, `transport.mode` / `transport.grpc_port`.
- **GPU:** never run on CUDA. All CUDA paths (learner device, AMP, `pin_memory`, kickstart/BC, tree batches and `Units` distributions on CUDA) are covered only by `gpu`-marked tests; see `docs/GPU_CHECKS.md`.

### Not implemented (see Roadmap)
- Scripted, frozen and external players; PFSP over snapshots; per-agent warm start (SP3).
- One agent on roles with different spaces; per-unit rewards, values or recurrent state; built-in autoregression between action components (a custom `Distribution` can do it); a PettingZoo adapter.
- OpenSkill / Bradley–Terry ratings, a tournament command, match log, dashboard (SP4).
- Off-policy algorithms (R2D2/DQN, replay buffer), SAC, AlphaZero/MuZero.
- GPU inference on workers; a vectorized worker path for envs with hundreds of seats (SP6).
- Shared-memory weight store and trajectory ring buffer; adding or removing agents and machines during a run; gRPC control plane (SP5).

## Roadmap

| Sub-project | Scope |
|---|---|
| **SP1 Foundation and stabilization** (done) | correct, observable, robust single-machine training; `PolicyModel` protocol |
| **SP2 Game model** (done, awaiting acceptance) | `GameSpec` / `MultiAgentEnv`, elimination, teams, roles, layouts, `Units`, Dict observations, bootstrap on the learner, centralized critic, demo games |
| **SP3 Players, league, warm start** | scripted / frozen / external players, PFSP over snapshots (including opponent checkpoints in asymmetric games), per-agent warm start (`init`, kickstart, critic warm-up), top-k snapshot storage |
| **SP4 Selection and observability** (parallel with SP5) | match log, OpenSkill / Bradley–Terry, `colosseum tournament`, dashboard, snapshot ratings |
| **SP5 Distributed** (parallel with SP4) | hub and nodes, distributed league, wire format, per-machine weight cache, fault tolerance, `max_policy_lag`, K8s images |
| **SP6 Speed and extensions** | vectorized fast path for array envs and hundreds of seats, cuDNN RNN path, fast transformer unroll, GPU inference on workers, inference server, new algorithms |

### Open items parked during SP1 and SP2
- **SP4:**
  - paired, rating-based checkpoint selection (Bradley–Terry with bootstrap) instead of per-pair Wilson intervals.
- **SP5:**
  - distributed learner resume (`run-learner` never applies `training.resume_from`);
  - metrics hub / `metrics.jsonl` / WandB for the distributed roles (today: per-process logs only);
  - resource leak when `run-learner` setup fails after the trajectory server, weight-store client or drainer started;
  - one budget semantics for distributed mode (today `total_timesteps / num_workers` per worker; learner progress = `consumed_samples / total_timesteps`, which counts decisions, not env steps);
  - newest-wins weight-queue eviction unpickles the stale payload (CPU ∝ model size × workers) → per-machine weight cache;
  - the distributed learner's final checkpoint can be lost when the checkpoint drainer stops first and the 16-slot queue is full;
  - distributed league and games where one agent does not play every role.
- **SP6:**
  - normalizer statistics are updated before the loss forward, so the ratio is not exactly 1 at zero lag on early steps;
  - `WindowAttentionCore` rebuilds its masks every step (cache with the fast path);
  - BC windows are not episode-aligned;
  - observation trees are handled per seat in Python on the worker (vectorized per-role arrays for envs with hundreds of seats).
- **Tier 2 algorithms:** replay buffer for R2D2/DQN (the replay path would also need its own normalizer-stat and warm-up handling).
- **Minor, unscheduled:**
  - `--set` lists follow YAML 1.1, so `--set x=[1e-4,2]` keeps `1e-4` as a string (write `1.0e-4` in lists);
  - the main process builds a ratings snapshot on every monitor pass (negligible cost);
  - stale parked rollout buffers have no age limit (bounded by agents × envs × seats);
  - SharedMemory zero-copy weight store (raw `shared_memory` instead of queue payloads);
  - dynamic add/remove of agents and machines during a run;
  - config inheritance / profiles (`extends: base.yaml`).
- **Parked during SP2** (from the SP2 acceptance report, «Открытые пункты»):
  - the team_tag slow learning test takes the best of two runs (about 2 of 9 single runs settle into a passive draw-seeking policy) → SP3 (PFSP over snapshots / anti-passivity measures), so a single run can be required again;
  - `ratio_mode` at K=128 on unit_harvest: `joint` beat `per_unit` by ≈0.09–0.13 `share_vs_scripted` (seed 0; noisy); the spec default (`auto` = `per_unit` with `Units`) is kept — owner decision pending;
  - `unit_trace` auto with `Units` = `joint` was chosen on noisy data (run-to-run spread exceeds the rule's 0.05 margin; `joint` vs `none` undecided; `geo_mean` consistently worst);
  - budget overshoot and resume: a local run overshoots `total_timesteps` by about one second of env steps; a resume needs a budget above the checkpoint's `env_steps` to train (optional warning when a resume starts at or past the budget) → SP5 "one budget semantics";
  - distributed workers get identical per-env seeds across machines when `training.seed` is set (ruling PR-3) → SP5;
  - SP4: `system` `dropped_reward_episodes` sums only recently reporting workers (can decrease, can read 0 in the last record); make it a per-run monotonic counter.

### Next step
SP3 (players, league, warm start), after the owner accepts the SP2 acceptance report and SP2 is merged. Like SP1 and SP2, it starts with a brainstorm and a written spec before any plan or code (see Development Workflow). The owner still has to run `docs/GPU_CHECKS.md` on a CUDA machine.

## Development Workflow

- **One sub-project at a time**, in roadmap order (SP4 and SP5 may run in parallel). Each one goes: brainstorm → spec in `docs/superpowers/specs/` → implementation plan in `docs/superpowers/plans/<date>-<name>/` (overview with global constraints, interface contract and contract amendments, then parts with tasks) → implementation on a branch `spN-<name>` → acceptance report in `docs/superpowers/reports/` → merge.
- **`main` stays stable.** A sub-project branch is merged with `git merge --no-ff` only after the owner explicitly accepts the acceptance report, so one `git revert -m 1 <merge>` undoes it. Push the branch to `origin` regularly while working.
- **Implementation process (SP1):** one subagent per plan task (TDD, commit), a spec + quality review after every task, fix rounds with scoped re-reviews, then one whole-branch review with a single fix wave, then acceptance against the spec's §3 criteria.
- **Decisions taken on the owner's behalf** are recorded as rulings ("what — why — cost if wrong"). The SP1 list (58 rulings) is at the end of `docs/superpowers/reports/2026-10-08-sp1-acceptance.md`. Check it before changing an area SP1 touched, and revise a ruling explicitly (with the owner) rather than silently undoing it.
- **Frozen material:** `review/` is the pre-SP1 review snapshot and SP specs/plans are history; never edit them to match new code.
- **Dev machine and tests:**
  - setup: `scripts/setup-dev.sh` (CPU-only torch from the PyTorch CPU index; `--gpu` for CUDA);
  - CI suite: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (must pass with zero warnings), plus `-m slow` for the learning test and `.venv/bin/ruff check .`;
  - test conventions: no `__init__.py` under `tests/`; import support modules by bare name (`from game_helpers import ...`, `from cli_runner import ...`); test basenames and support-class names are unique; every file goes under `tmp_path`; integration runs start processes through `tests/cli_runner.py` (process-group cleanup);
  - at most 2 worker processes in tests (the benchmark may use 4); `OMP_NUM_THREADS=1`;
  - any config-schema change also updates `scripts/bench_throughput.py::_make_config` (pinned benchmark workload; guarded by `tests/unit/test_bench_throughput.py`).

## Tech Stack

- Python 3.11+
- PyTorch (training + inference)
- gRPC + protobuf (distributed communication)
- Docker + Kubernetes (orchestration)
- WandB (metrics)
- lz4 (compression)
- Pydantic v2 (configuration)
- Click (CLI)

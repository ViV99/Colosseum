# Colosseum — Distributed RL Training Framework for Competitive Bot Programming

## Project Vision

Reusable framework for competitive bot programming competitions (Lux AI, Neural MMO, CodeCraft, etc.).
The core problem: competitions are short, no time to rebuild RL infrastructure each time.
Colosseum provides the full pipeline: BC → RL → Self-Play → PFSP/League, distributed, modular, with dynamic agent and machine management.

Owner priorities (judge every design against them):
1. Inference / trajectory collection and training run on DIFFERENT machines with different hardware (CPU boxes, GPU boxes); machines and agents can be added or removed dynamically.
2. Maximal flexibility and simplicity, both for deployment and for configuring training scenarios.
3. Several warm-start options (offline BC, kickstarting, resume, etc.).
4. A flexible final stage: an arena where bots with different weights and architectures play, with scripted bots mixed in; convenient selection of the best; monitoring of all metrics and results.
5. Every game type: solo (score), 1v1 (turn-based and simultaneous), one bot controlling a team of units vs other teams, team vs team with several bots, 1 vs N (FFA or asymmetric roles).

## Research Documents

- `research/01_distributed_rl_architectures.md` — Distributed RL: GORILA, A3C, IMPALA, Ape-X, R2D2, SEED RL, Podracer, Sample Factory, PureJaxRL, PufferLib, Cleanba, GPU envs, V-trace math
- `research/02_competitive_rl_framework_research.md` — AlphaStar, OpenAI Five, AlphaZero/MuZero, Cicero, Pluribus, HoK; matchmaking (PFSP, PSRO, PBT, ELO/TrueSkill); BC/DAgger/GAIL/IRL/kickstarting; N-player games; OpenSpiel, PettingZoo, JaxMARL, Mava, EPyMARL; MAPPO/QMIX/MADDPG/HAPPO
- `research/rl_frameworks_survey.md` — 30+ frameworks: RLlib, SB3, CleanRL, TorchRL, Tianshou, Acme, Sample Factory, EnvPool, JaxMARL, Mava, EPyMARL, MARLlib, PettingZoo, OpenSpiel, Gymnax, Pgx, Brax, PufferLib
- `research/MULTI_AGENT_RL_RESEARCH.md` — survey of multi-agent / multiplayer RL: landmark systems, self-play methods, MARL paradigms, matchmaking and opponent modeling, BC/imitation for cold start, practical approaches for bot competitions
- `research/dialogue.txt` (Russian) — the owner's original brief for the project
- `research/sp2_game_apis.md` (Russian) — game structures of real competitions (Lux AI S1–S3, Orbit Wars, Halite/Kore, Hungry Geese, GRF, Neural MMO, Pommerman, MicroRTS, Generals, CodinGame, Battlecode, …) and existing multi-agent APIs (PettingZoo, RLlib, OpenSpiel, kaggle-environments, Melting Pot, JaxMARL, PufferLib); conclusions that shaped `GameSpec` / `MultiAgentEnv`
- `research/sp2_per_unit_ppo.md` (Russian) — what practitioners do for many units per bot: joint vs per-unit PPO ratios and clipping, V-trace weights in factorized action spaces, entropy normalization, per-unit vs team value, dead/absent units, GridNet vs entity lists

---

## AGREED Architecture Decisions

### Core Distributed Architecture: IMPALA-style Async Actor-Learner

- **CPU Workers** (N machines): run environments + local inference (policy forward pass on CPU; workers compute no values), collect trajectories in fixed-length chunks (T=128-512 slots), send to learners async
- **Learners** (M machines, usually with GPU but not required): receive trajectory chunks, compute every value (including bootstrap) with the current network, run backward pass + optimization, push updated weights to Weight Store
- **Weight Store**: shared storage for latest model weights per agent. Single-machine: shared memory (target; see Roadmap — today: newest-wins `mp.Queue` of numpy payloads per agent and worker). Multi-machine: dedicated service (today: the gRPC `serve-weight-store`) or Redis
- Workers periodically pull fresh weights from Weight Store (not every step — every `rollout.weight_sync_interval_sec` seconds)
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
  - LSTM/GRU (and windowed attention) cores for partial observability
  - Action masking for constrained action spaces
  - Many units per seat (`Units`): `ratio_mode` (`joint` / `per_unit`), `unit_trace` (scalar ρ for V-trace), `entropy_reduction`; per-decider diagnostics
  - Centralized critic: `global_state` reaches only the value path (`networks.critic_encoder_class`)
  - AMP (mixed precision) training support
  - This is what Sample Factory, PufferLib use; similar to OpenAI Five's approach

**Tier 2 (planned; see Roadmap — SP6 "new algorithms"):**
- **Rainbow DQN / R2D2** — off-policy, discrete actions
  - Prioritized replay buffer (Ape-X style distribution)
  - R2D2 = Rainbow + LSTM for partial observability
  - Good for turn-based games

**Tier 3 (later):**
- SAC (continuous), AlphaZero/MuZero (MCTS)

**Behavioral Cloning:**
- Offline BC: supervised learning from recorded trajectories (`-log_prob` of the recorded action for every distribution type, masks applied)
- Online BC (kickstarting): `loss = RL_loss + λ * KL(BC_teacher || policy)` (forward KL by default, `training.kickstart_kl`), λ decays over training; KL per decider over `act` slots; one global teacher `.pt` (`training.kickstart_teacher`, built from the agent's networks config; a teacher per agent is SP3)
- Both approaches available; BC phase runs before RL phase (`colosseum bc` writes `bc.pt`, then `training.resume_from: bc.pt` or `training.kickstart_teacher: bc.pt`)

### Framework: PyTorch
- JAX can be added later but not initial priority

### No Ray — Custom Distribution Layer

**Transport: gRPC**
- Control plane (coordinator ↔ workers/learners): gRPC unary RPCs (RegisterWorker, RequestMatch, ReportResult, etc.) (target; see Roadmap)
- Data plane (workers → learners): gRPC client streaming for trajectory chunks
- Weight sync: Weight Store service (gRPC or shared memory)
- Serialization: protobuf for metadata, raw bytes + lz4 compression for tensors (today: protobuf messages carry one blob = JSON skeleton + `np.savez` archive, optionally lz4, no pickle; the wire format is revised in SP5)

**Single-machine fallback:**
- multiprocessing.Queue for control
- multiprocessing.shared_memory for weights (zero-copy) (target; see Roadmap — today: newest-wins mp.Queue of numpy payloads)
- Shared memory ring buffer for trajectories (Sample Factory style) (target; see Roadmap — today: mp.Queue of numpy chunk payloads)
- Same interfaces, different transport backend

### Trajectory Handling

- Fixed-length rollout chunks of T slots of ONE agent (`rollout.chunk_length`, default 256); every collecting (env, seat) owns one buffer, parked across lineup changes
- Chunk slots are `act` (a decision), `boot` (an observation for the learner's bootstrap value) or `pad`; episode boundaries are handled within chunks (traces stop at `terminal` and `boot`), and the learner computes every value with its current network
- For LSTM: hidden state carried across chunks within episode, reset after `reset_after` slots (terminal `act`, truncation `boot`, `pad`). Stored state approach: the chunk's `initial_state` is saved at chunk start, replayed during training.
- One chunk ≈ 300KB per agent per chunk (for obs_dim=256, T=256)
- At 100 workers × 10 chunks/sec = 300 MB/sec total throughput — manageable for 10Gbit or shared mem

---

## AGREED Training Pipeline (Competition Lifecycle)

### Phase 1: Setup
- User writes: `MyGame(MultiAgentEnv)` with a `GameSpec` (roles, layouts, teams) — see `docs/ENV_GUIDE.md`
- User writes: `MyEncoder(BaseEncoder)` — observation tree → latent (`EncoderOutput(latent, aux)`; `aux` carries per-unit features)
- User writes: `MyPolicy(BasePolicy)` + `MyValue(BaseValue)` — neural network heads; optional `MyCritic(BaseCriticEncoder)` over `global_state` (value path only)
- User selects: algorithm (APPO, DQN, etc.; only APPO exists today) + hyperparameters via config

### Phase 2: Behavioral Cloning (optional)
- **Offline BC**: load trajectories from disk (recorded games from other players, etc.)
- **Online BC**: scripted agent generates trajectories, neural agent learns from them (target; see Roadmap — scripted players are SP3; today: kickstarting from a neural teacher `.pt`)
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
- PFSP priority function: `f(win_rate) = (1 - win_rate)^p` (focus on hard opponents; `matchmaking.pfsp_exponent`, implemented, per layout, over other agents' latest weights) or `f(x) = x*(1-x)` (balanced) (target; see Roadmap — the balanced variant and PFSP over snapshots are SP3)
- ELO/TrueSkill tracking for all agents (ELO implemented; TrueSkill-type ratings are a target; see Roadmap)
- Dynamic: add/remove agents and machines at any time without stopping (target; see Roadmap)

### Eval Mode
- Inference-only matchups between any set of checkpoints / `.pt` state dicts (`colosseum.eval`); no training
- Same engine as training rollouts (`MatchRunner`): one model `State` per (env, seat), `StepResult.acting` and action masks
- Lineups via `schedule_lineups` (per layout): for a pair (a, b), match m gives team i as a whole to `(a, b)[(i + m) % 2]` (or to the pair's other agent when that one does not play every role of the team); an orientation the roles forbid is replaced by the other, and a pair is dropped from a layout only when both are impossible; one agent fills every team; one-team layouts play each agent's homogeneous team plus mixed compositions (cross-play); `--num-matches` is per pair (or composition) and layout
- Reports per layout and role: `wdl` — W/D/L, win rate and score with 95% Wilson intervals, per side; `rank` — mean rank with CI, first-place share, pairwise matrix; `score` — mean score with CI, cross-play table
- JSON output with `--output`; architecture of each checkpoint rebuilt from its `meta.json` (`.pt` files from `--config`)
- Command: `colosseum eval -c config.yaml -a A=runs/x/checkpoints/agent_0/ckpt_v100 -a B=runs/x/checkpoints/agent_0/ckpt_v200 --num-matches 1000 --output result.json` (`--num-matches` is per pair (or composition) and layout; the architecture comes from each checkpoint's `meta.json`)

---

## AGREED Detailed Design Decisions

### Agent Pool / Registry

Agents in pool can be:
- **TrainableAgent**: has its own learner, actively training, generates checkpoints. Can have completely different architecture, encoder, algorithm, hyperparameters from other agents.
- **FrozenCheckpoint**: static weights loaded from disk, used as sparring partner, no training (target; see Roadmap — SP3; today only an agent's own checkpoints play as opponents)
- **ScriptedAgent**: rule-based bot, no neural network (target; see Roadmap — SP3; in eval and tests any in-process `PolicyModel` can take a seat via `play_lineups`)

Multiple trainable agents can coexist with different architectures, different RL algorithms, different encoders — they are fully independent (per-agent `agents.<id>.{roles,networks,algorithm,learner}`). This allows comparing approaches side by side.

### Learner-Agent Mapping

- One learner process = one trainable agent
- Multiple learner processes can share one physical machine (GPU time-sharing for small models)
- Load balancing: if 10 agents and 2 GPU machines → distribute ~5 learner processes per machine (target; see Roadmap — SP5; today all learners run on the training machine, or one `run-learner` per agent by hand)
- Learner machines don't require GPU (modern AI-focused CPUs work too) — "learner" just means "machine that does backward pass + optimization"

### Worker Parallelism (INTRA-machine)

Each CPU worker machine runs:
- Multiple worker PROCESSES, each managing multiple ENVs
- Each process does: step multiple envs, run inference (batch forward pass), collect trajectories
- Vectorized env execution within each process (like Gymnasium's VectorEnv; `rollout.vec_env: sync | subprocess`)
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

- Save checkpoints every K training steps (configurable per agent; today one global `checkpoint.interval`)
- FIFO pool per agent (keep last N checkpoints, configurable; top-k snapshot storage is SP3)
- Checkpoints stored on shared filesystem or object store (today: `runs/<name>/checkpoints/<agent>/ckpt_v<N>/`)
- Checkpoint = state_dict + optimizer_state + training_step + metrics snapshot (today: `model.pt`, `trainer_state.pt` with optimizer, LR progress, AMP scaler, kickstart and counters, `meta.json` with `policy_version`, `env_steps`, roles and role signature; no metrics snapshot)

### Orchestration

- (target; see Roadmap — SP5: `deployment/` has Dockerfiles, docker-compose and K8s manifests, untested; there is no coordinator service yet)
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

**Mode A — Offline BC (simpler, implement first; implemented as `colosseum bc`):**
1. Scripted agent plays games, trajectories saved to disk (today the user's own script writes the `.pt` data files; format in `bc/offline_bc.py`)
2. OR: download external replay files
3. Neural agent trains supervised on these trajectories
4. Standard supervised learning loop, no env interaction needed

**Mode B — Online BC / DAgger-like** (target; see Roadmap — a scripted expert needs SP3's scripted players; today only kickstarting from a frozen neural teacher):
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
- Dashboard: real-time view of training progress, agent pool, active workers/learners (target; see Roadmap — SP4; today: console progress, `metrics.jsonl`, optional WandB)

---

## Modularity Requirements

Everything inheritable/pluggable:

```python
# User implements these per competition:
class MyGame(MultiAgentEnv): ...    # GameSpec + reset/step -> StepResult (docs/ENV_GUIDE.md)
class MyEncoder(BaseEncoder): ...   # obs tree → EncoderOutput(latent, aux)
class MyPolicy(BasePolicy): ...     # latent → action distribution
class MyValue(BaseValue): ...       # latent → scalar value
class MyCritic(BaseCriticEncoder): ...  # optional: global_state → value-path latent

# Framework provides these (swappable; target names — today: APPO via `algorithm.algorithm_class`,
# one LineupMatchmaker for self-play and league, GRPCTransport / LocalTransport):
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

State after SP2 (game model; developed on branch `sp2-game-model`, accepted by the owner and merged into `main` on 2026-10-09). Test suite: **1248** fast tests, **9** slow, **18** GPU-only (`pytest -m "not gpu and not slow"` is the CI suite).
- SP2 records: spec `docs/superpowers/specs/2026-10-08-sp2-game-model-design.md` (Russian); plan `docs/superpowers/plans/2026-10-08-sp2-game-model/` (`00-overview.md` has the interface contract and amendments R1–R18); acceptance report `docs/superpowers/reports/2026-10-09-sp2-acceptance.md` (criteria, measurements, every ruling, open items, all deferred task minors in its appendix); final review, fix wave and FIX-1/FIX-2 records `docs/superpowers/reports/2026-10-09-sp2-review/`.
- SP1 records: `docs/superpowers/reports/2026-10-08-sp1-acceptance.md`; the pre-SP1 review `review/README.md` is a frozen snapshot (never edit it).

Commands (run from the repo root after `scripts/setup-dev.sh`; the cwd is put on `sys.path`, also for spawned children):
- `colosseum validate -c cfg.yaml [--set k=v ...]`: `GameSpec` and config checks (roles, matchmaking), `reset` of every enabled layout and random legal steps with full contract checks, `step` and `unroll` of every agent's model, the kickstart teacher.
- `colosseum train -c cfg.yaml [--set k=v ...]`: single-machine training into `runs/<name>/`.
- `colosseum eval -c cfg.yaml -a name=path [-a ...] [--layout L ...] --num-matches N --output result.json [--num-envs E] [--deterministic] [--seed S]`.
- `colosseum bc -c cfg.yaml --agent A --data <file-or-dir> --output bc.pt [--epochs E] [--batch-size B] [--lr LR] [--seq-len L]`.
- Distributed (limited): `colosseum serve-weight-store --port P [--max-message-mb M]`, `colosseum run-learner -c cfg.yaml --agent A --traj-port P --weight-store host:port`, `colosseum run-workers -c cfg.yaml --weight-store host:port -l A=host:port`.
- Measurements: `scripts/bench_throughput.py`, `scripts/units_experiment.py`, `scripts/measure_global_state.py` (results in `docs/benchmarks.md`).

### Works (single machine)
- **Game model (SP2):**
  - env contract `GameSpec` (roles, layouts = lists of seats with roles and teams) + `MultiAgentEnv` + `StepResult` (`acting`, `terminated`, `truncated` + `final_obs`, `global_state`, `Outcome`); `EpisodeTracker` checks every rule and raises `EnvContractError` with "worker, env, seat, episode step, layout";
  - outcome kind from the number of teams (1 → `score`, 2 → `wdl`, ≥3 → `rank`); default outcome = mean of a team's seat returns;
  - seat lifecycle: empty / live / eliminated; a dead teammate is removed from `acting` but keeps team rewards; steps without decisions bounded by `env.max_idle_steps`;
  - observation, action, mask and `global_state` trees with preserved dtypes (`uint8` reaches the model); bool masks only; entity lists with masks; `Units(max_units, per_unit, only_if)` with `Discrete`/`MultiDiscrete`/`Box`/`Dict` components; deciders (each unit + the non-unit part).
- **Training loop:** IMPALA-style; one `MatchRunner` core for training (`RolloutLoop`) and eval. Workers infer only the policy; chunk v2 with `act`/`boot`/`pad` slots; the learner computes every value (bootstrap from `boot` slots and `final_obs`) with its current network. Only numpy payloads cross process boundaries. Every learner update uses exactly `learner.batch_chunks` chunks; `training.total_timesteps` is a global env-step budget.
- **Algorithm:** APPO with V-trace over slots; `ratio_mode` (`joint` / `per_unit`; `auto` = `per_unit` with `Units`), `unit_trace` (`joint` / `geo_mean` / `none`; `auto` = `joint` for every action, decided by the units experiment), `entropy_reduction`; with one decider every mode collapses to `joint` (except an explicit `unit_trace: none`, ruling PR-1); every loss and metric reduces over `act` slots; per-unit diagnostics (`clip_fraction_joint`, `ess`, `log_rho_abs_p95`, `deciders_valid_mean`, ...). Centralized critic: `networks.critic_encoder_class` over `global_state`, value path only.
- **Models:** `PolicyModel` protocol (`step` for the policy path, `unroll` for the learner, `with_value=False` for BC and the kickstart teacher); `ComposedModel(encoder, core, policy, value, critic_encoder)` with `EncoderOutput(latent, aux)` for per-unit features; cores `NoCore`, `LSTMCore`, `GRUCore`, `WindowAttentionCore`; `UnitsHead`, `gridnet_to_units`; `NormalizeObs` on a leaf path. The learner reproduces the worker's joint and per-unit log-probs for all four cores (contract tests).
- **League:** lineups (layout + seat assignments); owner rotation; layout weights; self-play / arena with PFSP per layout; teams with a core and `teammates: self | mixed`; roles per agent (`agents.<id>.roles`), asymmetric games with one agent per role; seat permutation within equal role compositions. Ratings per layout (`ratings.json` = `{"env_steps", "layouts": {...}}`): team-pair ELO, win-rate matrix, `wr_vs_past`, `ScoreTracker` and cross-play for one-team layouts. FIFO checkpoint pool with roles and role signature in `meta.json`; strict resume.
- **Eval:** `MatchRunner`-based; CLI lineups per spec block 8; reports for `wdl` (Wilson intervals, per side), `rank` (mean rank, first places, pairwise matrix), `score` (mean, cross-play), by role; in-process `play_lineups` with any `PolicyModel`.
- **BC / kickstart:** BC on observation/action trees (`-log_prob`, mean over valid deciders with `Units`; an illegal masked expert action is a `DataError`); one global kickstart teacher with a per-decider KL.
- **Demo games** (`examples/`, configs in `configs/examples/`): `coin_grid` (solo), `tic_tac_toe` (1v1 turn-based; also `tic_tac_toe_attention.yaml`, `tic_tac_toe_multi.yaml`), `unit_harvest` (units), `team_tag` (2v2, `global_state`), `tron` (FFA 2–4, elimination), `predator_prey` (asymmetric roles), `coop_buttons` (cooperative); reference examples `space_miners` (Box2D) and `composite_action` (chase). Each demo game has a slow learning test against random players (`tests/learning/test_demo_learning_slow.py`; team_tag takes the better of two runs); fast learning tests cover a Units bandit and a cooperative bandit. `docs/ENV_GUIDE.md` (Russian) explains how to write an env for every game type.
- **Observability:** `runs/<name>/` (`run.dir`/`run.name`; default name `<config stem>-<YYYYmmdd-HHMMSS>`; an existing explicit name is refused) with `config.resolved.yaml`, `logs/` (`main`, `learner-<agent>`, `worker-<i>`, `worker-<i>-env<k>`), `metrics.jsonl` (`train` / `episodes` / `ratings` / `system`, with per-layout and per-role breakdowns; `system` includes `dropped_reward_episodes`), `ratings.json`, `checkpoints/<agent>/ckpt_v<N>/{model.pt, trainer_state.pt, meta.json}`; console progress per agent; optional WandB. A seat that never acts in an episode loses that episode's rewards: one WARNING per worker, then counted.
- **Config:** Pydantic with `extra="forbid"`; partial agent overrides deep-merged onto the global sections (`agents.<id>.{roles,networks,algorithm,learner}`); agent ids `[A-Za-z0-9_][A-Za-z0-9_-]*`, not `ratings`/`system`/`episodes`/`train`; `--set` with YAML values for `train` / `validate` / `run-learner` / `run-workers`; `self_play.*` became `matchmaking.*` + `checkpoint.*`, `env.num_players` and `training.phase` are gone.
- **Lifecycle:** exit codes 0 (budget reached) / 1 (config error or a dead child) / 130 (SIGINT, also during startup and for `serve-weight-store`) / 143 (SIGTERM); `eval` also 2 for bad arguments. Children ignore SIGINT and die with the parent; a dead child stops the run with a pointer to its log; final checkpoints on every stop within `SHUTDOWN_GRACE_SEC = 7`; no process alive after 10 s. Explicit resume is strict (`ConfigError` on a missing or malformed `meta.json` or a role-signature mismatch; SP1 checkpoints have no role signature and cannot be resumed).
- **Throughput:** 4848 / 9298 / 13702 env steps/s with 1 / 2 / 4 workers on tic-tac-toe (104 % / 98 % / 128 % of SP1; `docs/benchmarks.md`).

### Partial
- **Distributed mode** (`serve-weight-store`, `run-learner`, `run-workers`) runs on chunk v2 for games where every agent plays every role, latest weights only: no coordinator, league, ratings, `metrics.jsonl` or WandB; per-worker budgets; `run-learner` ignores `training.resume_from`.
- **`deployment/`** (Docker, K8s) is not tested.
- **Unused, kept for SP5:** `transport/local.py::LocalTransport`, `weight_store/shared_memory.py::SharedMemoryWeightStore`, `transport.mode` / `transport.grpc_port`.
- **GPU:** never run on CUDA. All CUDA paths (learner device, AMP, `pin_memory`, kickstart/BC, tree batches and `Units` distributions on CUDA) are covered only by `gpu`-marked tests; see `docs/GPU_CHECKS.md`.

### Not implemented (see Roadmap)
- Scripted, frozen and external players; PFSP over snapshots; per-agent warm start and kickstart teacher; top-k snapshot storage (SP3).
- Not planned at all (SP2 spec §2): one agent on roles with different spaces; per-unit rewards, values or recurrent state; built-in autoregression between action components (a custom `Distribution` can do it); match series as a framework entity (best-of-N lives inside the env); MCTS controllers and communication channels between bots; canonical perspective and augmentations (the env's job).
- A PettingZoo adapter (later, if needed).
- OpenSkill / Bradley–Terry ratings, a tournament command, match log, dashboard (SP4).
- Off-policy algorithms (R2D2/DQN, replay buffer), SAC, AlphaZero/MuZero (SP6 "new algorithms" or later).
- GPU inference on workers; a vectorized worker path for envs with hundreds of seats (SP6).
- Shared-memory weight store and trajectory ring buffer; adding or removing agents and machines during a run; gRPC control plane, distributed league and asymmetric games across machines (SP5).

## Roadmap

| Sub-project | Scope |
|---|---|
| **SP1 Foundation and stabilization** (done) | correct, observable, robust single-machine training; `PolicyModel` protocol |
| **SP2 Game model** (done) | `GameSpec` / `MultiAgentEnv`, elimination, teams, roles, layouts, `Units`, Dict observations, bootstrap on the learner, centralized critic, demo games |
| **SP3 Players, league, warm start** | scripted / frozen / external players (also as BC data recorders and DAgger experts), PFSP over snapshots (including opponent checkpoints in asymmetric games; balanced `x(1-x)` weighting), anti-passivity in self-play (so team_tag can again pass in one run), per-agent warm start (`init`, kickstart teacher per agent, critic warm-up), top-k snapshot storage |
| **SP4 Selection and observability** (parallel with SP5) | match log, OpenSkill / Bradley–Terry, `colosseum tournament`, dashboard, snapshot ratings |
| **SP5 Distributed** (parallel with SP4) | hub and nodes, distributed league (including asymmetric games and agents that do not play every role), adding / removing agents and machines during a run, shared-memory weight store and trajectory ring buffer, wire format, per-machine weight cache, fault tolerance, `max_policy_lag`, K8s images |
| **SP6 Speed and extensions** | vectorized fast path for array envs and hundreds of seats (Neural MMO), cuDNN RNN path, fast transformer unroll, GPU inference on workers, inference server, new algorithms |

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
  - config inheritance / profiles (`extends: base.yaml`);
  - a PettingZoo adapter (later, if needed; SP2 spec §2).
- **Parked during SP2** (from the SP2 acceptance report, «Открытые пункты»):
  - the team_tag slow learning test takes the best of two runs (about 2 of 9 single runs settle into a passive draw-seeking policy) → SP3 (PFSP over snapshots / anti-passivity measures), so a single run can be required again;
  - `ratio_mode` at K=128 on unit_harvest: `joint` beat `per_unit` by ≈0.09–0.13 `share_vs_scripted` (seed 0; noisy); the spec default (`auto` = `per_unit` with `Units`) stands — open owner question (Next step);
  - `unit_trace` auto with `Units` = `joint` was chosen on noisy data (run-to-run spread exceeds the rule's 0.05 margin; `joint` vs `none` undecided; `geo_mean` consistently worst) — open owner question (Next step);
  - budget overshoot and resume: a local run overshoots `total_timesteps` by about one second of env steps; a resume needs a budget above the checkpoint's `env_steps` to train (optional warning when a resume starts at or past the budget) → SP5 "one budget semantics";
  - distributed workers get identical per-env seeds across machines when `training.seed` is set (a consequence of ruling PR-3: seeds are `training.seed + worker_id` on every machine) → SP5;
  - SP4: `system` `dropped_reward_episodes` sums only recently reporting workers (can decrease, can read 0 in the last record); make it a per-run monotonic counter.
- **SP2 final-review Minors worth knowing** (full lists: the SP2 report's «Финальное ревью ветки» and appendix, and `docs/superpowers/reports/2026-10-09-sp2-review/`):
  - SP3: `coordinator/agent_pool.py` keeps unused legacy (`register_frozen` / `register_scripted`, `AgentHandle.elo` / `win_rates`; no caller) — reuse it for SP3 players or delete it; `units_component_valid` is the single action-legality gate and can check scripted bots' actions; owner rotation (`Coordinator.generate_lineups`, `src/colosseum/coordinator/coordinator.py:98-106`) relies on `AgentPool.list_trainable()` returning the trainable agents in config order (T5.3) — keep that order when frozen or scripted entries join the pool;
  - SP5: with several agents the distributed roles silently ignore `matchmaking.mode: league`, `teammates: mixed` and `shuffle_seats` (warn or `ConfigError`); the learner trusts slot payload values from a peer (finite `behavior_logp`, zero rewards on `boot`/`pad`), and a chunk with a wrong observation shape fails only in the learner forward; `distributed_setup` duplicates `setup_run`'s spec/roles sequence;
  - GPU checks: Gaussian log-prob and entropy run in fp16 under AMP (also inside `Units`); compare per-decider log-ratios with fp32 on a Box-in-`Units` AMP step;
  - SP6: `unit_log_prob` is computed twice per step (`act` and APPO's evaluate), the `Units` gate table is rebuilt per call;
  - unscheduled: `global_state` is stored and sent but unused when the agent has no critic encoder (a `validate` warning); `role_signature` ignores `n` / `nvec` of discrete observation leaves; all seats eliminated without `episode_over` is reported only after `max_idle_steps` idle steps; "masked units give no NaN gradient" holds only for finite parameters; `MatchRunner` trusts that every seat's role is one its agent plays.

### Next step
SP3 (players, league, warm start; branch `sp3-<name>`). Like SP1 and SP2, it starts with a brainstorm and a written spec before any plan or code (see Development Workflow); read the SP3 row, «Parked during SP2» and the SP2 rulings first.
- Design inputs for SP3 (pre-SP1 material: re-check every finding against the current code):
  - `review/README.md` §6.3 (player model and league: `Trainable` / `Snapshot` / `Scripted` / `External`, where External = a foreign `.pt` + a network description; a policy pool in the worker; a matchmaker over snapshots), §6.4 (ratings and best-agent selection; snapshot storage `keep_last` / `keep_every` / top-k / `pinned`), §6.5 (warm start: per-agent `init`, `critic_warmup_steps`, kickstart from any policy, `colosseum record --bot X`);
  - `review/code/04_league.md` «Proposed design» (config sketch); `review/research/B_league_rating_warmstart.md` (research behind it);
  - SP2 spec items handed to SP3: §2 «Вне рамок» (players, PFSP over snapshots incl. opponent checkpoints in asymmetric games, per-agent `init` and critic warm-up, top-k snapshots; BC and kickstart were only ported); block 5 (`MatchRunner` takes the caller's (agent, network) → `PolicyModel` map, so a scripted player can be a `PolicyModel`); block 6 (a kickstart teacher per agent); §8 (only the minimum of league with teams, roles and layouts is in SP2); the team_tag single-run requirement (acceptance report, open item 1).
- Open owner questions (no explicit answer at the SP2 acceptance, so the defaults stand); decide them at the SP3 brainstorm:
  - team_tag slow test: best of two independent runs (threshold 0.80 unchanged), or another check / an SP3 anti-passivity fix;
  - `ratio_mode: auto` = `per_unit` with `Units` (spec default), or `joint` after a multi-seed check on a real game;
  - `unit_trace: auto` = `joint` (chosen on noisy data; `none` is within the spread);
  - backward compatibility in SP3: SP2 had none (owner's SP2 instruction); confirm the same for SP3 (SP2 configs and checkpoints may break), or keep SP2 checkpoints resumable / usable as frozen players.
- GPU checks still pending: the owner has to run `docs/GPU_CHECKS.md` (18 tests) on a CUDA machine.

## Development Workflow

- **One sub-project at a time**, in roadmap order (SP4 and SP5 may run in parallel). Each one goes: brainstorm → spec in `docs/superpowers/specs/` → implementation plan in `docs/superpowers/plans/<date>-<name>/` (overview with global constraints, interface contract and contract amendments, then parts with tasks) → implementation on a branch `spN-<name>` → acceptance report in `docs/superpowers/reports/` → merge.
- **`main` stays stable.** A sub-project branch is merged with `git merge --no-ff` only after the owner explicitly accepts the acceptance report, so one `git revert -m 1 <merge>` undoes it. Push the branch to `origin` regularly while working.
- **Implementation process (SP1):** one subagent per plan task (TDD, commit), a spec + quality review after every task, fix rounds with scoped re-reviews, then one whole-branch review with a single fix wave, then acceptance against the spec's §3 criteria.
- **SP2 was executed the same way**, with a pre-flight consistency scan of the plan (rulings P1–P22) and a shadow package `colosseum.sp2` built next to the old code and overlaid onto `colosseum.*` in task T7.3 (the shadow is gone now); an Opus/high implementer and an Opus/high reviewer per task (owner's instruction), three area reviewers for the final whole-branch review, then one fix wave and a scoped re-review.
- **The SDD workspace** (`.superpowers/sdd/<plan>/`: briefs, ledger `progress.md`, review packages, reports) is git-ignored scratch and is deleted after the merge; anything durable (rulings, deferred items, review records) goes into `docs/` before the merge. The loop, the instruction templates, `brief.sh` and the ledger line formats are in `docs/superpowers/process/` (at most 5 fix rounds per task; implementer status `DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT`).
- **Decisions taken on the owner's behalf** are recorded as rulings ("what — why — cost if wrong"). The SP1 list (58 rulings) is at the end of `docs/superpowers/reports/2026-10-08-sp1-acceptance.md`; the SP2 list (plan-stage PR-1..PR-5, pre-flight P1–P22, the 42 ledger rulings, the T8.3–T8.4 rulings) is at the end of `docs/superpowers/reports/2026-10-09-sp2-acceptance.md`. Check both lists before changing an area SP1/SP2 touched, and revise a ruling explicitly (with the owner) rather than silently undoing it.
- **Frozen material:** `review/` is the pre-SP1 review snapshot and SP specs/plans are history; never edit them to match new code.
- **Dev machine and tests:**
  - setup: `scripts/setup-dev.sh` (CPU-only torch from the PyTorch CPU index; `--gpu` for CUDA);
  - CI suite: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (must pass with zero warnings) and `.venv/bin/ruff check .`;
  - slow suite: `.venv/bin/python -m pytest -m slow -v` — 9 tests (the 7 demo-game learning tests in `tests/learning/test_demo_learning_slow.py` and 2 torch.compile checks in `tests/unit/test_sp2_performance.py`), 11–16 min; run it at acceptance and after any change to the training math, the worker or the learner;
  - test conventions: no `__init__.py` under `tests/`; import support modules by bare name (`from game_helpers import ...`, `from cli_runner import ...`); test basenames and support-class names are unique; every file goes under `tmp_path`; integration runs start processes through `tests/cli_runner.py` (process-group cleanup);
  - at most 2 worker processes in tests (the benchmark may use 4); `OMP_NUM_THREADS=1`;
  - any config-schema change also updates `scripts/bench_throughput.py::_make_config` (pinned benchmark workload; guarded by `tests/unit/test_bench_throughput.py`).
- **Conventions:** commits use conventional prefixes (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`, `perf:`) and **never** carry attribution or `Co-Authored-By` lines, in commits or PRs; push after every task. Language: code, comments, docstrings, log messages, `README.md` and `CLAUDE.md` in English; specs, acceptance reports and the project docs `docs/ENV_GUIDE.md`, `docs/benchmarks.md`, `docs/GPU_CHECKS.md` in Russian (plans so far in English).

## Working with the owner

- The owner writes in Russian; reply in Russian. Docs follow the language rules above.
- Long reports (acceptance, reviews) go on a private, phone-readable artifact page, with a short Russian summary in chat.
- Specs are approved section by section in the brainstorm; no code is written before the spec is approved.
- In autonomous runs ("работай автономно"), keep going without asking. Decide conflicts as rulings ("what — why — cost if wrong") in the ledger and hand over the full list at the end. Stop only for destructive, security-sensitive or outward actions; a merge into `main` always needs the owner's explicit "yes".
- Delegation: implementation and review subagents run on Opus with high effort (implementers never dispatch subagents); web research and surveys on Sonnet with high effort; the main session orchestrates and makes the key decisions.

## Tech Stack

- Python 3.11+
- PyTorch (training + inference)
- gRPC + protobuf (distributed communication)
- Docker + Kubernetes (orchestration)
- WandB (metrics)
- lz4 (compression)
- Pydantic v2 (configuration)
- Click (CLI)

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
- Offline BC: supervised learning from recorded trajectories (cross-entropy/MSE loss)
- Online BC (kickstarting): `loss = RL_loss + λ * KL(BC_teacher || policy)` (forward KL by default, `training.kickstart_kl`), λ decays over training
- Both approaches available; BC phase runs before RL phase

### Framework: PyTorch
- JAX can be added later but not initial priority

### No Ray — Custom Distribution Layer

**Transport: gRPC**
- Control plane (coordinator ↔ workers/learners): gRPC unary RPCs (RegisterWorker, RequestMatch, ReportResult, etc.)
- Data plane (workers → learners): gRPC client streaming for trajectory chunks
- Weight sync: Weight Store service (gRPC or shared memory)
- Serialization: protobuf for metadata, raw bytes + lz4 compression for tensors

**Single-machine fallback:**
- multiprocessing.Queue for control
- multiprocessing.shared_memory for weights (zero-copy)
- Shared memory ring buffer for trajectories (Sample Factory style)
- Same interfaces, different transport backend

### Trajectory Handling

- Fixed-length rollout chunks of T steps (T=128-512, configurable)
- Episode boundaries handled within chunks (V-trace traces are cut at done, new episode starts in same chunk)
- For LSTM: hidden state carried across chunks within episode, reset at done. Stored state approach: h_init saved at chunk start, replayed during training.
- One chunk ≈ 300KB per agent per chunk (for obs_dim=256, T=256)
- At 100 workers × 10 chunks/sec = 300 MB/sec total throughput — manageable for 10Gbit or shared mem

---

## AGREED Training Pipeline (Competition Lifecycle)

### Phase 1: Setup
- User writes: `MyGameEnv(BaseEnv)` — Gymnasium-compatible, supports 1-N players
- User writes: `MyEncoder(BaseEncoder)` — game state → tensor
- User writes: `MyPolicy(BasePolicy)` + `MyValue(BaseValue)` — neural network heads
- User selects: algorithm (APPO, DQN, etc.) + hyperparameters via config

### Phase 2: Behavioral Cloning (optional)
- **Offline BC**: load trajectories from disk (recorded games from other players, etc.)
- **Online BC**: scripted agent generates trajectories, neural agent learns from them
- Training: supervised cross-entropy/MSE loss
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
- If arena match has fewer agents than player slots → duplicate agents to fill
- ALL trainable agents in arena match collect trajectories → one env step feeds multiple learners
- PFSP priority function: `f(win_rate) = (1 - win_rate)^p` (focus on hard opponents) or `f(x) = x*(1-x)` (balanced)
- ELO/TrueSkill tracking for all agents
- Dynamic: add/remove agents and machines at any time without stopping

### Eval Mode
- Inference-only matchups between any set of checkpoints/agents
- No training, just collect win rates with confidence intervals
- Supports N-player games with round-robin agent assignment and pairwise result extraction
- Command: `colosseum eval -c cfg.yaml -a A=runs/<name>/checkpoints/<agent>/ckpt_v<N> -a B=runs/<name>/checkpoints/<agent>/ckpt_v<M> --num-matches 1000 --output result.json`

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
- Player slots mapped to agents via `slot_agent_map[env_idx][player_idx] -> agent_id`
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
class MyEnv(BaseEnv): ...           # Gymnasium-like interface, 1-N players
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

All 6 milestones + 5 improvement phases + composite actions + code cleanup fully implemented. **148 tests passing, 2 skipped.**

### Milestone 1: Core + Single-Machine APPO (MVP) — DONE

Working end-to-end IMPALA-style training loop on one machine.

| Component | File | Description |
|-----------|------|-------------|
| Core types | `core/types.py` | TrajectoryChunk (with lstm_hidden, action_masks), PlayerSlot, MatchConfig, MatchResult, WeightPayload |
| ActionSpec | `core/action_spec.py` | Flat ↔ structured codec for composite action spaces (Dict, Tuple, MultiDiscrete) |
| Config | `core/config.py` | Pydantic v2 models, YAML loading, per-agent overrides, `load_config()` |
| Registry | `core/registry.py` | `import_class()`, `build_network()` (with optional LSTM/GRU trunk) |
| BaseEnv | `envs/base_env.py` | N-player symmetric environment ABC |
| VectorEnv | `envs/vec_env.py` | K envs stepped sequentially, auto-reset |
| Networks | `networks/base.py` | BaseEncoder (with `latent_dim`), BasePolicy, BaseValue ABCs |
| Distributions | `networks/distributions.py` | CategoricalDist (with action masking), DiagGaussianDist, CompositeDist (multi-head for Dict/Tuple/MultiDiscrete) |
| ActorCritic | `networks/actor_critic.py` | `act()` (4-tuple with hidden), `evaluate_actions()`, `evaluate_actions_recurrent()`, optional LSTM/GRU trunk |
| V-trace | `algorithms/vtrace.py` | Pure `compute_vtrace()` function (numerically stabilized) |
| APPO | `algorithms/appo.py` | V-trace + PPO clip + entropy, minibatch, AMP, recurrent training path, kickstart integration |
| Weight Store | `weight_store/shared_memory.py` | InMemoryWeightStore, SharedMemoryWeightStore |
| Transport | `transport/local.py` | LocalTransport (mp.Queue per agent) |
| Worker | `worker/rollout_worker.py` | Multi-agent routing, network pool, LSTM hidden state tracking, action masking, pre-allocated buffers |
| Learner | `learner/learner.py` | Receive chunks, train, push weights (configurable interval), checkpoint snapshots, pin memory |
| Coordinator | `coordinator/coordinator.py` | Agent pool + matchmaking + checkpoint management + results feedback |
| WandB | `metrics/wandb_logger.py` | WandB integration |
| Launcher | `launcher.py` | Multi-agent training orchestration, monitor loop with results processing |
| CLI | `cli.py` | `colosseum train`, `bc`, `eval`, `serve-weight-store`, `serve-trajectory` |
| Example | `examples/tic_tac_toe/` | TicTacToeEnv, Encoder, Policy, Value (simple Discrete actions) |
| Example | `examples/composite_action/` | ChaseEnv with Dict action space, CompositeDist policy |

### Milestone 2: Self-Play + Checkpoint Management — DONE

| Component | File | Description |
|-----------|------|-------------|
| CheckpointManager | `coordinator/checkpoint_manager.py` | Save/load/FIFO per agent, metadata |
| AgentPool | `coordinator/agent_pool.py` | Registry: trainable, frozen, scripted agents |
| Matchmakers | `coordinator/matchmaker.py` | SimpleSelfPlayMatchmaker, SelfPlayMatchmaker, PFSPMatchmaker |
| Launcher integration | `launcher.py` | Match config generation, collect_mask, checkpoint saves |

### Milestone 3: Behavioral Cloning — DONE

| Component | File | Description |
|-----------|------|-------------|
| Offline BC | `bc/offline_bc.py` | OfflineBCTrainer: load .pt data, supervised CE/MSE loss |
| Kickstart | `bc/kickstart.py` | KL(teacher \|\| student) by default (configurable), masked, unrolled teacher, linear lambda decay |
| APPO integration | `algorithms/appo.py` | Optional `kickstart` param adds KL loss to total |
| CLI | `cli.py` | `colosseum bc --config <path> --data <dir> --output <path>` |

### Milestone 4: PFSP / League Training — DONE

| Component | File | Description |
|-----------|------|-------------|
| ELO | `coordinator/ratings.py` | EloRating: pairwise updates, K-factor |
| Win Rates | `coordinator/ratings.py` | WinRateTracker: pairwise tracking, matrix |
| PFSPMatchmaker | `coordinator/matchmaker.py` | PFSP priority `(1-wr)^p`, solo/arena match split |
| Coordinator | `coordinator/coordinator.py` | Match result reporting, ELO/WR updates, league phase |

### Milestone 5: gRPC Distribution — DONE

| Component | File | Description |
|-----------|------|-------------|
| Proto | `proto/colosseum.proto` | WeightStoreService, TrajectoryService |
| Serialization | `transport/serialization.py` | torch.save + lz4 compression |
| Weight Store Server | `weight_store/grpc_store.py` | WeightStoreServicer + serve_weight_store() |
| Weight Store Client | `weight_store/grpc_store.py` | GRPCWeightStore(BaseWeightStore) |
| Trajectory Server | `transport/grpc_transport.py` | TrajectoryServicer + serve_trajectory_receiver() |
| Trajectory Client | `transport/grpc_transport.py` | GRPCTransport(BaseTransport) |
| CLI | `cli.py` | `colosseum serve-weight-store`, `serve-trajectory` |

### Milestone 6: Eval + Docker/K8s + Polish — DONE

| Component | File | Description |
|-----------|------|-------------|
| Evaluator | `eval.py` | N-player inference-only matchups, round-robin slot assignment, Wilson CI, pairwise results |
| Docker | `deployment/Dockerfile.*` | Base, worker, weight-store images |
| Compose | `deployment/docker-compose.yaml` | Local multi-container setup |
| K8s | `deployment/k8s/` | Namespace, weight-store, worker + HPA |

### Improvement Phase 1: Stability — DONE

- T0.1: VectorEnv circular reference fix in terminal_info
- T0.2: APPO numerical stability — clamp log_ratio before exp
- T0.3: V-trace numerical stability — clamp log_rhos before exp
- T0.4: Worker inference fills log_probs/values for all slots
- T0.5: PFSP division-by-zero uniform fallback
- T0.6: CLI override parsing with proper validation
- T0.7: Replace bare `except Exception` with specific handlers
- C1: DRY `_apply_to_tensors()` in TrajectoryChunk
- C2: `build_network()` in registry.py (single factory)
- C3: Deduplicated worker inference logic
- C4: Fix Kickstart private field access
- C7: torch.load security (weights_only)

### Improvement Phase 2: Architecture — DONE

- T1.1: Action masking (`CategoricalDist` mask, `apply_mask()`, threaded through act/evaluate)
- T1.3: Dynamic algorithm selection via `algorithm_class` config + registry
- T1.5: Coordinator runtime feedback loop (workers report results, ELO/WR live updates)
- T1.7: Worker network pool (dict[str, ActorCriticNetwork] per agent, dynamic N networks)

### Improvement Phase 3: Performance — DONE

- T2.1: Pre-allocated rollout buffers (numpy arrays + write cursor, zero-copy torch.from_numpy)
- T2.2: Smart weight push (configurable `weight_push_interval`, shared WeightPayload)
- T2.3: Pinned memory for CPU→GPU transfer (`pin_memory` config, `non_blocking=True`)
- T2.7: `torch.compile` for V-trace (optional, configurable)

### Improvement Phase 4: Config & Tests — DONE

- T3.1: Global random seed (per-worker `seed + worker_id` streams)
- T3.2: AMP support (`use_amp`, `amp_dtype`, GradScaler)
- T3.4: LR schedule completion (linear, cosine, constant — all working)
- T3.6: Advantage normalization per minibatch
- T4.1-T4.7: VectorEnv, distributions, config validation, multi-player, LR schedule, graceful shutdown, trajectory boundary tests

### Improvement Phase 5: Multi-Agent & RNN — DONE

- T3.8: Per-agent config overrides (`AgentConfig`, `get_agent_config()`, `get_trainable_agent_ids()`)
- T1.4: Multi-agent launcher (per-agent learners, shared workers, multi-agent routing, slot_agent_map)
- T5.1: Eval N-player support (round-robin slot assignment, pairwise extraction from N-player matches)
- T1.2: RNN/LSTM support:
  - ActorCriticNetwork: optional recurrent trunk, `act()` 4-tuple, `evaluate_actions_recurrent()` for [T,B] sequences
  - Worker: hidden state tracking per (env, player), save at chunk start, reset on episode done
  - APPO: recurrent training path (detects `lstm_hidden`, uses `evaluate_actions_recurrent`)
  - Registry: `build_network()` creates LSTM/GRU from `recurrent_type` config
  - Config: `recurrent_type`, `recurrent_hidden_size`, `recurrent_num_layers` in NetworkConfig

### Composite Action Spaces — DONE

Native support for `gymnasium.spaces.Dict`, `Tuple`, and `MultiDiscrete` action spaces:
- `ActionSpec` codec (`core/action_spec.py`): auto-derives flat ↔ structured mapping from gymnasium space. Supports encode/decode/flatten_mask.
- `CompositeDist` (`networks/distributions.py`): multi-head distribution wrapping dict of sub-distributions. sample/log_prob/entropy/mode/apply_mask/kl_divergence on flat tensors.
- `action_dim` property on all Distribution subclasses for introspection.
- VectorEnv: auto-decodes flat actions to structured dicts at env boundary.
- Worker: uses ActionSpec for buffer allocation and dtype (fixes old int64 bug for Box spaces).
- Eval: uses ActionSpec for correct action array allocation.
- BC: auto-detects CompositeDist and handles log_prob delegation.
- Zero overhead for simple Discrete/Box spaces (all paths check `is_composite` flag).
- Example: `examples/composite_action/` — ChaseEnv with Dict(direction+speed).

### Code Cleanup — DONE

- Removed all backward-compat shims: single-agent params in worker, dual launch paths in launcher, legacy `_worker_target`/`_derive_worker_configs`. Single-agent is now a special case of multi-agent (one agent in dict).
- Removed `_ALGO_MAP` hardcoded dict — `algorithm_class` has proper default in config.
- `get_agent_config()` returns deep copy (not self) to prevent mutation of global config.
- Queue sizes extracted to named constants.
- Pre-allocated inference arrays in worker (reused per step via `.fill(0)` instead of `np.zeros()`).
- Cached obs_shape/dtype in VectorEnv (avoids `observation_space.sample()` per reset).
- Coordinator match_results bounded to deque(maxlen=10000).
- Test helpers deduplicated into `tests/helpers.py`.

### Remaining Open Items

1. **Replay buffer** for off-policy algorithms (R2D2, DQN — Tier 2)
2. **Fault tolerance** (worker/learner crash recovery)
3. **Distributed launcher** that uses gRPC transport instead of mp.Queue
4. **SAC / AlphaZero / MuZero** algorithms (Tier 3)
5. **Real K8s testing** and HPA tuning
6. **SubprocessVectorEnv** (parallel env stepping for CPU-heavy envs)
7. **Persistent gRPC streams** (one stream per agent channel, not per chunk)
8. **Faster serialization** (safetensors / direct tensor bytes)
9. **SharedMemory zero-copy weight store** (raw shared_memory instead of mp.Manager)
10. **Asymmetric environment support** (different obs/action spaces per player)
11. **Observation normalization** (running mean/std)
12. **Config inheritance / profiles** (`extends: base.yaml`)

## Tech Stack

- Python 3.11+
- PyTorch (training + inference)
- gRPC + protobuf (distributed communication)
- Docker + Kubernetes (orchestration)
- WandB (metrics)
- lz4 (compression)
- Pydantic v2 (configuration)
- Click (CLI)

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
- If arena match has fewer agents than player slots → duplicate agents to fill
- ALL trainable agents in arena match collect trajectories → one env step feeds multiple learners
- PFSP priority function: `f(win_rate) = (1 - win_rate)^p` (focus on hard opponents) or `f(x) = x*(1-x)` (balanced)
- ELO/TrueSkill tracking for all agents (ELO implemented; TrueSkill-type ratings are a target; see Roadmap)
- Dynamic: add/remove agents and machines at any time without stopping (target; see Roadmap)

### Eval Mode
- Inference-only matchups between any set of checkpoints / `.pt` state dicts (`colosseum.eval`); no training
- Same semantics as training rollouts: one model `State` per (env, seat), `info["active"]` and action masks
- Seat rotation via `schedule_lineups`: for a pair (a, b), match m gives seat s to `(a, b)[(s + m) % 2]`; `--num-matches` is per pair (rounded up to even), so every agent plays every seat equally often; in an N-player game a pairwise match fills all N seats with the two agents alternately
- Pairwise report: W/D/L, win rate and score (draw = half) with draw-aware 95% Wilson intervals, per-seat breakdown, mean returns and episode length
- Solo mode (one agent, or a 1-player env): mean return and outcome with 95% normal intervals
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

State after SP1 (foundation and stabilization, branch `sp1-stabilization`). Test suite: **868** fast tests, **3** slow, **16** GPU-only (`pytest -m "not gpu and not slow"` is the CI suite). The full review that motivated SP1 is `review/README.md` (a frozen snapshot of the pre-SP1 code; never edit it); the SP1 spec is `docs/superpowers/specs/2026-10-08-sp1-stabilization-design.md`.

Commands (run from the repo root after `scripts/setup-dev.sh`; the cwd is put on `sys.path`, also for spawned children):
- `colosseum validate -c cfg.yaml [--set k=v ...]`: schema, env `num_players`, a dummy step/unroll of every agent's model.
- `colosseum train -c cfg.yaml [--set k=v ...]`: single-machine training into `runs/<name>/`.
- `colosseum eval -c cfg.yaml -a name=path [-a ...] --num-matches N --output result.json [--deterministic] [--seed S]`.
- `colosseum bc -c cfg.yaml --data <file-or-dir> --output bc.pt [--epochs E] [--seq-len L]`.
- Distributed (limited): `colosseum serve-weight-store --port P`, `colosseum run-learner -c cfg.yaml --agent A --traj-port P --weight-store host:port`, `colosseum run-workers -c cfg.yaml --weight-store host:port -l A=host:port`.

### Works (single machine)
- **Training loop:** IMPALA-style. Workers (CPU inference, `RolloutLoop`) → per-agent learners (APPO + V-trace with `vtrace_lambda`) → newest-wins weight queues.
  - Only numpy payloads cross process boundaries.
  - Every learner update uses exactly `learner.batch_chunks` chunks.
  - `training.total_timesteps` is a global env-step budget; the LR follows its progress.
  - torch threads are limited per process (`rollout.torch_threads`, auto for learners).
- **Models:** `PolicyModel` protocol with an opaque `State` pytree.
  - `ComposedModel(encoder, core, policy, value)` with cores `NoCore`, `LSTMCore`, `GRUCore`, `WindowAttentionCore`.
  - Monolithic models are possible via `networks.model_class`.
  - The learner reproduces the worker's log-probs and values for all four cores (contract tests).
  - Optional observation normalization (`NormalizeObs`, updated once per train step).
- **Transitions:**
  - Agent-owned buffers parked across match changes.
  - Transitions stay open until the slot acts again; there is no extra bootstrap forward.
  - Turn-based games via `info["active"]`.
  - Truncation adds `γ·V(final_obs)`.
  - Action masks everywhere.
  - `ActionSpec` keeps natural component order; composite (`Dict`/`Tuple`/`MultiDiscrete`) actions via `CompositeDist`.
- **League:**
  - Every trainable agent owns envs in rotation.
  - Self-play against own checkpoints; league with N-player PFSP arenas; shuffled seats.
  - Pairwise per-seat ratings (ELO with K/(N−1), fractional win-rate matrix, `wr_vs_past` over the last 500 pairs). `k_scale = 1/(N−1)` is per seat, so an agent duplicated into several seats of one match gets proportionally more ELO exposure from that match.
  - FIFO checkpoint pool, atomic and confined to the run dir.
  - Final checkpoint on every stop.
  - Resume from a checkpoint dir, a run dir or a `.pt`. Explicit resume is strict: a missing or malformed `meta.json` in the selected agent dir is a `ConfigError` naming the path (the background checkpoint index skips such entries with a warning).
- **Observability:** `runs/<name>/` (`run.dir`/`run.name`; default name `<config stem>-<YYYYmmdd-HHMMSS>`; an existing explicit name is refused) with `config.resolved.yaml`, `logs/` (`main`, `learner-<agent>`, `worker-<i>`, `worker-<i>-env<k>`), `metrics.jsonl` (train / episodes / ratings / system), `ratings.json`, `checkpoints/<agent>/ckpt_v<N>/{model.pt, trainer_state.pt, meta.json}`; console progress per agent; optional WandB (one run, `<agent>/train_step` axis per agent, `ratings/*`, `system/*`, `episodes/*` on `env_steps`).
- **Config:**
  - Pydantic with `extra="forbid"`.
  - Partial agent overrides deep-merged onto the global sections (`agents.<id>.{networks,algorithm,learner}`).
  - Agent ids are safe path components without `.` (`[A-Za-z0-9_][A-Za-z0-9_-]*`) and not `ratings`/`system`/`episodes`/`train`.
  - `--set` with YAML values for `train` / `validate` / `run-learner` / `run-workers`.
  - `validate` checks `env.num_players`, reset masks with the worker's seat rules, and the model protocol.
- **Lifecycle:**
  - Exit codes 0 (budget reached) / 1 (config error or a dead child) / 130 (SIGINT) / 143 (SIGTERM); `eval` also 2 for bad arguments.
  - Children ignore SIGINT and die with the parent.
  - A dead child stops the run with a pointer to its log.
  - On Ctrl+C / SIGTERM the final checkpoints are saved within the shutdown grace (`SHUTDOWN_GRACE_SEC = 7`); no process is alive after 10 s.
- **Eval:** `PolicyModel`-based with per-seat state and seat rotation; W/D/L with Wilson CIs, per-seat breakdown, solo mode, JSON output; architecture taken from checkpoint `meta.json`.
- **BC / kickstart:** distribution-aware `-log_prob` loss with masks, sequence training for stateful models, forward-KL kickstarting.
- **Learning checks:** fast bandit and combination-lock chain tests (with a `gamma=0` negative control); the slow tic-tac-toe test (`configs/examples/tic_tac_toe.yaml`, ~70 s on 8 CPU cores) reaches 86–93% wins against a random legal-move player (threshold 80%).

### Partial
- **Distributed mode** (`serve-weight-store`, `run-learner`, `run-workers`) works for latest-weights self-play only: no coordinator, league, ratings, `metrics.jsonl` or WandB; per-worker budgets; `run-learner` ignores `training.resume_from`.
- **`deployment/`** (Docker, K8s) is not tested.
- **GPU:** never run on CUDA during SP1. All CUDA paths (learner device, AMP fp16/bf16, `pin_memory`, kickstart/BC on CUDA) are covered only by `gpu`-marked tests that have not been executed yet; see `docs/GPU_CHECKS.md`.

### Not implemented (see Roadmap)
- Scripted, frozen and external players.
- Asymmetric or team games, player elimination, Dict observations.
- Snapshot-level PFSP, OpenSkill / Bradley–Terry ratings, a tournament command.
- Off-policy algorithms (R2D2/DQN, replay buffer), SAC, AlphaZero/MuZero.
- GPU inference on workers.
- Shared-memory zero-copy weight store and shared-memory trajectory ring buffer (single machine uses `mp.Queue` with numpy payloads).
- Adding or removing agents and machines while a run is going (agents and workers are fixed at start).
- gRPC control plane between a coordinator and workers/learners (SP5).
- TrueSkill-type ratings (SP4: OpenSkill / Bradley–Terry).

## Roadmap

| Sub-project | Scope |
|---|---|
| **SP1 Foundation and stabilization** (done) | correct, observable, robust single-machine training; `PolicyModel` protocol |
| **SP2 Game model** | `GameSpec` / `MultiAgentEnv`, elimination, teams, roles, variable unit counts, per-unit actions, Dict observations, bootstrap moved to the learner |
| **SP3 Players, league, warm start** | scripted / frozen / external players, PFSP over snapshots, per-agent warm start (`init`, kickstart, critic warm-up), top-k snapshot storage |
| **SP4 Selection and observability** (parallel with SP5) | match log, OpenSkill / Bradley–Terry, `colosseum tournament`, dashboard, snapshot ratings |
| **SP5 Distributed** (parallel with SP4) | hub and nodes, wire format, per-machine weight cache, fault tolerance, `max_policy_lag`, K8s images |
| **SP6 Speed and extensions** | cuDNN RNN path, fast transformer unroll, GPU inference on workers, inference server, new algorithms |

### Open items parked during SP1
- **SP4:**
  - full unification of the eval engine with `RolloutLoop` (eval reuses the seat/mask rules from `core/seat_info.py` but has its own loop);
  - paired, rating-based checkpoint selection (Bradley–Terry with bootstrap) instead of per-pair Wilson intervals.
- **SP5:**
  - distributed learner resume (`run-learner` never applies `training.resume_from`);
  - metrics hub / `metrics.jsonl` / WandB for the distributed roles (today: per-process logs only);
  - resource leak when `run-learner` setup fails after the trajectory server, weight-store client or drainer started (seeding or drainer start failure);
  - one budget semantics for distributed mode (today `total_timesteps / num_workers` per worker; learner progress = `consumed_samples / total_timesteps`);
  - newest-wins weight-queue eviction unpickles the stale payload (CPU ∝ model size × workers) → per-machine weight cache.
- **SP6:**
  - normalizer statistics are updated before the loss forward, so the ratio is not exactly 1 at zero lag on early steps (updating after the epochs would give exact parity);
  - `WindowAttentionCore` rebuilds its masks every step (cache with the fast path);
  - BC windows are not episode-aligned.
- **Tier 2 algorithms:** replay buffer for R2D2/DQN (the replay path would also need its own normalizer-stat and warm-up handling).
- **Minor, unscheduled:**
  - `--set` lists follow YAML 1.1, so `--set x=[1e-4,2]` keeps `1e-4` as a string (scalars are parsed correctly; write `1.0e-4` in lists);
  - the main process builds a ratings snapshot on every monitor pass (negligible cost);
  - stale parked rollout buffers have no age limit (bounded by agents × envs × players);
  - SharedMemory zero-copy weight store (raw `shared_memory` instead of queue payloads);
  - dynamic add/remove of agents and machines during a run;
  - config inheritance / profiles (`extends: base.yaml`).

## Tech Stack

- Python 3.11+
- PyTorch (training + inference)
- gRPC + protobuf (distributed communication)
- Docker + Kubernetes (orchestration)
- WandB (metrics)
- lz4 (compression)
- Pydantic v2 (configuration)
- Click (CLI)

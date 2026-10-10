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
- Online BC (kickstarting): `loss = RL_loss + λ * KL(BC_teacher || policy)` (forward KL by default, `kickstart.kl`), λ decays over training; KL per decider over `act` slots; a teacher per agent: `kickstart` section per agent (global section = default), any frozen agent, `.pt` or checkpoint dir (own architecture), or a scripted agent (DAgger: `λ * mean(-log π(a_teacher))` over the workers' labels)
- Both approaches available; BC phase runs before RL phase (`colosseum record` → `colosseum bc` writes `bc.pt`, then `init.from: bc.pt` with `critic_warmup_steps`, and/or `kickstart.teacher`)

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
- **Online BC**: scripted agent generates trajectories, neural agent learns from them (scripted teachers (DAgger) exist since SP3: `kickstart.teacher: <scripted agent>`; also neural teachers of any architecture)
- Training: supervised `-log_prob` loss (masked; sequences for stateful models)
- Result: initial policy that plays at basic level

### Phase 3: Self-Play
- Agent plays against pool of its own checkpoints + current self
- Each N-player match: random mix of latest (collect trajectories) + historical checkpoints (don't collect)
- Checkpoint pool: `keep_last` + `keep_every` + final snapshot, save every K training steps; snapshots are opponents via `matchmaking.opponents.snapshots` (PFSP)
- Purpose: stable improvement without being crushed by much stronger opponents

### Phase 4: PFSP / League (longest phase)
- For each opposing team, a core category drawn by `matchmaking.opponents` shares (since SP3; SP2's `self_play_ratio` / `latest_prob` are translated):
  - **latest** / **snapshots**: the owner's latest weights or stored snapshots (PFSP)
  - **rivals** (arena): other TRAINABLE agents' latest weights (PFSP; these seats collect too)
  - **anchors**: scripted / frozen agents (by weight); shares and anchor weights may be schedules over env steps
- Every team gets a core agent (the owner, its checkpoints, or another trainable agent by PFSP); `matchmaking.teammates: self | mixed` fills the team's other seats; a role the core does not play goes to an agent that plays it
- ALL trainable agents in arena match collect trajectories → one env step feeds multiple learners
- PFSP priority function: `f(win_rate) = (1 - win_rate)^p` (focus on hard opponents) or `f(x) = x*(1-x)` (balanced): `matchmaking.pfsp` `hard` / `balanced` / `uniform` (EMA score with `halflife_games`, per layout), over latest weights and stored snapshots of every agent that plays the team
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
- **FrozenCheckpoint**: static weights loaded from disk, used as sparring partner, no training (implemented: `agents.<id>.kind: frozen`, a checkpoint dir or `.pt`, any architecture)
- **ScriptedAgent**: rule-based bot, no neural network (implemented: `agents.<id>.kind: scripted`, a `colosseum.players.ScriptedBot` subclass; `RandomBot` built in)

Multiple trainable agents can coexist with different architectures, different RL algorithms, different encoders — they are fully independent (per-agent `agents.<id>.{roles,networks,algorithm,learner,matchmaking,init,kickstart}`). This allows comparing approaches side by side.

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
- Snapshot retention per agent: `checkpoint.keep_last` + `keep_every` + the final snapshot (`trainer_state.pt` only in the `keep_last` window; a run-dir resume carries the pool over); top-k is SP4
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
1. Scripted agent plays games, trajectories saved to disk (`colosseum record --player <bot>`; format in `bc/offline_bc.py`)
2. OR: download external replay files
3. Neural agent trains supervised on these trajectories
4. Standard supervised learning loop, no env interaction needed

**Mode B — Online BC / DAgger-like** (implemented in SP3 as the scripted kickstart teacher: the workers label every decision of a collecting seat with the bot's action, the learner adds `λ · mean(-log π(a_teacher))`):
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

State after SP3 (players, league, warm start; developed on branch `sp3-league`, awaiting the owner's acceptance; SP2 was merged into `main` on 2026-10-09). Test suite: **1662** fast tests, **10** slow, **22** GPU-only (`pytest -m "not gpu and not slow"` is the CI suite).
- SP3 records: spec `docs/superpowers/specs/2026-10-10-sp3-league-design.md` (Russian); plan `docs/superpowers/plans/2026-10-10-sp3-league/` (`00-overview.md` has the interface contract and amendments A1–A33; pre-flight rulings P1–P18); acceptance report and final-review records in `docs/superpowers/reports/` (`<date>-sp3-acceptance.md`, `<date>-sp3-review/`, written at acceptance; every ruling and deferred minor is in the report).
- SP2 records: spec `docs/superpowers/specs/2026-10-08-sp2-game-model-design.md` (Russian); plan `docs/superpowers/plans/2026-10-08-sp2-game-model/` (`00-overview.md` has the interface contract and amendments R1–R18); acceptance report `docs/superpowers/reports/2026-10-09-sp2-acceptance.md` (criteria, measurements, every ruling, open items, all deferred task minors in its appendix); final review, fix wave and FIX-1/FIX-2 records `docs/superpowers/reports/2026-10-09-sp2-review/`.
- SP1 records: `docs/superpowers/reports/2026-10-08-sp1-acceptance.md`; the pre-SP1 review `review/README.md` is a frozen snapshot (never edit it).

Commands (run from the repo root after `scripts/setup-dev.sh`; the cwd is put on `sys.path`, also for spawned children):
- `colosseum validate -c cfg.yaml [--set k=v ...]`: `GameSpec` and config checks (roles, matchmaking), `reset` of every enabled layout and random legal steps with full contract checks, `step` and `unroll` of every agent's model, the kickstart teacher.
- `colosseum train -c cfg.yaml [--set k=v ...]`: single-machine training into `runs/<name>/`.
- `colosseum eval -c cfg.yaml -a name=path [-a name ...] [--layout L ...] --num-matches N --output result.json [--num-envs E] [--deterministic] [--seed S]` (a bare `-a name` plays a scripted or frozen agent of the config).
- `colosseum record -c cfg.yaml --player P [--against A ...] --num-matches N --output DIR [--layout L ...] [--num-envs E] [--seed S] [--deterministic]` (P, A: a scripted/frozen agent or `name=path`; BC data per role + `record.json`).
- `colosseum bc -c cfg.yaml --agent A --data <file|dir|record-dir> [--data ...] --output bc.pt [--epochs E] [--batch-size B] [--lr LR] [--seq-len L]` (`--data` repeatable; a `record` output dir contributes the folders of the agent's roles).
- Distributed (limited): `colosseum serve-weight-store --port P [--max-message-mb M]`, `colosseum run-learner -c cfg.yaml --agent A --traj-port P --weight-store host:port`, `colosseum run-workers -c cfg.yaml --weight-store host:port -l A=host:port`.
- Measurements: `scripts/bench_throughput.py`, `scripts/units_experiment.py` (`--cells ratio_mode:unit_trace ...` for the SP3 defaults rule), `scripts/measure_global_state.py`, `scripts/team_tag_anchors.py`, `scripts/pipeline_vs_scratch.py` (results in `docs/benchmarks.md`).
- Docs: `docs/ENV_GUIDE.md` (writing an env; §10 `infos` for bots), `docs/LEAGUE_GUIDE.md` (players, league, warm start recipes), `docs/benchmarks.md`, `docs/GPU_CHECKS.md` (all Russian).

### Works (single machine)
- **Game model (SP2):**
  - env contract `GameSpec` (roles, layouts = lists of seats with roles and teams) + `MultiAgentEnv` + `StepResult` (`acting`, `terminated`, `truncated` + `final_obs`, `global_state`, `Outcome`); `EpisodeTracker` checks every rule and raises `EnvContractError` with "worker, env, seat, episode step, layout";
  - outcome kind from the number of teams (1 → `score`, 2 → `wdl`, ≥3 → `rank`); default outcome = mean of a team's seat returns;
  - seat lifecycle: empty / live / eliminated; a dead teammate is removed from `acting` but keeps team rewards; steps without decisions bounded by `env.max_idle_steps`;
  - observation, action, mask and `global_state` trees with preserved dtypes (`uint8` reaches the model); bool masks only; entity lists with masks; `Units(max_units, per_unit, only_if)` with `Discrete`/`MultiDiscrete`/`Box`/`Dict` components; deciders (each unit + the non-unit part).
- **Training loop:** IMPALA-style; one `MatchRunner` core for training (`RolloutLoop`) and eval. Workers infer only the policy; chunk v2 with `act`/`boot`/`pad` slots; the learner computes every value (bootstrap from `boot` slots and `final_obs`) with its current network. Only numpy payloads cross process boundaries. Every learner update uses exactly `learner.batch_chunks` chunks; `training.total_timesteps` is a global env-step budget.
- **Algorithm:** APPO with V-trace over slots; `ratio_mode` (`joint` / `per_unit`; `auto` = `per_unit` with `Units`), `unit_trace` (`joint` / `geo_mean` / `none`; `auto` = `joint` for every action), `entropy_reduction` — the `Units` defaults were kept by SP3's pre-fixed multi-seed rule (T6.5: at K=128 `joint` ratio +0.037 < 2·SE 0.092, `unit_trace: none` −0.005; `docs/benchmarks.md`); with one decider every mode collapses to `joint` (except an explicit `unit_trace: none`, ruling PR-1); every loss and metric reduces over `act` slots; per-unit diagnostics (`clip_fraction_joint`, `ess`, `log_rho_abs_p95`, `deciders_valid_mean`, ...). Centralized critic: `networks.critic_encoder_class` over `global_state`, value path only.
- **Models:** `PolicyModel` protocol (`step` for the policy path, `unroll` for the learner, `with_value=False` for BC and the kickstart teacher); `ComposedModel(encoder, core, policy, value, critic_encoder)` with `EncoderOutput(latent, aux)` for per-unit features; cores `NoCore`, `LSTMCore`, `GRUCore`, `WindowAttentionCore`; `UnitsHead`, `gridnet_to_units`; `NormalizeObs` on a leaf path. The learner reproduces the worker's joint and per-unit log-probs for all four cores (contract tests).
- **Players (SP3):** agent kinds `trainable` / `scripted` / `frozen` (`agents.<id>.kind`); an implicit trainable `agent_0` when the config has none; `colosseum.players.ScriptedBot` (`reset(role, seat, layout, rng)`, `act(obs, mask, info)`, one instance per (agent, env, seat)) and the built-in `RandomBot`; fixed players sit on the `fixed` network id and never collect; bots get `StepResult.infos[seat]`; a bot's illegal action or exception is a `PlayerError` with "worker, env, seat, episode step, layout, agent"; frozen agents of any architecture from a checkpoint dir (meta.json) or a `.pt` (+ `networks` / `roles`); SP2 checkpoints load as frozen agents, `init` and for resume. Demo bots: `examples/tic_tac_toe/bots.py`, `examples/unit_harvest/bots.py`.
- **League:** lineups (layout + seat assignments); owner rotation (config order, from the player registry); layout weights; the v3 mix `matchmaking.opponents` (`latest` / `snapshots` / `rivals` / `anchors`; numbers or piecewise-linear schedules over env steps; empty categories spread proportionally, a `fallback` source when all positive ones are empty), `anchors` weights, `pfsp` (`hard` / `balanced` / `uniform`, EMA with `halflife_games`, per player and layout, over latest weights and snapshots of every agent that plays the team; opponents' snapshots in asymmetric games); per-agent `matchmaking` overrides; SP2 knobs translated with one warning (exact, shares rounded to 12 decimals); `colosseum.league.BaseMatchmaker` + `MatchmakerContext` for custom matchmakers (every lineup checked); teams with a core and `teammates: self | mixed`; roles per agent; seat permutation within equal role compositions; `validate` prints each agent's effective mix per layout. Ratings per layout (`ratings.json` = `{"env_steps", "layouts": {...}}`): team-pair ELO, win-rate matrix, `wr_vs_past`, `ScoreTracker` and cross-play, scripted/frozen agents as entities of their own, the PFSP table; played opponent shares per category and anchor in the `episodes` records.
- **Snapshot storage (SP3):** `checkpoint.keep_last` / `keep_every` / the final snapshot never deleted; `trainer_state.pt` only in the `keep_last` window; evictions reach the workers (models unloaded) and the PFSP tables; a run-dir resume carries the stored snapshots over (hard links, else copies); strict resume with roles and role signature in `meta.json`.
- **Warm start (SP3):** per-agent `init` (`from`: `.pt`, checkpoint dir, run dir or a frozen agent; `strict` or partial with a report in the log and `validate`; weights only, policy version 0); `critic_warmup_steps` (value path only, the policy and normalizer statistics bit-identical); kickstart per agent from a neural teacher of any architecture (KL per decider; a recurrent teacher only with the student's state layout) or a scripted teacher (DAgger: workers write `teacher_action` / `has_teacher` while `teacher_active`); `training.resume_from` takes precedence over `init`.
- **record / BC:** `colosseum record` (any player kind, with or without `--against`, per-role BC files with contiguous seat-episodes, `record.json` with an eval summary); `colosseum bc` with repeatable `--data` and `record` output dirs.
- **Eval:** `MatchRunner`-based; CLI lineups per SP2 spec block 8; `-a name` for scripted/frozen agents of the config; reports for `wdl` (Wilson intervals, per side), `rank` (mean rank, first places, pairwise matrix), `score` (mean, cross-play), by role; in-process `play_lineups` with any `PolicyModel` and `ScriptedPlayer`s.
- **BC:** BC on observation/action trees (`-log_prob`, mean over valid deciders with `Units`; an illegal masked expert action is a `DataError`).
- **Demo games** (`examples/`, configs in `configs/examples/`): `coin_grid` (solo), `tic_tac_toe` (1v1 turn-based; also `tic_tac_toe_attention.yaml`, `tic_tac_toe_multi.yaml`), `unit_harvest` (units), `team_tag` (2v2, `global_state`), `tron` (FFA 2–4, elimination), `predator_prey` (asymmetric roles), `coop_buttons` (cooperative); reference examples `space_miners` (Box2D) and `composite_action` (chase). Each demo game has a slow learning test against random players (`tests/learning/test_demo_learning_slow.py`; team_tag is a single run again since SP3: its config trains against a `RandomBot` anchor, `opponents.anchors` 0.2 by the T6.3 measurement, played share ≈0.36); `configs/examples/unit_harvest_league.yaml` (agent `main`, anchors `greedy` and `random`) drives the README pipeline and a slow pipeline test (`tests/learning/test_sp3_pipeline_slow.py`; thresholds by T6.4: win vs random ≥ 0.95, score vs greedy ≥ 0.40, vs the BC net ≥ 0.45); fast learning tests cover a Units bandit, a cooperative bandit, a scripted-teacher kickstart and a `RandomBot` anchor, and a fast tic-tac-toe pipeline smoke (record → bc → train with init/warm-up/kickstart → eval). Example configs are in the SP3 form (exact translations of their SP2 form; SP2 copies under `tests/fixtures/sp2/configs/`). `docs/ENV_GUIDE.md` (Russian) explains how to write an env for every game type.
- **Observability:** `runs/<name>/` (`run.dir`/`run.name`; default name `<config stem>-<YYYYmmdd-HHMMSS>`; an existing explicit name is refused) with `config.resolved.yaml`, `logs/` (`main`, `learner-<agent>`, `worker-<i>`, `worker-<i>-env<k>`), `metrics.jsonl` (`train` / `episodes` / `ratings` / `system`, with per-layout and per-role breakdowns; `system` includes `dropped_reward_episodes`), `ratings.json`, `checkpoints/<agent>/ckpt_v<N>/{model.pt, trainer_state.pt, meta.json}`; console progress per agent; optional WandB. A seat that never acts in an episode loses that episode's rewards: one WARNING per worker, then counted.
- **Config:** Pydantic with `extra="forbid"`; partial agent overrides deep-merged onto the global sections (`agents.<id>.{roles,networks,algorithm,learner,matchmaking,init,kickstart}`); SP2 knobs (`matchmaking.mode` / `self_play_ratio` / `latest_prob` / `pfsp_exponent`, `checkpoint.pool_size`, `training.kickstart_*`) accepted in the input YAML only, translated with one warning each, never stored; old + new knob together = `ConfigError`; agent ids `[A-Za-z0-9_][A-Za-z0-9_-]*`, not `ratings`/`system`/`episodes`/`train`; `--set` with YAML values for `train` / `validate` / `run-learner` / `run-workers`; `self_play.*` became `matchmaking.*` + `checkpoint.*`, `env.num_players` and `training.phase` are gone.
- **Lifecycle:** exit codes 0 (budget reached) / 1 (config error or a dead child) / 130 (SIGINT, also during startup and for `serve-weight-store`) / 143 (SIGTERM); `eval` also 2 for bad arguments. Children ignore SIGINT and die with the parent; a dead child stops the run with a pointer to its log; final checkpoints on every stop within `SHUTDOWN_GRACE_SEC = 7`; no process alive after 10 s. Explicit resume is strict (`ConfigError` on a missing or malformed `meta.json` or a role-signature mismatch; SP1 checkpoints have no role signature and cannot be resumed).
- **Throughput:** 4815 / 9361 / 14090 env steps/s with 1 / 2 / 4 workers on tic-tac-toe (SP3 branch, mean of two runs; 102.7 % / 100.2 % / 99.8 % of `main` measured alternately on the same machine, criterion 3.5; SP2 at acceptance: 4848 / 9298 / 13702; `docs/benchmarks.md`).

### Partial
- **Distributed mode** (`serve-weight-store`, `run-learner`, `run-workers`) runs on chunk v2 for games where every agent plays every role, latest weights only: no coordinator, league, ratings, `metrics.jsonl` or WandB; per-worker budgets; `run-learner` ignores `training.resume_from`. SP3 features are local only: scripted and frozen agents, `init`, critic warm-up, a custom matchmaker, per-agent `matchmaking` or `kickstart` are refused with a `ConfigError` pointing to SP5 (`distributed.check_distributed_scope`), and any opponent mix is reduced to latest weights with one warning.
- **`deployment/`** (Docker, K8s) is not tested.
- **Unused, kept for SP5:** `transport/local.py::LocalTransport`, `weight_store/shared_memory.py::SharedMemoryWeightStore`, `transport.mode` / `transport.grpc_port`.
- **GPU:** never run on CUDA. All CUDA paths (learner device, AMP, `pin_memory`, kickstart/BC, tree batches and `Units` distributions on CUDA) are covered only by `gpu`-marked tests; see `docs/GPU_CHECKS.md`.

### Not implemented (see Roadmap)
- Top-k snapshot storage, snapshot ratings (SP4); distributed league with scripted/frozen players and SP3 warm start (SP5); recurrent kickstart teacher with another state layout, batched scripted-bot API, exploiters, preset mixes, value pre-training on BC data, a one-command phase pipeline (later, as needed; SP3 spec §2).
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
| **SP3 Players, league, warm start** (done) | scripted and frozen players (also as `record` data sources and DAgger teachers), the v3 opponent mix (latest / snapshots / rivals / anchors, schedules, per-agent overrides, custom matchmaker) with PFSP over snapshots (including the opponent's in asymmetric games; `hard` / `balanced` / `uniform`), a `RandomBot` anchor against passivity (team_tag passes in one run), snapshot retention (`keep_last` / `keep_every` / final, pool carried over on resume), per-agent warm start (`init`, critic warm-up, kickstart neural or DAgger), `colosseum record`, APPO `Units` defaults re-checked |
| **SP4 Selection and observability** (parallel with SP5) | match log, OpenSkill / Bradley–Terry, `colosseum tournament`, dashboard, snapshot ratings, top-k snapshot storage |
| **SP5 Distributed** (parallel with SP4) | hub and nodes, distributed league (including asymmetric games, agents that do not play every role, scripted/frozen players and SP3 warm start), adding / removing agents and machines during a run, shared-memory weight store and trajectory ring buffer, wire format, per-machine weight cache, fault tolerance, `max_policy_lag`, K8s images |
| **SP6 Speed and extensions** | vectorized fast path for array envs and hundreds of seats (Neural MMO), cuDNN RNN path, fast transformer unroll, GPU inference on workers, inference server, new algorithms |

### Open items parked during SP1, SP2 and SP3
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
  - budget overshoot and resume: a local run overshoots `total_timesteps` by about one second of env steps; a resume needs a budget above the checkpoint's `env_steps` to train (optional warning when a resume starts at or past the budget) → SP5 "one budget semantics";
  - distributed workers get identical per-env seeds across machines when `training.seed` is set (a consequence of ruling PR-3: seeds are `training.seed + worker_id` on every machine) → SP5;
  - SP4: `system` `dropped_reward_episodes` sums only recently reporting workers (can decrease, can read 0 in the last record); make it a per-run monotonic counter.
- **SP2 final-review Minors worth knowing** (full lists: the SP2 report's «Финальное ревью ветки» and appendix, and `docs/superpowers/reports/2026-10-09-sp2-review/`):
  - SP5: with several agents the distributed roles silently ignore `teammates: mixed` and `shuffle_seats` (SP3 now refuses players, overrides and custom matchmakers and warns on any opponent mix); the learner trusts slot payload values from a peer (finite `behavior_logp`, zero rewards on `boot`/`pad`), and a chunk with a wrong observation shape fails only in the learner forward; `distributed_setup` duplicates `setup_run`'s spec/roles sequence;
  - GPU checks: Gaussian log-prob and entropy run in fp16 under AMP (also inside `Units`); compare per-decider log-ratios with fp32 on a Box-in-`Units` AMP step;
  - SP6: `unit_log_prob` is computed twice per step (`act` and APPO's evaluate), the `Units` gate table is rebuilt per call;
  - unscheduled: `global_state` is stored and sent but unused when the agent has no critic encoder (a `validate` warning); `role_signature` ignores `n` / `nvec` of discrete observation leaves; all seats eliminated without `episode_over` is reported only after `max_idle_steps` idle steps; "masked units give no NaN gradient" holds only for finite parameters; `MatchRunner` trusts that every seat's role is one its agent plays.
- **Parked during SP3** (from the SDD ledger; the acceptance report lists every ruling; items the final fix wave closes are removed there):
  - SP4: snapshot ratings and top-k storage (`keep_every` / frozen agents meanwhile); the online ELO stays a progress indicator; `system` `dropped_reward_episodes` (above) still sums recent workers only;
  - SP5: the "reduced to latest only" distributed warning is emitted before file logging starts (console only) and also fires when only empty categories carry non-latest shares; a global `teammates: mixed` is still silently ignored by the distributed roles; `validate_chunk_payload` checks only presence and shape of the DAgger fields (`teacher_action`, `has_teacher`) — the peer-trust class above; the `pool_size` translation warning is printed before the run dir exists (not in `logs/main.log`);
  - final fix wave candidates (code quality, no behaviour risk): `validate`'s scripted-player check duplicates `_exercise_env`'s reset/step error wrapping; `play_lineups` should force `collect=False` on every seat (eval never collects; until then callers pass it for scripted seats); "construct the user matchmaker, wrap failure" duplicated in coordinator and validate; `mixture.CATEGORIES` duplicates `OPPONENT_CATEGORIES[:4]`; `lineup_for` raises `KeyError` for "no playable layout"; the zero-weight anchor fallback branch in `mixture.py` is unreachable; `learner/factory.py` duplicates checkpoint discovery (`_CKPT_DIR_RE` vs `checkpoint_manager._CKPT_RE`, `classify_resume_source`); `record._parse_player` duplicates `cli._parse_agent_spec`; the "competitive layout" odd-count rounding predicate exists in `cli eval` and `record`; the teacher-bot handling in `RolloutLoop` parallels `MatchRunner`'s and duplicates `EpisodeTracker`'s episode bookkeeping; W/D/L logic in three places (`team_tag_anchors.wdl`, the team_tag slow test, `demo_learning.win_rate`);
  - minor, behaviour: `colosseum bc` has no `--seed` (BC init and the pipeline differ run to run); `kickstart_label_frac` reads 0 during the critic warm-up; a missing-checkpoint fallback to latest keeps `source="snapshots"` in the played shares (documented in LEAGUE_GUIDE); `validate` plays bots only in the enabled layouts and exercises a custom matchmaker only at env step 0 without snapshots; `validate` does not check resumed weights/signatures per agent (a broken run-dir checkpoint also hides that agent's `init` check); `--set` cannot reach inside an anchor map or schedule (set the whole value); `pfsp.exponent` accepts `inf`, `PfspStats` does not validate its prior; integer action components accept `bool`; `FrozenSpec ==` raises (numpy; use `eq=False`); `_check_critic_warmup` lets an import error of a mistyped `algorithm_class` escape raw, and its signature check rejects `**kwargs`-forwarding APPO subclasses; shared value/policy parameters are not detected by the warm-up (documented contract); `record` leaves partial part files after a mid-run failure; the `unit_harvest` pipeline thresholds leave a 0.017 margin on the BC-net check (≈0.2–10 % flake estimate, ruling T6.4); team_tag anchor tie rule compares without float tolerance;
  - minor, cost: player roles and frozen `meta.json` resolved twice in `setup_run`, frozen agents loaded twice by `eval` (validate + evaluate), kickstart teachers and `init` weights resolved twice in `train` (validate + launcher), `_check_kickstart_teachers` rebuilds the student, `ObsSpec` rebuilt per bot decision in `validate`; `Coordinator._evicted` grows when match refresh is disabled; `_notify_evicted` drops later callback errors without a log line;
  - minor, tests: no test of the snapshot copy fallback / hard-link proof; no test with a frozen or scripted TEAMMATE; no APPO-level DAgger test with K > 1 (`Units`); no neural `Units` recording test; no `--against` test with an explicit one-team layout; no test for an invalid `.pt` networks override, a malformed `meta.json` via `resolve_player_roles`, `assert_no_tensors(FixedPlayers)`; no uint8-observation test for bots; `test_sp2_validate.py:159`'s name no longer matches its content; a few weak assertions (`match="agents"`, loose "KL", `loss > 0` for the recurrent-teacher state); `units_experiment.py` decides with n=2 if a seed fails and prints raw tracebacks for bad `--cells`.

### Next step
SP4 (selection and observability) and SP5 (distributed) — they may run in parallel (see Development Workflow); each starts with a brainstorm and a written spec before any plan or code, on its own branch (`sp4-<name>`, `sp5-<name>`). SP3 must first be accepted by the owner and merged.
- Design inputs: the SP4 / SP5 rows of the Roadmap; «Parked during SP3» and the SP4 / SP5 items above; SP3 spec §2 «Вне рамок» (`docs/superpowers/specs/2026-10-10-sp3-league-design.md`: top-k snapshots, snapshot ratings, OpenSkill / Bradley–Terry, tournament, match log, dashboard → SP4; hub and nodes, distributed league, scripted / frozen players, anchors and SP3 warm start across machines → SP5); the SP1–SP3 rulings (acceptance reports).
- SP4 starts from what SP3 records: ratings per layout with scripted/frozen entities, the PFSP tables, played opponent shares in `metrics.jsonl`, `keep_every` as the interim way to keep strong snapshots.
- SP5 starts from `distributed.check_distributed_scope` (the list of SP3 features the distributed roles refuse today).
- GPU checks still pending: the owner has to run `docs/GPU_CHECKS.md` (22 tests) on a CUDA machine.

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
  - slow suite: `.venv/bin/python -m pytest -m slow -v` — 10 tests (the 7 demo-game learning tests in `tests/learning/test_demo_learning_slow.py`, the unit_harvest pipeline test `tests/learning/test_sp3_pipeline_slow.py` and 2 torch.compile checks in `tests/unit/test_sp2_performance.py`), about 15 min (906 s at T6.4); run it at acceptance and after any change to the training math, the worker or the learner;
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

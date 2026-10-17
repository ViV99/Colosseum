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
- `research/sp4_ratings_selection.md` (Russian) — SP4 research: rating models for our setting (Elo, BT, TrueSkill / OpenSkill, teams, FFA, asymmetric roles, anchors), online vs batch fitting, snapshot ratings, selecting the best with a given confidence (sample sizes, paired designs, winner's curse, successive halving), top-k retention
- `research/sp4_selection_practice.md` (Russian) — SP4 practice: how competition winners choose submissions, ladder rating systems, tournament formats and tools, match-log formats and sizes, league dashboards
- `research/sp4_sims/` — the simulation scripts behind the two SP4 research documents

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
- Online BC (kickstarting): `loss = RL_loss + λ * KL(BC_teacher || policy)` (forward KL by default, `kickstart.kl`), λ decays over training; KL per decider over `act` slots; a teacher per agent: `kickstart` section per agent (global section = default), a frozen agent (own architecture; `networks` for a `.pt`) or a checkpoint dir (own architecture), a bare `.pt` path (built with the student's architecture), or a scripted agent (DAgger: `λ * mean(-log π(a_teacher))` over the workers' labels)
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
- Checkpoint pool: `keep_last` + `keep_every` + the current run's final snapshot + pinned snapshots + (SP4) `keep_top_k` by the lower bound of the snapshot's Bradley–Terry rating, capped by `max_snapshots`; save every K training steps; snapshots are opponents via `matchmaking.opponents.snapshots` (PFSP)
- Purpose: stable improvement without being crushed by much stronger opponents

### Phase 4: PFSP / League (longest phase)
- For each opposing team, a core category drawn by `matchmaking.opponents` shares (since SP3; SP2's `self_play_ratio` / `latest_prob` are translated):
  - **latest** / **snapshots**: the owner's latest weights or stored snapshots (PFSP)
  - **rivals** (arena): other TRAINABLE agents' latest weights (PFSP; these seats collect too)
  - **anchors**: scripted / frozen agents (by weight); shares and anchor weights may be schedules over env steps
- Every team gets a core agent (the owner, its checkpoints, or another trainable agent by PFSP); `matchmaking.teammates: self | mixed` fills the team's other seats; a role the core does not play goes to an agent that plays it
- ALL trainable agents in arena match collect trajectories → one env step feeds multiple learners
- PFSP priority function: `f(win_rate) = (1 - win_rate)^p` (focus on hard opponents) or `f(x) = x*(1-x)` (balanced): `matchmaking.pfsp` `hard` / `balanced` / `uniform` (EMA score with `halflife_games`, per layout), over latest weights and stored snapshots of every agent that plays the team
- Ratings: the online ELO of SP2/SP3 (a progress indicator, unchanged) and, since SP4, Bradley–Terry ratings of every snapshot (version bucket) and scripted / frozen agent from the match log, with intervals: live during training (refit on a timer, Laplace) and with bootstrap in reports; TrueSkill-type online ratings are not planned (the live rating is BT on a timer; SP4 spec §8, controller decision 1)
- Selection: `colosseum tournament` (SP4) — a controlled design on common seeds with seat rotations; `--mode select` screens by successive halving, confirms the finalists on fresh seeds and reports `confirmed`; pulse (`evaluation`) plays every new snapshot against a fixed field during training
- Dynamic: add/remove agents and machines at any time without stopping (target; see Roadmap)

### Eval Mode
- Inference-only matchups between any set of checkpoints / `.pt` state dicts (`colosseum.eval`); no training; `-a name` (no path) plays a scripted or frozen agent of the config
- Same engine as training rollouts (`MatchRunner`): one model `State` per (env, seat), `StepResult.acting` and action masks
- Lineups via `schedule_lineups` (per layout): for a pair (a, b), match m gives team i as a whole to `(a, b)[(i + m) % 2]` (or to the pair's other agent when that one does not play every role of the team); an orientation the roles forbid is replaced by the other, and a pair is dropped from a layout only when both are impossible; one agent fills every team; one-team layouts play each agent's homogeneous team plus mixed compositions (cross-play); `--num-matches` is per pair (or composition) and layout
- Reports per layout and role: `wdl` — W/D/L, win rate and score with 95% Wilson intervals, per side; `rank` — mean rank with CI, first-place share, pairwise matrix; `score` — mean score with CI, cross-play table
- JSON output with `--output`; architecture of each checkpoint rebuilt from its `meta.json` (`.pt` files from `--config`)
- Command: `colosseum eval -c config.yaml -a A=runs/x/checkpoints/agent_0/ckpt_v100 -a B=runs/x/checkpoints/agent_0/ckpt_v200 --num-matches 1000 --output result.json` (`--num-matches` is per pair (or composition) and layout; the architecture comes from each checkpoint's `meta.json`)
- `-a name=path` never takes a frozen config entry's `.pt` settings (SP4; a trainable entry's architecture still applies to a `.pt`); selecting among many candidates is `colosseum tournament` (SP4), which shares eval's player loading and `play_lineups`

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
- Snapshot retention per agent (`colosseum.arena.retention`): `checkpoint.keep_last` + `keep_every` + the current run's final snapshot + pinned snapshots (`colosseum pin`, `tournament --pin`) + `keep_top_k` by the live BT rating's lower bound (with protection of new snapshots until `top_k_min_games`), capped by `max_snapshots` (`trainer_state.pt` only in the `keep_last` window; a run-dir resume carries the pool and the pins over; finals of parent runs are ordinary snapshots)
- Checkpoints stored on shared filesystem or object store (today: `runs/<name>/checkpoints/<agent>/ckpt_v<N>/`)
- Checkpoint = state_dict + optimizer_state + training_step + metrics snapshot (today: `model.pt`, `trainer_state.pt` with optimizer, LR progress, AMP scaler, kickstart and counters, `meta.json` with `policy_version`, `env_steps`, roles and role signature, and since SP4 `run_id` and `pinned`; no metrics snapshot)

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
- Match log (SP4): every training, tournament and pulse match in `runs/<name>/matches.db` (SQLite, bounded: a detail window plus compacted counts)
- Dashboard (SP4): `colosseum report` (one self-contained HTML file) and `colosseum dashboard` (the same page live, stdlib `http.server`, 127.0.0.1): ratings by version, results matrix, anchors, opponents, training, system, matches, tournaments; console progress, `metrics.jsonl` and optional WandB stay

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
# the built-in colosseum.league.MixtureMatchmaker, pluggable via `matchmaking.matchmaker_class` (a
# colosseum.league.BaseMatchmaker subclass), GRPCTransport / LocalTransport):
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

State after SP4 (selection and observability): implemented on `sp4-selection`, acceptance pending (T8.5); `main` holds SP3 (accepted by the owner and merged on 2026-10-15; acceptance report 2026-10-10; SP2 was merged on 2026-10-09). Test suite on `sp4-selection`: **2121** fast tests, **14** slow, **25** GPU-only (`pytest -m "not gpu and not slow"` is the CI suite). The GPU checks (`docs/GPU_CHECKS.md`) were run on 2026-10-15 on an RTX 4070 Ti and merged into `main` the same day (report `docs/superpowers/reports/2026-10-15-gpu-checks.md`).
- SP4 records: spec `docs/superpowers/specs/2026-10-16-sp4-selection-design.md` (Russian; approved 2026-10-16); plan `docs/superpowers/plans/2026-10-16-sp4-selection/` (`00-overview.md` has the interface contract and amendments A1–A33; parts 01–03 have the tasks); the pre-flight rulings P1–P22, the controller rulings C1–C14 and the ledger rulings live in the git-ignored SDD workspace until the acceptance report `docs/superpowers/reports/<date>-sp4-acceptance.md` (written by T8.5) copies every one of them.
- SP3 records: spec `docs/superpowers/specs/2026-10-10-sp3-league-design.md` (Russian); plan `docs/superpowers/plans/2026-10-10-sp3-league/` (`00-overview.md` has the interface contract and amendments A1–A33; pre-flight rulings P1–P18); acceptance report `docs/superpowers/reports/2026-10-10-sp3-acceptance.md` (criteria, measurements, every ruling, open items, every deferred minor in its appendix); final review and fix-wave records `docs/superpowers/reports/2026-10-10-sp3-review/`.
- SP2 records: spec `docs/superpowers/specs/2026-10-08-sp2-game-model-design.md` (Russian); plan `docs/superpowers/plans/2026-10-08-sp2-game-model/` (`00-overview.md` has the interface contract and amendments R1–R18); acceptance report `docs/superpowers/reports/2026-10-09-sp2-acceptance.md` (criteria, measurements, every ruling, open items, all deferred task minors in its appendix); final review, fix wave and FIX-1/FIX-2 records `docs/superpowers/reports/2026-10-09-sp2-review/`.
- SP1 records: `docs/superpowers/reports/2026-10-08-sp1-acceptance.md`; the pre-SP1 review `review/README.md` is a frozen snapshot (never edit it).

Commands (run from the repo root after `scripts/setup-dev.sh`; the cwd is put on `sys.path`, also for spawned children):
- `colosseum validate -c cfg.yaml [--set k=v ...]`: `GameSpec` and config checks (roles, matchmaking), `reset` of every enabled layout and random legal steps with full contract checks, `step` and `unroll` of every agent's model, the kickstart teacher; every scripted agent played under the legality gate, every frozen agent loaded, `init` sources and a custom matchmaker checked; prints each agent's effective mix. SP4: checks that `ratings.anchor` names an agent of the config and, with `evaluation.enabled`, the pulse plan (field, agents, layouts, role coverage); prints a `match log:` line (on with its window / free-disk / compaction settings, or off), a `ratings:` line (refit interval, the anchor rule, bootstrap samples), each trainable agent's retention rule (`agent '<id>': snapshots: ...`, plus a WARNING when the pinned snapshots a run-dir resume carries over plus `keep_last` exceed `max_snapshots`) and a `pulse:` line.
- `colosseum train -c cfg.yaml [--set k=v ...]`: single-machine training into `runs/<name>/`.
- `colosseum eval -c cfg.yaml -a name=path [-a name ...] [--layout L ...] --num-matches N --output result.json [--num-envs E] [--deterministic] [--seed S] [--set k=v ...]` (a bare `-a name` plays a scripted or frozen agent of the config).
- `colosseum record -c cfg.yaml --player P [--against A ...] --num-matches N --output DIR [--layout L ...] [--num-envs E] [--seed S] [--deterministic] [--set k=v ...]` (P, A: a scripted/frozen agent or `name=path`; BC data per role + `record.json`).
- `colosseum bc -c cfg.yaml --agent A --data <file|dir|record-dir> [--data ...] --output bc.pt [--epochs E] [--batch-size B] [--lr LR] [--seq-len L] [--seed S] [--set k=v ...]` (`--data` repeatable; a `record` output dir contributes the folders of the agent's roles).
- `colosseum tournament [-c cfg.yaml] [--run [alias=]runs/x [--agent A] [--snapshots all|last:N|top:N|every:K]] ... [-a name=path | -a name] ... [--field a,b,...] [--mode gauntlet|round-robin|halving|select] [--layout L ...] [--blocks B] [--halving-blocks B] [--confirm-blocks B] [--finalists F] [--seed S] [--num-workers W] [--num-envs E] [--deterministic] [--name NAME] [--output DIR] [--pin] [--set k=v ...]`: candidates against a fixed field on common seeds with seat rotations; reports in `runs/x/tournaments/<name>/` (without `--run`: `--output DIR`); the same `--name` continues; exit 0 also for `confirmed: false`.
- `colosseum pin RUN [AGENT@vN] [--remove] [--list]`: pin / unpin / list stored snapshots (no config).
- `colosseum report RUN [-o FILE]`: one self-contained HTML report of a run dir (or of a tournament dir); `colosseum dashboard RUN [--host 127.0.0.1] [--port 8765] [--refresh 10]`: the same page live, `/api/<section>` JSON, no authentication (SSH tunnel).
- `record`, `bc`, `eval` and `tournament` validate in the "play" scope (`validate_config(scope="play", players=...)`): only the env, the trainable agents' models and the scripted / frozen agents the command seats, so one config can name the BC output (`init.from`, a frozen `bc_net`) before it exists; `validate`, `train` and the distributed roles run the full check.
- Distributed (limited): `colosseum serve-weight-store --port P [--max-message-mb M]`, `colosseum run-learner -c cfg.yaml --agent A --traj-port P --weight-store host:port`, `colosseum run-workers -c cfg.yaml --weight-store host:port -l A=host:port`.
- Measurements: `scripts/bench_throughput.py` (`--clock counters`, the default since SP4, works on WSL2's jumping wall clock; `--tree PATH` measures another checkout), `scripts/units_experiment.py` (`--cells ratio_mode:unit_trace ...` for the SP3 defaults rule), `scripts/measure_global_state.py`, `scripts/team_tag_anchors.py`, `scripts/pipeline_vs_scratch.py`; SP4: `scripts/measure_tournament_defaults.py` (tournament sizes on the bot ladders), `scripts/bt_crossval.py` (BT prior / smoothness by cross-validation), `scripts/measure_bt_cost.py` (refit cost at ~2 000 entities), `scripts/measure_pulse_cadence.py` (pulse cadence and the pipeline thresholds) (results in `docs/benchmarks.md`, raw data in `docs/benchmarks/sp4-*.json`).
- Docs: `docs/ENV_GUIDE.md` (writing an env; §10 `infos` for bots), `docs/LEAGUE_GUIDE.md` (players, league, warm start recipes), `docs/ARENA_GUIDE.md` (match log, ratings, tournament and the submission recipe, snapshot retention, pulse, report and dashboard, common mistakes), `docs/benchmarks.md`, `docs/GPU_CHECKS.md` (all Russian).

### Works (single machine)
- **Game model (SP2):**
  - env contract `GameSpec` (roles, layouts = lists of seats with roles and teams) + `MultiAgentEnv` + `StepResult` (`acting`, `terminated`, `truncated` + `final_obs`, `global_state`, `Outcome`); `EpisodeTracker` checks every rule and raises `EnvContractError` with "worker, env, seat, episode step, layout";
  - outcome kind from the number of teams (1 → `score`, 2 → `wdl`, ≥3 → `rank`); default outcome = mean of a team's seat returns;
  - seat lifecycle: empty / live / eliminated; a dead teammate is removed from `acting` but keeps team rewards; steps without decisions bounded by `env.max_idle_steps`;
  - observation, action, mask and `global_state` trees with preserved dtypes (`uint8` reaches the model); bool masks only; entity lists with masks; `Units(max_units, per_unit, only_if)` with `Discrete`/`MultiDiscrete`/`Box`/`Dict` components; deciders (each unit + the non-unit part).
- **Training loop:** IMPALA-style; one `MatchRunner` core for training (`RolloutLoop`) and eval. Workers infer only the policy; chunk v2 with `act`/`boot`/`pad` slots; the learner computes every value (bootstrap from `boot` slots and `final_obs`) with its current network. Only numpy payloads cross process boundaries. Every learner update uses exactly `learner.batch_chunks` chunks; `training.total_timesteps` is a global env-step budget.
- **Algorithm:** APPO with V-trace over slots; `ratio_mode` (`joint` / `per_unit`; `auto` = `per_unit` with `Units`), `unit_trace` (`joint` / `geo_mean` / `none`; `auto` = `joint` for every action), `entropy_reduction` — the `Units` defaults were kept by SP3's pre-fixed multi-seed rule (T6.5: at K=128 `joint` ratio +0.037 < 2·SE 0.092, `unit_trace: none` −0.005; `docs/benchmarks.md`); with one decider every mode collapses to `joint` (except an explicit `unit_trace: none`, ruling PR-1); every loss and metric reduces over `act` slots; per-unit diagnostics (`clip_fraction_joint`, `ess`, `log_rho_abs_p95`, `deciders_valid_mean`, ...). Centralized critic: `networks.critic_encoder_class` over `global_state`, value path only.
- **Models:** `PolicyModel` protocol (`step` for the policy path, `unroll` for the learner, `with_value=False` for BC and the kickstart teacher); `ComposedModel(encoder, core, policy, value, critic_encoder)` with `EncoderOutput(latent, aux)` for per-unit features (under CUDA autocast it runs the policy head in float32, ruling G-R6); cores `NoCore`, `LSTMCore`, `GRUCore`, `WindowAttentionCore`; `UnitsHead`, `gridnet_to_units`; `NormalizeObs` on a leaf path. The learner reproduces the worker's joint and per-unit log-probs for all four cores (contract tests).
- **Players (SP3):** agent kinds `trainable` / `scripted` / `frozen` (`agents.<id>.kind`); an implicit trainable `agent_0` when the config has none; `colosseum.players.ScriptedBot` (`reset(role, seat, layout, rng)`, `act(obs, mask, info)`, one instance per (agent, env, seat)) and the built-in `RandomBot`; fixed players sit on the `fixed` network id and never collect; bots get `StepResult.infos[seat]`; a bot's illegal action or exception is a `PlayerError` with "worker, env, seat, episode step, layout, agent"; frozen agents of any architecture from a checkpoint dir (meta.json) or a `.pt` (+ `networks` / `roles`); SP2 checkpoints load as frozen agents, `init` and for resume. Demo bots: `examples/tic_tac_toe/bots.py`, `examples/unit_harvest/bots.py`.
- **League:** lineups (layout + seat assignments); owner rotation (config order, from the player registry); layout weights; the v3 mix `matchmaking.opponents` (`latest` / `snapshots` / `rivals` / `anchors`; numbers or piecewise-linear schedules over env steps; empty categories spread proportionally, a `fallback` source when all positive ones are empty), `anchors` weights, `pfsp` (`hard` / `balanced` / `uniform`, EMA with `halflife_games`, per player and layout, over latest weights and snapshots of every agent that plays the team; opponents' snapshots in asymmetric games); per-agent `matchmaking` overrides; SP2 knobs translated with one warning (exact, shares rounded to 12 decimals); `colosseum.league.BaseMatchmaker` + `MatchmakerContext` for custom matchmakers (every lineup checked); teams with a core and `teammates: self | mixed`; roles per agent; seat permutation within equal role compositions; `validate` prints each agent's effective mix per layout. Ratings per layout (`ratings.json` = `{"env_steps", "layouts": {...}}`): team-pair ELO, win-rate matrix, `wr_vs_past`, `ScoreTracker` and cross-play, scripted/frozen agents as entities of their own, the PFSP table; played opponent shares per category and anchor in the `episodes` records.
- **Snapshot storage (SP3):** `checkpoint.keep_last` / `keep_every` / the final snapshot (since SP4: the current run's final; SP4 retention below); `trainer_state.pt` only in the `keep_last` window; evictions reach the workers (models unloaded) and the PFSP tables; a run-dir resume carries the stored snapshots over (hard links, else copies); strict resume with roles and role signature in `meta.json`.
- **Warm start (SP3):** per-agent `init` (`from`: `.pt`, checkpoint dir, run dir or a frozen agent; `strict` or partial with a report in the log and `validate`; weights only, policy version 0); `critic_warmup_steps` (value path only: the policy and the observation-normalizer statistics stay bit-identical, the critic's global-state normalizers keep learning; fix-wave ruling C-I2); kickstart per agent from a neural teacher (KL per decider; a frozen agent or checkpoint dir brings its own architecture, a bare `.pt` path is built with the student's; a recurrent teacher only with the student's state layout) or a scripted teacher (DAgger: workers write `teacher_action` / `has_teacher` while `teacher_active`); `training.resume_from` takes precedence over `init`.
- **record / BC:** `colosseum record` (any player kind, with or without `--against`, per-role BC files with contiguous seat-episodes, `record.json` with an eval summary); `colosseum bc` with repeatable `--data` and `record` output dirs.
- **Eval:** `MatchRunner`-based; CLI lineups per SP2 spec block 8; `-a name` for scripted/frozen agents of the config; reports for `wdl` (Wilson intervals, per side), `rank` (mean rank, first places, pairwise matrix), `score` (mean, cross-play), by role; in-process `play_lineups` with any `PolicyModel` and `ScriptedPlayer`s.
- **BC:** BC on observation/action trees (`-log_prob`, mean over valid deciders with `Units`; an illegal masked expert action is a `DataError`).
- **Demo games** (`examples/`, configs in `configs/examples/`): `coin_grid` (solo), `tic_tac_toe` (1v1 turn-based; also `tic_tac_toe_attention.yaml`, `tic_tac_toe_multi.yaml`), `unit_harvest` (units), `team_tag` (2v2, `global_state`), `tron` (FFA 2–4, elimination), `predator_prey` (asymmetric roles), `coop_buttons` (cooperative); reference examples `space_miners` (Box2D) and `composite_action` (chase). Each demo game has a slow learning test against random players (`tests/learning/test_demo_learning_slow.py`; team_tag is a single run again since SP3: its config trains against a `RandomBot` anchor, `opponents.anchors` 0.2 by the T6.3 measurement, played share ≈0.36); `configs/examples/unit_harvest_league.yaml` (agent `main`, anchors `greedy` and `random`) drives the README pipeline and a slow pipeline test (`tests/learning/test_sp3_pipeline_slow.py`; thresholds by T6.4: win vs random ≥ 0.95, score vs greedy ≥ 0.40, vs the BC net ≥ 0.45); fast learning tests cover a Units bandit, a cooperative bandit, a scripted-teacher kickstart and a `RandomBot` anchor, and a fast tic-tac-toe pipeline smoke (record → bc → train with init/warm-up/kickstart → eval, all from one config). Example configs are in the SP3 form (exact translations of their SP2 form; SP2 copies under `tests/fixtures/sp2/configs/`). `docs/ENV_GUIDE.md` (Russian) explains how to write an env for every game type.
- **Observability:** `runs/<name>/` (`run.dir`/`run.name`; default name `<config stem>-<YYYYmmdd-HHMMSS>`; an existing explicit name is refused) with `config.resolved.yaml`, `logs/` (`main`, `learner-<agent>`, `worker-<i>`, `worker-<i>-env<k>`), `metrics.jsonl` (`train` / `episodes` / `ratings` / `system`, with per-layout and per-role breakdowns; `system` includes `dropped_reward_episodes`), `ratings.json`, `checkpoints/<agent>/ckpt_v<N>/{model.pt, trainer_state.pt, meta.json}`; since SP4 also `run.json`, `run_state.json`, `matches.db`, `league_state.json`, `pulse_state.json` / `logs/pulse.log` (with pulse), `report.html` (`colosseum report`) and `tournaments/<name>/`; console progress per agent (with `bt <rating> ± <half CI>` since SP4); optional WandB. A seat that never acts in an episode loses that episode's rewards: one WARNING per worker, then counted.
- **Config:** Pydantic with `extra="forbid"`; partial agent overrides (`agents.<id>.{roles,networks,algorithm,learner,matchmaking,init,kickstart}`) merged per section onto the global sections (`matchmaking` via `merge_matchmaking`: `opponents` / `pfsp` per key; a share's schedule, `anchors`, `layouts` and the teammate knobs replaced whole); SP2 knobs (`matchmaking.mode` / `self_play_ratio` / `latest_prob` / `pfsp_exponent`, `checkpoint.pool_size`, `training.kickstart_*`) accepted in the input YAML only, translated with one warning each, never stored; old + new knob together = `ConfigError`; agent ids `[A-Za-z0-9_][A-Za-z0-9_-]*`, not `ratings`/`system`/`episodes`/`train`; `--set` with YAML values for `train` / `validate` / `record` / `bc` / `eval` / `tournament` / `run-learner` / `run-workers`; SP4's `match_log`, `ratings`, `evaluation` sections and new `checkpoint` keys are global only; `self_play.*` became `matchmaking.*` + `checkpoint.*`, `env.num_players` and `training.phase` are gone.
- **Lifecycle:** exit codes 0 (budget reached) / 1 (config error or a dead child) / 130 (SIGINT, also during startup and for `serve-weight-store`) / 143 (SIGTERM); `eval` / `record` / `bc` / `tournament` / `report` / `dashboard` / `pin` also 2 for bad arguments (`bc` has no 143); `tournament` 0 also for `confirmed: false`, 130 keeps finished blocks. Children ignore SIGINT and die with the parent; a dead child stops the run with a pointer to its log; final checkpoints on every stop within `SHUTDOWN_GRACE_SEC = 7`; no process alive after 10 s. Explicit resume is strict (`ConfigError` on a missing or malformed `meta.json` or a role-signature mismatch; SP1 checkpoints have no role signature and cannot be resumed).
- **Throughput:** 4911 / 9244 / 14107 env steps/s with 1 / 2 / 4 workers on tic-tac-toe (SP3 branch after the final fix wave, mean of two runs; 106.8 % / 99.0 % / 98.2 % of `main` measured alternately on the same machine at the acceptance, criterion 3.5; SP2 at acceptance: 4848 / 9298 / 13702; `docs/benchmarks.md`). SP4 (by counters on the 20-core WSL2 machine, CPU learner, journal on, pulse off): 5119 / 6697 / 13058 env steps/s = 101.9 / 101.0 / 96.7 % of `main` (`1366bec`) measured alternately (criterion 3.6; `docs/benchmarks.md`, «После SP4»).
- **Match log (SP4):** `runs/<name>/matches.db` (stdlib SQLite, WAL; `colosseum.arena.matchlog`): every training, tournament and pulse match (`MatchResult.uid` / `source` / `seed` / `env_steps` / `block`, per seat `version_start` / `version_end` and the rating version computed by the journal) written by a writer thread of the main process (bounded queue, one transaction per batch, flush at stop within `SHUTDOWN_GRACE_SEC`); detailed training matches within `match_log.detail_window_mb`, older ones (evicted snapshots first) compacted into team-pair `counts` that keep the BT point estimate and Hessian exact; `min_free_disk_mb` switches new training matches to counts only; a run-dir resume imports the parent's counts, tournament and pulse matches; every lost result is counted (`dropped_match_results`, WARNING on the first); read-only `MatchLogReader`; workers send results in batches (up to 256 results or 0.1 s; FX1), the results queue holds 1 000 batches.
- **Snapshot ratings (SP4):** `colosseum.arena.bt` — Bradley–Terry per layout on team pairs (weight `1/(T−1)`, draws 0.5), team strength = mean + side bias, Gaussian prior (`PRIOR_SIGMA` 8 logits) and lineage smoothness (`SMOOTHNESS` 1), both by cross-validation (T8.2 ruling); entities `agent@v<bucket>` / snapshot versions / fixed agents by name; role groups with bridges, connected components, an anchor per cell (`ratings.anchor`, else the first `RandomBot`, else the mean of fixed agents, else `v0`); Laplace and bootstrap (by matches for training, by blocks for pulse / tournament) intervals, the lineage-contrast `LCB`, `P(best)` / `P(top-k)`, the G² transitivity flag; `score` layouts as mean score with CI. Live: `arena.live.LiveRatings` refits every `ratings.refit_interval_sec` in a thread over the in-memory comparison table (one BLAS thread in the main process; a final refit after the journal writer flushed, skipped when the previous refit took over `FINAL_REFIT_MAX_SEC` = 2 s) into `ratings.json` → `bt`, the `ratings` records (compact) and the console.
- **Tournament (SP4):** `colosseum tournament` (`colosseum.arena.tournament`): candidates from `--run` snapshots (`all` / `last:N` / `top:N` / `every:K`; several runs with aliases), `-a name=path` (a checkpoint dir of any architecture, or a `.pt` built with a trainable entry's or the global `networks`; entity `name#<hash8>`) and `-a name`; a field (default every scripted / frozen agent); blocks = one seed + one composition + every admissible cyclic seat rotation, common seeds for every candidate, seed domains A / B / P disjoint by construction; FFA compositions, asymmetric roles by role group, several layouts with `matchmaking.layouts` weights, `score` layouts; modes `gauntlet` / `round-robin` (two-team layouts) / `halving` / `select` (halving on A, confirmation on fresh B; paired stratified bootstrap; `confirmed` or the indistinguishable finalists); repeated deterministic games flagged; BT of the tournament's own matches; reports `report.{json,txt,html}` + `tournament.yaml`; resume by `--name` (settings compared, whole blocks reused); runs next to training (WAL, snapshot hard links); `--pin` of a confirmed winner snapshot (ruling C6). Defaults `--blocks` 50 / `--halving-blocks` 8 / `--confirm-blocks` 100 / `--finalists` 3 (T8.2 ruling, bot ladders); ladder bots `examples/tic_tac_toe/ladder.py`, `examples/tron/bots.py`. Shares player loading with `eval` (the SP3 duplicates removed: one `-a` parser, one rounding rule, frozen agents loaded once, no `.pt` overlay of a frozen entry on a user path).
- **Retention (SP4):** `colosseum.arena.retention.retain` (pure): always `keep_last` / the current run's final / pinned / the just-saved snapshot; `keep_every`; `keep_top_k` by the live rating's `LCB` (`top_k_quantile`) with protection of new snapshots until `top_k_min_games` rated games; the cap `max_snapshots` (evicts keep_every-only, then new-only, then top-k; WARNING once when the protected exceed it); before the first live refit (with top-k) nothing unrated is evicted and the cap is 2 × `max_snapshots` (ruling, a spec deviation for the owner); applied on every save and after every refit; finals of parent runs are ordinary; `meta.json` `run_id` / `pinned`; `colosseum pin`; the distributed learner caps FIFO.
- **Pulse (SP4):** `evaluation` section; a `pulse` child process (worker lifecycle, `logs/pulse.log`) plays every `every`-th saved snapshot (numpy weights from the main process) as a gauntlet against `evaluation.field` on the fixed seed domain P; results (`source = pulse`, the snapshot's `env_steps`) go to the journal only (not to the online ELO / PFSP); a backlog of `max_backlog` drops the oldest (INFO, `pulse_skipped_snapshots`); the final snapshot of a local run stays pending at the stop (ruling: never evaluated, not counted as skipped); `pulse_state.json` (not resumed: ruling).
- **Report and dashboard (SP4):** `colosseum.arena.report` — one JSON dict per section (summary, ratings, matrix, anchors, opponents, training, system, matches, tournaments) from a run dir (read-only; order by counters; bounded: 2 000 series points, 2 000 entity rows, 1 000 matches, 100 snapshots in the matrix; SP3 run dirs and running runs); `colosseum report` embeds it in one HTML page with own CSS / JS / SVG (no network; phone layout, light / dark); `colosseum dashboard` serves it live (`http.server`, one builder thread, `/api/<section>` with filters, 127.0.0.1 by default, a WARNING off loopback, a Host-header guard); BT from `ratings.json` while running, else a bootstrap fit of the journal cached until it changes.
- **Run and league state (SP4):** `run.json` (run id), `run_state.json` (`running` / `finished` / `failed` / `stopped`, pid + Linux start ticks; readers derive `stopped` for a dead pid); `league_state.json` (ELO, win rates, `wr_vs_past`, PFSP with forgotten snapshots; written each metrics tick and at the end; loaded by a run-dir resume).
- **Metrics (SP4):** `train` `rho_trunc_mass` / `c_trunc_mass` (`Σ max(ρ − ρ̄, 0) / Σ ρ` over ACT slots; exactly 0 on-policy) replace `rho_clip_frac` / `c_clip_frac` (owner's Q2); `system` `dropped_reward_episodes` monotonic per run, `dropped_match_results`, `match_log` (writer counters, sizes, `degraded`), pulse counters; `ratings` records carry `pfsp_summary` instead of the full PFSP table and a compact `bt`; the ratings snapshot is built only on a metrics tick; `colosseum bc` logs its device.

### Partial
- **Distributed mode** (`serve-weight-store`, `run-learner`, `run-workers`) runs on chunk v2 for games where every agent plays every role, latest weights only: no coordinator, league, ratings, `metrics.jsonl` or WandB; per-worker budgets; `run-learner` ignores `training.resume_from`. SP3 features are local only: scripted and frozen agents, `init`, critic warm-up, a custom matchmaker, per-agent `matchmaking` or `kickstart`, a kickstart teacher other than one global `.pt` with the student's architecture are refused with a `ConfigError` pointing to SP5 (`distributed.check_distributed_scope`), and any opponent mix is reduced to latest weights with one warning (which also names an ignored `teammates: mixed`). SP4: no match log, snapshot ratings, tournament or pulse there; `evaluation.enabled` and `checkpoint.keep_top_k > 0` are refused the same way, `checkpoint.max_snapshots` works FIFO in the distributed learner.
- **`deployment/`** (Docker, K8s) is not tested.
- **Unused, kept for SP5:** `transport/local.py::LocalTransport`, `weight_store/shared_memory.py::SharedMemoryWeightStore`, `transport.mode` / `transport.grpc_port`.
- **GPU:** checked on 2026-10-15 on an RTX 4070 Ti (WSL2, torch 2.14.1+cu130; report `docs/superpowers/reports/2026-10-15-gpu-checks.md`): all `gpu` tests pass (22 after the TF32 fix of `test_state.py`, ruling G-R1: torch's TF32 defaults are kept; 25 with the fp32-head tests), the fast suite passes on the CUDA machine (only its `configs/examples/` integration tests train a CUDA learner; `write_test_config` pins `learner.device: cpu`) and the slow suite passes with CUDA learners (423.96 s), the README pipeline under AMP float16 and bfloat16 passes its thresholds, AMP resume, CUDA checkpoints on CPU and the distributed learner on CUDA work. AMP numerics (M-6): the policy head's `mean` in fp16/bf16 gave zero-lag log-ratio noise for `Box` actions that grows as σ shrinks (≈1/σ at small σ; bf16 p95 up to 0.19 at σ = e^−4); since ruling G-R6 `ComposedModel` runs the policy head in float32 under CUDA autocast (bf16 p95 at most 0.10, fp16 at most 1.29e-2; the rest is the encoder's AMP rounding), measured by `scripts/measure_amp_logratio.py` (`--amp-head` for the old behaviour; `docs/GPU_CHECKS.md`). Caveats: workers still infer on CPU; on the RTX 4070 Ti AMP is not faster than fp32 for the example-size models (+1–13 % per learner step with the head under autocast, +3–16 % with the fp32 head; LSTM-512 ≈+50 %), so `algorithm.use_amp` stays false unless AMP measurably speeds a model up (`docs/benchmarks.md`).

### Not implemented (see Roadmap)
- Distributed league with scripted/frozen players and SP3 warm start (SP5); recurrent kickstart teacher with another state layout, batched scripted-bot API, exploiters, preset mixes, value pre-training on BC data, a one-command phase pipeline (later, as needed; SP3 spec §2).
- Not planned at all (SP2 spec §2): one agent on roles with different spaces; per-unit rewards, values or recurrent state; built-in autoregression between action components (a custom `Distribution` can do it); match series as a framework entity (best-of-N lives inside the env); MCTS controllers and communication channels between bots; canonical perspective and augmentations (the env's job).
- A PettingZoo adapter (later, if needed).
- Out of SP4's scope (SP4 spec §2; later, as needed): Swiss / king-of-the-hill / playoff tournaments, an SPRT submission gate, duelling bandits; α-Rank, Nash averaging, mElo or other cycle models in the rating core (the report has the results matrix and the worst opponent); the Davidson draw model and exact Plackett–Luce for FFA (draw = 0.5, FFA by team pairs); strategy diversity in top-k; exploiters, PBT, weight averaging (EMA / SWA) as the published policy; replay storage (the journal has a `replay` column, but no env API fills it yet). OpenSkill / TrueSkill as a separate online rating is not planned (the live rating is Bradley–Terry on a timer). Distributed tournament, pulse, match log and ratings in a hub: SP5.
- Off-policy algorithms (R2D2/DQN, replay buffer), SAC, AlphaZero/MuZero (SP6 "new algorithms" or later).
- GPU inference on workers; a vectorized worker path for envs with hundreds of seats (SP6).
- Shared-memory weight store and trajectory ring buffer; adding or removing agents and machines during a run; gRPC control plane, distributed league and asymmetric games across machines (SP5).

## Roadmap

| Sub-project | Scope |
|---|---|
| **SP1 Foundation and stabilization** (done) | correct, observable, robust single-machine training; `PolicyModel` protocol |
| **SP2 Game model** (done) | `GameSpec` / `MultiAgentEnv`, elimination, teams, roles, layouts, `Units`, Dict observations, bootstrap on the learner, centralized critic, demo games |
| **SP3 Players, league, warm start** (done) | scripted and frozen players (also as `record` data sources and DAgger teachers), the v3 opponent mix (latest / snapshots / rivals / anchors, schedules, per-agent overrides, custom matchmaker) with PFSP over snapshots (including the opponent's in asymmetric games; `hard` / `balanced` / `uniform`), a `RandomBot` anchor against passivity (team_tag passes in one run), snapshot retention (`keep_last` / `keep_every` / final, pool carried over on resume), per-agent warm start (`init`, critic warm-up, kickstart neural or DAgger), `colosseum record`, APPO `Units` defaults re-checked |
| **SP4 Selection and observability** (in progress: implemented on `sp4-selection`, acceptance pending; parallel with SP5) | match log, Bradley–Terry snapshot ratings with intervals (live and bootstrap), `colosseum tournament` with two-stage selection, top-k / pinned / capped snapshot storage, pulse (evaluation during training), `colosseum report` / `colosseum dashboard`, the deferred metric items (truncated V-trace mass, monotonic drop counters, compact `ratings` records, league state over resume, finals no longer bloat the pool) |
| **SP5 Distributed** (parallel with SP4) | hub and nodes, distributed league (including asymmetric games, agents that do not play every role, scripted/frozen players and SP3 warm start), adding / removing agents and machines during a run, shared-memory weight store and trajectory ring buffer, wire format, per-machine weight cache, fault tolerance, `max_policy_lag`, K8s images |
| **SP6 Speed and extensions** | vectorized fast path for array envs and hundreds of seats (Neural MMO), cuDNN RNN path, fast transformer unroll, GPU inference on workers, inference server, new algorithms |

### Open items parked during SP1–SP4
- **SP4 (done):** paired, rating-based checkpoint selection instead of per-pair Wilson intervals — `colosseum tournament` (paired stratified bootstrap of blocks, Bradley–Terry with bootstrap; SP4).
- **SP5:**
  - distributed learner resume (`run-learner` never applies `training.resume_from`);
  - metrics hub / `metrics.jsonl` / WandB for the distributed roles (today: per-process logs only);
  - resource leak when `run-learner` setup fails after the trajectory server, weight-store client or drainer started;
  - one budget semantics for distributed mode (today `total_timesteps / num_workers` per worker; learner progress = `consumed_samples / total_timesteps`, which counts decisions, not env steps);
  - newest-wins weight-queue eviction unpickles the stale payload (CPU ∝ model size × workers) → per-machine weight cache;
  - the distributed learner's final checkpoint can be lost when the checkpoint drainer stops first and the 16-slot queue is full;
  - distributed league and games where one agent does not play every role;
  - distributed tournament, pulse, match log and snapshot ratings (the hub; SP4 made them transport-free libraries).
- **SP6:**
  - normalizer statistics are updated before the loss forward, so the ratio is not exactly 1 at zero lag on early steps;
  - `WindowAttentionCore` rebuilds its masks every step (cache with the fast path);
  - BC windows are not episode-aligned;
  - observation trees are handled per seat in Python on the worker (vectorized per-role arrays for envs with hundreds of seats).
- **Tier 2 algorithms:** replay buffer for R2D2/DQN (the replay path would also need its own normalizer-stat and warm-up handling).
- **Minor, unscheduled:**
  - `--set` lists follow YAML 1.1, so `--set x=[1e-4,2]` keeps `1e-4` as a string (write `1.0e-4` in lists);
  - stale parked rollout buffers have no age limit (bounded by agents × envs × seats);
  - config inheritance / profiles (`extends: base.yaml`);
  - a PettingZoo adapter (later, if needed; SP2 spec §2).
- **Parked during SP2** (from the SP2 acceptance report, «Открытые пункты»):
  - budget overshoot and resume: a local run overshoots `total_timesteps` by about one second of env steps; a resume needs a budget above the checkpoint's `env_steps` to train (optional warning when a resume starts at or past the budget) → SP5 "one budget semantics";
  - distributed workers get identical per-env seeds across machines when `training.seed` is set (a consequence of ruling PR-3: seeds are `training.seed + worker_id` on every machine) → SP5;
  - done in SP4: `system` `dropped_reward_episodes` is a per-run monotonic counter.
- **SP2 final-review Minors worth knowing** (full lists: the SP2 report's «Финальное ревью ветки» and appendix, and `docs/superpowers/reports/2026-10-09-sp2-review/`):
  - SP5: with several agents the distributed roles silently ignore `shuffle_seats` (SP3 now refuses players, overrides and custom matchmakers and warns on any opponent mix and on an ignored `teammates: mixed`); the learner trusts slot payload values from a peer (finite `behavior_logp`, zero rewards on `boot`/`pad`), and a chunk with a wrong observation shape fails only in the learner forward; `distributed_setup` duplicates `setup_run`'s spec/roles sequence;
  - GPU checks: Gaussian log-prob and entropy run in fp16 under AMP (also inside `Units`) — measured on 2026-10-15 (`scripts/measure_amp_logratio.py`, M-6 in the GPU checks report; the dominant error is the policy head's `mean`, see «Found by the GPU checks»);
  - SP6: `unit_log_prob` is computed twice per step (`act` and APPO's evaluate), the `Units` gate table is rebuilt per call;
  - unscheduled: `global_state` is stored and sent but unused when the agent has no critic encoder (a `validate` warning); `role_signature` ignores `n` / `nvec` of discrete observation leaves; all seats eliminated without `episode_over` is reported only after `max_idle_steps` idle steps; "masked units give no NaN gradient" holds only for finite parameters; `MatchRunner` trusts that every seat's role is one its agent plays.
- **Parked during SP3** (a selection: the most relevant deferred items by addressee; the complete list of deferred minors is in the appendix of the SP3 acceptance report):
  - done in SP4: snapshot ratings and top-k storage, the monotonic `dropped_reward_episodes`, the compact `ratings` records, finals of parent runs as ordinary snapshots, the ratings snapshot built only on a metrics tick, `colosseum bc` logging its device; the online ELO stays a progress indicator;
  - outside SP4 (SP4 spec §2: a training topic, not selection): show the value of warm start on a game where training from scratch does not reach the bot within the budget (on `unit_harvest` the pipeline ≈ from scratch: 0.483 vs 0.492 against the bot; `docs/benchmarks.md`);
  - SP5: the "reduced to latest only" distributed warning is emitted before file logging starts (console only) and also fires when only empty categories carry non-latest shares; likewise the SP4 INFO "distributed roles write no match log and no Bradley-Terry ratings" reaches only the console; `validate_chunk_payload` checks only presence and shape of the DAgger fields (`teacher_action`, `has_teacher`) — the peer-trust class above; the `pool_size` translation warning is printed before the run dir exists (not in `logs/main.log`);
  - code quality, no behaviour risk: `learner/factory.py` duplicates checkpoint discovery (`_CKPT_DIR_RE` vs `checkpoint_manager._CKPT_RE`, `classify_resume_source`); the teacher-bot handling in `RolloutLoop` parallels `MatchRunner`'s and duplicates `EpisodeTracker`'s episode bookkeeping; W/D/L logic in three places (`team_tag_anchors.wdl`, the team_tag slow test, `demo_learning.win_rate`); done in SP4 (T5.1a): one `-a` parser (`eval.parse_agent_spec`; `record.parse_player` and `cli._parse_agent_spec` are gone) and one odd-count rounding rule (`eval.balanced_match_count`);
  - minor, behaviour: a missing-checkpoint fallback to latest keeps `source="snapshots"` in the played shares (documented in LEAGUE_GUIDE); `validate` plays bots only in the enabled layouts and exercises a custom matchmaker only at env step 0 without snapshots; `validate` does not check resumed weights/signatures per agent (a broken run-dir checkpoint also hides that agent's `init` check); `PfspStats` does not validate its prior; integer action components accept `bool`; `FrozenSpec ==` raises (numpy; use `eq=False`); `_check_critic_warmup`'s signature check rejects `**kwargs`-forwarding APPO subclasses; shared value/policy parameters are not detected by the warm-up (documented contract); `record` leaves partial part files after a mid-run failure; the `unit_harvest` pipeline thresholds leave a 0.017–0.033 margin on the BC-net check (≈0.2–10 % flake estimate, ruling T6.4); team_tag anchor tie rule compares without float tolerance;
  - minor, cost: player roles and frozen `meta.json` resolved twice in `setup_run`, kickstart teachers and `init` weights resolved twice in `train` (validate + launcher), `_check_kickstart_teachers` rebuilds the student, `ObsSpec` rebuilt per bot decision in `validate`; done in SP4: frozen agents are loaded once by `eval` (`ValidationReport.fixed` reused);
  - minor, tests: no test of the snapshot copy fallback / hard-link proof; no test with a frozen or scripted TEAMMATE; no neural `Units` recording test; no `--against` test with an explicit one-team layout; no test for an invalid `.pt` networks override, a malformed `meta.json` via `resolve_player_roles`, `assert_no_tensors(FixedPlayers)`; no uint8-observation test for bots; `test_sp2_validate.py:159`'s name no longer matches its content; a few weak assertions (`match="agents"`, loose "KL", `loss > 0` for the recurrent-teacher state); `units_experiment.py` decides with n=2 if a seed fails and prints raw tracebacks for bad `--cells`.
  - final-review Minors not in the fix wave (records in `docs/superpowers/reports/2026-10-10-sp3-review/`): frozen agents used only as a teacher or `init` source are still built in every worker (memory × workers); played-share metrics are absent for teams that mix roles of several trainable agents, and anchor attribution can tie with `teammates: mixed`; `kickstart.kl: reverse` is silently ignored for a scripted teacher; a resume from a snapshot without `trainer_state.pt` repeats the critic warm-up and restarts kickstart λ (documented in LEAGUE_GUIDE §12); workers query a scripted teacher during the critic warm-up although the labels are ignored; `bc` does not compare `record.json`'s game / env kwargs with its config; done in SP4: a user-given `-a name=path` no longer takes a frozen entry's `.pt` overrides (eval, record, tournament), the `ratings` records carry a PFSP summary instead of the full table, and only the current run's final snapshot is protected (frequent stop/resume no longer grows the pool);
  - found at the acceptance: `colosseum bc` with torch's default 8 threads takes ~1.7× as long as with `OMP_NUM_THREADS=1` (16 s vs 10 s on the README pipeline; set the BC process's torch threads); the `coin_grid` slow test's ratio varies 3.8–5.4× between runs of the same code (threshold 2×).
- **Found by the GPU checks (2026-10-15)** (`docs/superpowers/reports/2026-10-15-gpu-checks.md`):
  - done (owner Q1, commit `a2d00a6`): `ComposedModel` runs the policy head in float32 under CUDA autocast (ruling G-R6; M-6 numbers in the GPU checks report);
  - done in SP4 (owner chose option 3, 2026-10-15): `rho_clip_frac` / `c_clip_frac` are replaced by the truncated mass `rho_trunc_mass` / `c_trunc_mass` = `Σ max(ρ − ρ̄, 0) / Σ ρ` (and with c̄) over ACT slots — exactly 0 on-policy; the fraction metrics counted numeric noise above ρ̄ = 1 (0.13–0.54 at zero lag on CUDA, 0.01–0.08 on CPU; median 0.41–0.52 in every real run);
  - far future (owner Q3, 2026-10-15): check whether a learner TF32 knob is actually needed; until then torch defaults (TF32 on cuDNN RNN/conv) apply;
  - SP6: AMP is slower than fp32 for the example-size models on an RTX 4070 Ti (+1–13 % per learner step with the head under autocast, +3–16 % with the fp32 head; LSTM-512 ≈+50 %); revisit AMP (or recommend against it) with the fast paths;
  - environment: WSL2 wall-clock jumps make `metrics.jsonl` `ts` non-monotonic on the owner's CUDA machine (since SP4 `scripts/bench_throughput.py` measures by counters by default; `--clock ts` still breaks there); ROS `PYTHONPATH` needs `env -u PYTHONPATH`;
  - done in SP4: `colosseum bc` logs its device.
- **Parked during SP4** (a selection from the SDD ledger's deferred items by addressee, refilled at the acceptance (T8.5); until then the complete list is in the git-ignored ledger `.superpowers/sdd/2026-10-16-sp4-selection/progress.md`):
  - done in SP4 (FX1): dropped match results on fast games — workers now send results in batches (up to 256 results or 0.1 s; the queue holds 1 000 batches); on the measured workloads none are dropped (the benchmark at 4 workers and the default tic-tac-toe config at 2 workers: 0 lost; `main` 1366bec lost 28–30 % / 74 % on the same workloads); drops on overload are still possible and counted in `dropped_match_results` with a WARNING; headroom beyond 4 workers is unmeasured (`docs/benchmarks.md`, «После SP4»);
  - rulings for the owner: before the first live refit (with top-k) the cap is 2 × `max_snapshots` (a spec deviation; up to `max_snapshots` extra snapshots for about one refit interval); the pulse never evaluates a local run's final snapshot (pending at the stop; `colosseum tournament` covers it); `pulse_state.json` is not resumed (the cadence restarts); `PRIOR_SIGMA` 8 sits at the edge of the cross-validation grid and rests essentially on one game: only unit_harvest_league improves towards the weak prior, tron and predator_prey are nearly flat in the prior, and the tic-tac-toe journal was pure self-play (weight 0; `docs/benchmarks.md`, «Приор и гладкость Bradley–Terry») — re-check on real competition journals;
  - SP5: the distributed "no match log" INFO is console-only (above); an aborted tournament's worker may put a large "done" message after the stop and be killed after `STOP_GRACE_SEC` (results discarded anyway);
  - journal / live ratings: the writer's abandon WARNING should say an abandoned batch may still commit (over-count bounded by `WRITE_BATCH`), `_take` counts the first item outside the lock (off by one at a 7 s stall), "the 0 results it was writing" wording when stuck in compaction, `_match_log_stats()` built on every monitor iteration; BT's side-bias count comes from the data's highest team index, not the layout; `_finish_metrics` joins the live thread again after `finish()` timed out (up to 2 × 7 s, daemon thread); `ratings_snapshot(compact=True)` unused in production; snapshot immutability by convention only; the GIL cost of a multi-second refit in the training main process on a large journal is not measured (the throughput benchmark's journal is small); `version_start` is read at reset, so an episode's first decision can use one sync newer weights (a docstring line "version at reset" is due in the final fix wave);
  - league state (T3.2): the "SP3 run dir" INFO also fires for an SP4 run dir that never wrote `league_state.json`; two INFO reasons are worded wrongly; after a SIGKILL of the parent, PFSP rows of snapshots evicted after its last tick survive the resume (never drawn; visible in `ratings.json` `pfsp`); `forgotten` grows without bound (spec-required, small); no end-to-end tests for import-evicts-restored-rows, a corrupt `league_state.json` (exit 1 before children), the save ERROR path;
  - retention / pin: a malformed `meta.json` reads as "has no stored snapshot" (the reason only in an unformatted stderr WARNING); an eviction racing `set_pinned` gives a raw `FileNotFoundError`; an empty `--list` prints nothing; no test of an idempotent unpin, nor that the trainer is alive after a live pin;
  - tournament: round-robin under `--deterministic` never detects repeated pairs (a zero-width interval without a mark); halving's ValueError for an active candidate without a score is untyped (unreachable through the CLI); (T−1)! orderings per composition for many-team asymmetric layouts; FFA blocks rotate cyclically, so neighbour positions are not balanced (spec-mandated; documented in ARENA_GUIDE); `_pin` words every pin error as "evicted by the run"; without `-c` the first run's config is read twice; an error in an unscheduled repeat episode after an env's last lineup names that lineup's uid; a dead `or '-'` in `_drop`;
  - eval / players: `MatchRunner.set_next_lineup` / `lineup()` docstrings do not mention `snapshot_fallback=False`; `load_player`'s reuse of `fixed.frozen` does not mark the architecture validated (a re-check costs time only); `--agent` named twice in a bad `-a` message;
  - pulse: a backlog-dropped final snapshot is reported at the end as "the run stopped first" (the earlier INFO has the right reason);
  - report: score-layout caps apply before the agent filter; `evaluation.agents: []` is listed as "every trainable agent unevaluated"; the ratings fit reads the journal fresh while matrix / anchors use the cached table (sections may differ by a few games); the matrix agent filter does not filter score cross-play; the run report embeds every tournament's report with no count limit (documented); the opponents section does not draw per-anchor shares;
  - benchmark: `CounterSampler` uses queue depth −1 for both "no system record yet" and "unknown"; under `--tree` the workload summary and bound classification use this branch's constants; no test of `run_one`'s unknown-clock ValueError; a sampler-thread exception surfaces away from the reported error;
  - tests / cosmetics: no test of the truncated masses under `unit_trace: geo_mean` or for a batch without ACT slots; `_diagnostics` takes `exp` of `trace_log_rho` in its own dtype (rounding level under AMP); the ladder docstring says z ≥ 3.9 / ≤ 5e-5 (it is z ≈ 3.87, ≈ 5.4e-5 per gap).

### Next step
SP4 acceptance (T8.5) and, in parallel, SP5 (distributed).
- SP4: after the slow tests (T8.3), this documentation (T8.4), the controller's drops diagnostic and the final whole-branch review with its fix wave, T8.5 checks every criterion of spec §3, writes `docs/superpowers/reports/<date>-sp4-acceptance.md` (Russian) with every ruling and the deferred minors, and stops for the owner; `sp4-selection` is merged into `main` (`git merge --no-ff`) only after the owner's explicit "yes", and then the Roadmap marks SP4 done.
- SP5 starts with a brainstorm and a written spec before any plan or code, on its own branch (`sp5-<name>`). Design inputs: the SP5 row of the Roadmap; the SP5 items of «Open items» (SP1–SP4); SP3 spec §2 and SP4 spec §2 «Вне рамок» (hub and nodes, distributed league, scripted / frozen players, anchors and SP3 warm start across machines; distributed tournament, pulse, match log and ratings in the hub); the SP1–SP4 rulings (acceptance reports). It starts from `distributed.check_distributed_scope` (the SP3 and SP4 features the distributed roles refuse today) and SP4's transport-free libraries (`colosseum.arena`: match log, Bradley–Terry, retention, tournament schedule and modes, report data), whose match-record schema is SP4's contract.
- The two SP3 fix-wave rulings that refine the SP3 spec stay in force (the owner may still revise them): the critic warm-up updates the value-path normalizers (ruling C-I2); `record` / `bc` / `eval` (and since SP4 `tournament`) validate in the "play" scope.
- GPU checks done on 2026-10-15 (`docs/superpowers/reports/2026-10-15-gpu-checks.md`); owner's answers: Q1 done (fp32 policy head, G-R6), Q2 done in SP4 (truncated importance-weight mass), Q3 parked for the far future, Q4 accepted.

## Development Workflow

- **One sub-project at a time**, in roadmap order (SP4 and SP5 may run in parallel). Each one goes: brainstorm → spec in `docs/superpowers/specs/` → implementation plan in `docs/superpowers/plans/<date>-<name>/` (overview with global constraints, interface contract and contract amendments, then parts with tasks) → implementation on a branch `spN-<name>` → acceptance report in `docs/superpowers/reports/` → merge.
- **`main` stays stable.** A sub-project branch is merged with `git merge --no-ff` only after the owner explicitly accepts the acceptance report, so one `git revert -m 1 <merge>` undoes it. Push the branch to `origin` regularly while working.
- **Implementation process (SP1):** one subagent per plan task (TDD, commit), a spec + quality review after every task, fix rounds with scoped re-reviews, then one whole-branch review with a single fix wave, then acceptance against the spec's §3 criteria.
- **SP2 was executed the same way**, with a pre-flight consistency scan of the plan (rulings P1–P22) and a shadow package `colosseum.sp2` built next to the old code and overlaid onto `colosseum.*` in task T7.3 (the shadow is gone now); an Opus/high implementer and an Opus/high reviewer per task (owner's instruction), three area reviewers for the final whole-branch review, then one fix wave and a scoped re-review (in SP2 after the acceptance, which re-ran criterion 3.1 after the wave).
- **SP3 likewise** (pre-flight rulings P1–P18, contract amendments A1–A33; final review by three area reviewers, one fix wave and a scoped re-review, all before the acceptance, so the acceptance measured the final code).
- **SP4 likewise** (contract amendments A1–A33, pre-flight rulings P1–P22, controller rulings C1–C14; an autonomous run; independent tasks in parallel git worktrees on local branches, integrated by cherry-pick with the full fast suite after each integration; T5.1 and T8.2 split in two dispatches each; measurements (T8.2) and slow tests (T8.3) only on a quiet machine, one at a time).
- **The SDD workspace** (`.superpowers/sdd/<plan>/`: briefs, ledger `progress.md`, review packages, reports) is git-ignored scratch and is deleted after the merge; anything durable (rulings, deferred items, review records) goes into `docs/` before the merge. The loop, the instruction templates (SP3 versions), `brief.sh` and the ledger line formats are in `docs/superpowers/process/` (at most 5 fix rounds per task; implementer status `DONE | DONE_WITH_CONCERNS | BLOCKED | NEEDS_CONTEXT`).
- **Decisions taken on the owner's behalf** are recorded as rulings ("what — why — cost if wrong"). The SP1 list (58 rulings) is at the end of `docs/superpowers/reports/2026-10-08-sp1-acceptance.md`; the SP2 list (plan-stage PR-1..PR-5, pre-flight P1–P22, the 42 ledger rulings, the T8.3–T8.4 rulings) is at the end of `docs/superpowers/reports/2026-10-09-sp2-acceptance.md`; the SP3 list (contract amendments A1–A33, pre-flight P1–P18, the ledger rulings of tasks T0.1–T6.6, the final-review and fix-wave rulings) is in the section «Решения контроллера во время SP3» of `docs/superpowers/reports/2026-10-10-sp3-acceptance.md`. The SP4 list (amendments, pre-flight and controller rulings, the ledger rulings) goes into the SP4 acceptance report (T8.5); until then it is in the SDD ledger. Check all lists before changing an area SP1–SP4 touched, and revise a ruling explicitly (with the owner) rather than silently undoing it.
- **Frozen material:** `review/` is the pre-SP1 review snapshot and SP specs/plans are history; never edit them to match new code.
- **Dev machine and tests:**
  - setup: `scripts/setup-dev.sh` (CPU-only torch from the PyTorch CPU index; `--gpu` for CUDA);
  - on a machine with ROS (`PYTHONPATH` set, e.g. the owner's CUDA machine) every `.venv/bin/python` / `.venv/bin/colosseum` / `.venv/bin/ruff` command needs the prefix `env -u PYTHONPATH` (ROS's pytest plugin breaks pytest at startup);
  - CI suite: `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` (must pass with zero warnings) and `.venv/bin/ruff check .`;
  - slow suite: `.venv/bin/python -m pytest -m slow -v` — 14 tests (the 7 demo-game learning tests in `tests/learning/test_demo_learning_slow.py`, the unit_harvest pipeline test `tests/learning/test_sp3_pipeline_slow.py`, 2 torch.compile checks in `tests/unit/test_sp2_performance.py`, and SP4's bot ladder on two games `tests/learning/test_sp4_ladder_slow.py`, pipeline with pulse and `select` `tests/learning/test_sp4_pipeline_slow.py` and disk bound `tests/integration/test_sp4_disk_bound_slow.py`), about 10 min on the owner's machine (567 s at T8.3); run it at acceptance and after any change to the training math, the worker or the learner, never next to another CPU-heavy job (policy lag, and with it every learning number, depends on the load);
  - test conventions: no `__init__.py` under `tests/`; import support modules by bare name (`from game_helpers import ...`, `from cli_runner import ...`, SP4's `report_kit`, `arena_kit`); test basenames and support-class names are unique; every file goes under `tmp_path`; integration runs start processes through `tests/cli_runner.py` (process-group cleanup; `report` / `dashboard` / `tournament` through `cli_runner.run_in_session`);
  - at most 2 worker processes in tests (tournament workers and the pulse process count; the benchmark may use 4); `OMP_NUM_THREADS=1`;
  - order data by counters (`env_steps`, `train_step`, versions, journal row ids), never by `ts`, and measure durations with `time.monotonic()`: the owner's WSL2 wall clock jumps; no test asserts on `ts` or wall-clock rates;
  - any config-schema change also updates `scripts/bench_throughput.py::_make_config` (pinned benchmark workload; guarded by `tests/unit/test_bench_throughput.py`).
- **Conventions:** commits use conventional prefixes (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`, `perf:`) and **never** carry attribution or `Co-Authored-By` lines, in commits or PRs; push after every task. Language: code, comments, docstrings, log messages, `README.md` and `CLAUDE.md` in English; specs, acceptance reports and the project docs `docs/ENV_GUIDE.md`, `docs/LEAGUE_GUIDE.md`, `docs/ARENA_GUIDE.md`, `docs/benchmarks.md`, `docs/GPU_CHECKS.md` in Russian (plans so far in English); the report / dashboard UI text and CLI help in English.

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

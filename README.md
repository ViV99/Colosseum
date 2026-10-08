# Colosseum

Distributed RL training framework for competitive bot programming competitions (Lux AI, Neural MMO, CodeCraft, etc.).

Full pipeline: **Behavioral Cloning -> Self-Play -> PFSP/League**, with IMPALA-style async architecture, multi-agent league training, composite action spaces (Dict/Tuple/MultiDiscrete), action masking, RNN/LSTM support, and distributed training via gRPC.

**148 tests passing.** Single-machine and distributed (gRPC + Kubernetes) modes.

---

## Installation

```bash
git clone <repo-url> && cd Colosseum
python -m venv .venv && source .venv/bin/activate
pip install -e .

# For distributed training (gRPC)
pip install -e ".[grpc]"

# For development
pip install -e ".[dev]"
```

Requirements: Python 3.11+, PyTorch 2.2+.

---

## Quick Start

### Train (Self-Play)

```bash
colosseum train --config configs/examples/tic_tac_toe.yaml
```

### Train (Multi-Agent League)

```bash
colosseum train --config configs/examples/tic_tac_toe_multi.yaml
```

### Evaluate Checkpoints

```bash
colosseum eval -c configs/examples/tic_tac_toe.yaml \
  -a agent_a:checkpoints/agent_0/ckpt_v100/model.pt \
  -a agent_b:checkpoints/agent_0/ckpt_v200/model.pt \
  --num-matches 1000
```

### Behavioral Cloning

```bash
colosseum bc -c configs/examples/tic_tac_toe.yaml \
  --data path/to/expert_data/ \
  --output pretrained_weights.pt \
  --epochs 20
```

### Override Config from CLI

```bash
colosseum train -c configs/examples/tic_tac_toe.yaml \
  --set training.total_timesteps=500000 \
  --set rollout.num_workers=8 \
  --set algorithm.learning_rate=1e-4 \
  --set metrics.use_wandb=true
```

---

## Writing Your Own Game

Implement 4 classes and a YAML config to use Colosseum for any competition.

### 1. Environment

```python
# my_game/env.py
import gymnasium
import numpy as np
from colosseum.envs.base_env import BaseEnv

class MyGameEnv(BaseEnv):
    @property
    def num_players(self) -> int:
        return 2  # supports 1-N players

    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Box(low=0, high=1, shape=(8, 8, 3), dtype=np.float32)

    @property
    def action_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Discrete(64)
        # Also supported: Dict, Tuple, MultiDiscrete, Box — see Composite Actions below

    def reset(self, seed=None):
        # Returns: (obs_dict, info_dict) keyed by player index 0..N-1
        obs = {i: np.zeros((8, 8, 3), dtype=np.float32) for i in range(self.num_players)}
        info = {i: {} for i in range(self.num_players)}
        return obs, info

    def step(self, actions):
        # actions: dict[int, action] keyed by player index
        # Returns: (obs, rewards, terminated, truncated, infos) — all dicts keyed by player
        ...
```

**Action masking** — return `"action_mask"` in per-player info dicts:

```python
def step(self, actions):
    ...
    infos = {
        0: {"action_mask": np.array([True, True, False, ...], dtype=bool)},
        1: {"action_mask": np.array([True, False, True, ...], dtype=bool)},
    }
    return obs, rewards, terminated, truncated, infos
```

### 2. Neural Networks

```python
# my_game/networks.py
import torch
import torch.nn as nn
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.distributions import CategoricalDist

class MyEncoder(BaseEncoder):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(192, 128), nn.ReLU())

    @property
    def latent_dim(self) -> int:
        return 128  # required: tells the framework the output dimension

    def forward(self, obs):
        return self.net(obs)

class MyPolicy(BasePolicy):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)

    def forward(self, latent):
        return CategoricalDist(logits=self.fc(latent))

class MyValue(BaseValue):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, latent):
        return self.fc(latent).squeeze(-1)
```

For continuous actions, use `DiagGaussianDist`. For composite actions (Dict/Tuple/MultiDiscrete), use `CompositeDist` — see below.

### 3. Config

```yaml
# my_game/config.yaml
env:
  env_class: "my_game.env.MyGameEnv"
  num_players: 2

networks:
  encoder_class: "my_game.networks.MyEncoder"
  policy_class: "my_game.networks.MyPolicy"
  value_class: "my_game.networks.MyValue"

algorithm:
  name: "appo"
  learning_rate: 3.0e-4

rollout:
  chunk_length: 128
  num_workers: 4
  envs_per_worker: 16

training:
  phase: "self_play"
  total_timesteps: 10_000_000
  seed: 42

self_play:
  checkpoint_interval: 1000
  pool_size: 20
  latest_prob: 0.5

metrics:
  use_wandb: true
  wandb_project: "my-competition"
```

### 4. Train

```bash
colosseum train --config my_game/config.yaml
```

---

## Training Phases

### Phase 1: Behavioral Cloning (optional)

Pre-train from expert data to bootstrap the policy:

```bash
colosseum bc -c config.yaml --data expert_games/ --output bc_weights.pt --epochs 50
```

Each `.pt` file is a dict with `observations` `[N, *obs_shape]`, `actions` (`[N]` int for Discrete,
`[N, D]` float for Box, the `ActionSpec` flat layout for Dict/Tuple/MultiDiscrete), and optional
`action_masks` `[N, mask_size]` bool and `dones` `[N]` bool. The loss is `-log pi(a|s)` with the
masks applied; an expert action that is illegal under its mask is an error. Stateful models
(LSTM/GRU/attention cores) train on windows of `bc.seq_len` transitions (`--seq-len`), resetting
the state at `dones`.

Use **kickstarting** during RL to regularize towards a teacher via decaying KL loss:

```yaml
training:
  phase: "self_play"
  resume_from: "bc_weights"   # start from BC checkpoint
```

### Phase 2: Self-Play

Agent plays against a pool of its own historical checkpoints.

```yaml
training:
  phase: "self_play"
self_play:
  checkpoint_interval: 1000    # save every N train steps
  pool_size: 20                # FIFO: keep last 20 checkpoints
  latest_prob: 0.5             # 50% chance opponent is current policy
```

### Phase 3: PFSP / League

Multiple trainable agents compete. PFSP focuses training on hard opponents.

```yaml
training:
  phase: "league"
self_play:
  self_play_ratio: 0.5        # 50% solo (own checkpoints), 50% arena (vs other agents)
  pfsp_exponent: 1.0          # higher = more focus on hard opponents

agents:
  agent_alpha:
    networks: null             # null = inherit global config
    algorithm: null
    learner: null
  agent_beta:
    algorithm:
      learning_rate: 1.0e-4   # agent_beta uses different LR
```

Each agent in `agents:` gets its own learner process. Workers are shared — chunks are routed to the correct learner by `agent_id`.

ELO and pairwise win rates tracked automatically by the coordinator.

---

## Multi-Agent League Training

Define multiple agents with optional per-agent overrides:

```yaml
training:
  phase: "league"

agents:
  aggressive_agent:
    algorithm:
      entropy_coeff: 0.02      # more exploration
      learning_rate: 5.0e-4
  defensive_agent:
    algorithm:
      entropy_coeff: 0.005     # less exploration
      learning_rate: 1.0e-4
  baseline_agent:
    networks: null              # uses global network config
    algorithm: null             # uses global algorithm config
```

Per-agent fields that can be overridden: `networks`, `algorithm`, `learner`. Unset (`null`) fields inherit from the global config. When `agents:` is empty or absent, a single agent `agent_0` uses the global config.

---

## RNN/LSTM for Partial Observability

For games with fog-of-war or memory requirements, add a recurrent trunk:

```yaml
networks:
  encoder_class: "my_game.networks.MyEncoder"
  policy_class: "my_game.networks.MyPolicy"
  value_class: "my_game.networks.MyValue"
  recurrent_type: "lstm"        # "lstm", "gru", or null (feedforward)
  recurrent_hidden_size: 128
  recurrent_num_layers: 1
```

When `recurrent_type` is set, an LSTM/GRU module is inserted between the encoder and policy/value heads. The policy and value head input dimension must match `recurrent_hidden_size` (not `encoder.latent_dim`).

The worker automatically:
- Maintains hidden state per (env, player) pair
- Passes hidden through inference calls
- Resets hidden state on episode boundaries
- Stores initial hidden at each chunk start for the learner

APPO automatically detects recurrent chunks and uses sequence-level training (`evaluate_actions_recurrent`) instead of the stateless path.

---

## Action Masking

Environments return valid action masks via info dicts. The framework automatically:

1. Passes masks to `CategoricalDist` which sets invalid action logits to `-inf`
2. Threads masks through `act()` (inference) and `evaluate_actions()` (training)
3. Stores masks in `TrajectoryChunk` for correct off-policy evaluation

Enable by returning `"action_mask"` in your env's `step()` and `reset()` info dicts — no config changes needed.

---

## Composite Action Spaces (Dict / Tuple / MultiDiscrete)

For competitions where agents output structured actions like `{"action_type": int, "accel": float, "target": int}`, use `gymnasium.spaces.Dict` (or `Tuple` / `MultiDiscrete`):

```python
# Environment
class MyEnv(BaseEnv):
    @property
    def action_space(self):
        return gymnasium.spaces.Dict({
            "action_type": gymnasium.spaces.Discrete(5),
            "target_pos": gymnasium.spaces.Box(low=-1, high=1, shape=(2,)),
        })

    def step(self, actions):
        # actions[player_idx] is a dict: {"action_type": 3, "target_pos": array([0.5, -0.2])}
        ...
```

```python
# Policy — returns CompositeDist with multiple heads
from colosseum.networks.distributions import CompositeDist, CategoricalDist, DiagGaussianDist

class MyPolicy(BasePolicy):
    def __init__(self):
        super().__init__()
        self.type_head = nn.Linear(128, 5)
        self.pos_mean = nn.Linear(128, 2)
        self.pos_logstd = nn.Parameter(torch.zeros(2))

    def forward(self, latent):
        return CompositeDist({
            "action_type": CategoricalDist(self.type_head(latent)),
            "target_pos": DiagGaussianDist(
                self.pos_mean(latent),
                self.pos_logstd.expand(latent.shape[0], -1),
            ),
        })
```

The framework handles everything transparently:
- Internally, composite actions are flattened to a single `float32` tensor (zero overhead for the training pipeline)
- `ActionSpec` auto-derives the flat layout from the action space
- VectorEnv decodes flat tensors back to dicts before passing to `env.step()`
- APPO, V-trace, TrajectoryChunk all work unchanged — they see flat tensors
- Action masking works with composite spaces: return a flat mask or a dict of per-head masks in info

Supported spaces: `Discrete`, `Box`, `Dict`, `Tuple`, `MultiDiscrete`. Nested composite (Dict-in-Dict) is not supported.

See `examples/composite_action/` for a complete working example.

---

## Automatic Mixed Precision (AMP)

For faster GPU training:

```yaml
algorithm:
  use_amp: true
  amp_dtype: "float16"    # or "bfloat16"
```

Wraps the forward pass in `torch.autocast` and uses `GradScaler` for float16. Only active on CUDA devices.

---

## Evaluation

Inference-only matchups between any agents/checkpoints:

```bash
colosseum eval -c config.yaml \
  -a agent_a:checkpoints/agent_0/ckpt_v100/model.pt \
  -a agent_b:checkpoints/agent_0/ckpt_v200/model.pt \
  -a agent_c:checkpoints/agent_1/ckpt_v50/model.pt \
  --num-matches 1000 \
  --num-envs 16
```

Supports N agents in N-player games. Agents are assigned to player slots round-robin. Pairwise results with Wilson score 95% confidence intervals.

Output: win rate matrix, per-pair statistics, average episode length.

---

## Distributed Training (gRPC)

### Multi-Machine Setup

```bash
# Machine 1: Weight Store
colosseum serve-weight-store --port 50051

# Machine 2: Trajectory Receiver (learner side)
colosseum serve-trajectory --port 50052

# Machine 3+: Workers
colosseum train --config config.yaml --set transport.mode=grpc
```

```yaml
transport:
  mode: "grpc"                 # "local" (mp.Queue) or "grpc"
  grpc_port: 50051
  grpc_max_message_mb: 64
```

### Docker Compose

```bash
cd deployment && docker compose up
```

### Kubernetes

```bash
kubectl apply -f deployment/k8s/
```

Worker pods auto-scale via HPA based on CPU utilization. See `deployment/k8s/` for manifests.

---

## Full Configuration Reference

### `algorithm`

| Key | Default | Description |
|-----|---------|-------------|
| `name` | `"appo"` | Algorithm name (cosmetic) |
| `algorithm_class` | `"colosseum.algorithms.appo.APPO"` | Dotted import path to algorithm class |
| `gamma` | `0.99` | Discount factor |
| `vtrace_lambda` | `1.0` | V-trace λ: trace coefficients c_t = λ·min(c̄, ρ_t); 1.0 = plain V-trace (GAE is not used) |
| `eps_clip` | `0.2` | PPO clipping epsilon |
| `value_loss_coeff` | `0.5` | Value-function loss coefficient |
| `entropy_coeff` | `0.01` | Entropy bonus coefficient |
| `max_grad_norm` | `0.5` | Max gradient norm for clipping |
| `num_epochs` | `1` | PPO epochs per batch |
| `minibatch_chunks` | `0` | Minibatch size in trajectory **chunks** (over batch dim B, not timesteps; sequences stay intact). 0 = all chunks as one batch |
| `vtrace_rho_bar` | `1.0` | V-trace truncation for importance weights |
| `vtrace_c_bar` | `1.0` | V-trace truncation for trace-cutting |
| `learning_rate` | `3e-4` | Initial learning rate |
| `lr_schedule` | `"linear"` | `"constant"`, `"linear"`, or `"cosine"` |
| `normalize_advantages` | `true` | Normalize advantages per minibatch |
| `use_torch_compile` | `false` | Compile V-trace with torch.compile |
| `use_amp` | `false` | Enable automatic mixed precision |
| `amp_dtype` | `"float16"` | AMP dtype: `"float16"` or `"bfloat16"` |

### `env`

| Key | Default | Description |
|-----|---------|-------------|
| `env_class` | **required** | Dotted import path to BaseEnv subclass |
| `num_players` | `2` | Number of player slots per match |
| `kwargs` | `{}` | Extra kwargs forwarded to env constructor |

### `networks`

| Key | Default | Description |
|-----|---------|-------------|
| `encoder_class` | **required** | Dotted path to BaseEncoder subclass |
| `policy_class` | **required** | Dotted path to BasePolicy subclass |
| `value_class` | **required** | Dotted path to BaseValue subclass |
| `kwargs` | `{}` | Extra kwargs forwarded to network constructors |
| `recurrent_type` | `null` | `"lstm"`, `"gru"`, or `null` (feedforward) |
| `recurrent_hidden_size` | `128` | Hidden size for recurrent trunk |
| `recurrent_num_layers` | `1` | Number of recurrent layers |

### `rollout`

| Key | Default | Description |
|-----|---------|-------------|
| `chunk_length` | `256` | Timesteps per trajectory chunk (T) |
| `num_workers` | `4` | Number of worker processes |
| `envs_per_worker` | `8` | Vectorized envs per worker |
| `weight_sync_interval_sec` | `5.0` | How often workers pull fresh weights (seconds) |

### `learner`

| Key | Default | Description |
|-----|---------|-------------|
| `device` | `"auto"` | `"auto"`, `"cuda:0"`, `"cpu"` |
| `queue_size` | `64` | Max trajectory chunks buffered |
| `batch_chunks` | `16` | Chunks aggregated into one training batch |
| `weight_push_interval` | `5` | Push weights to workers every N training steps (each push clones the state_dict; workers pull every `weight_sync_interval_sec`) |
| `pin_memory` | `false` | Pin batch tensors for faster CPU-to-GPU transfer |

### `training`

| Key | Default | Description |
|-----|---------|-------------|
| `phase` | `"self_play"` | `"bc"`, `"self_play"`, or `"league"` |
| `total_timesteps` | `10_000_000` | Total env steps before training ends |
| `seed` | `null` | Global random seed (null = non-deterministic) |
| `resume_from` | `null` | Checkpoint ID to resume from |

### `self_play`

| Key | Default | Description |
|-----|---------|-------------|
| `checkpoint_interval` | `1000` | Save checkpoint every N **training** steps (optimizer updates), not env steps. One train step = `chunk_length * batch_chunks` env steps |
| `pool_size` | `20` | Max checkpoints in FIFO pool per agent |
| `latest_prob` | `0.5` | Probability of latest policy as opponent |
| `self_play_ratio` | `0.5` | Fraction of solo matches (rest are arena) |
| `pfsp_exponent` | `1.0` | PFSP priority exponent: `(1 - win_rate)^p` |

### `checkpoint`

| Key | Default | Description |
|-----|---------|-------------|
| `dir` | `"checkpoints"` | Checkpoint directory |
| `save_optimizer` | `true` | Include optimizer state in checkpoints |

### `metrics`

| Key | Default | Description |
|-----|---------|-------------|
| `use_wandb` | `false` | Enable Weights & Biases logging |
| `wandb_project` | `"colosseum"` | WandB project name |
| `wandb_entity` | `null` | WandB entity (team or user) |
| `log_interval` | `10` | Log metrics every N training steps |

### `transport`

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `"local"` | `"local"` (mp.Queue) or `"grpc"` |
| `grpc_port` | `50051` | gRPC service port |
| `grpc_max_message_mb` | `64` | Max gRPC message size in MiB |

### `agents`

Per-agent config overrides. Keys are agent IDs. Empty = single `agent_0` using global config.

```yaml
agents:
  my_agent:
    networks: null              # null = inherit global
    algorithm:                  # override specific fields
      learning_rate: 1.0e-4
    learner:
      device: "cuda:1"
```

---

## Architecture

```
                         ┌────────────────┐
                         │  Coordinator   │
                         │  (matchmaking, │
                         │   checkpoints, │
                         │   ELO/PFSP)    │
                         └───────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
     ┌────────▼────────┐  ┌─────▼──────┐  ┌────────▼────────┐
     │  Worker 0       │  │  Worker 1  │  │  Worker N       │
     │  VectorEnv(K)   │  │  ...       │  │  VectorEnv(K)   │
     │  Multi-Agent    │  │            │  │  Multi-Agent    │
     │  Inference      │  │            │  │  Inference      │
     └────────┬────────┘  └─────┬──────┘  └────────┬────────┘
              │                  │                   │
              │  TrajectoryChunks (routed by agent_id)
              │  (mp.Queue / gRPC)
              │                  │                   │
     ┌────────▼────────┐  ┌─────▼──────┐  ┌────────▼────────┐
     │  Learner A      │  │  Learner B │  │  Learner C      │
     │  (agent_alpha)  │  │  (agent_β) │  │  (agent_γ)      │
     │  APPO + V-trace │  │  APPO      │  │  APPO           │
     │  GPU/CPU        │  │  GPU/CPU   │  │  GPU/CPU        │
     └────────┬────────┘  └─────┬──────┘  └────────┬────────┘
              │                  │                   │
              └──────── Weight Updates ─────────────┘
                        (mp.Queue / gRPC)
                                 │
                         ┌───────▼────────┐
                         │  Weight Store  │
                         │  (per-agent    │
                         │   weights)     │
                         └────────────────┘
```

- **Workers**: run environments + batched inference (grouped by agent_id + network_id), collect trajectory chunks, route to correct learner. Support action masking, LSTM hidden state tracking.
- **Learners**: one per trainable agent. Receive chunks, train with APPO (V-trace + PPO clip), push weights. Support recurrent training path, AMP.
- **Coordinator**: agent pool, checkpoint FIFO, matchmaking (self-play/PFSP), ELO/win-rate tracking, match result processing.
- **Weight Store**: latest model weights per agent (shared memory or gRPC service).

Workers and learners run fully async with no synchronization barriers. V-trace corrects for policy lag.

---

## Tests

```bash
# Run all tests (148 pass, 2 skip)
python -m pytest tests/ -v

# By category
python -m pytest tests/test_vtrace.py             # V-trace math
python -m pytest tests/test_appo.py               # APPO loss computation
python -m pytest tests/test_action_masking.py     # Action masking
python -m pytest tests/test_recurrent.py          # LSTM/GRU support
python -m pytest tests/test_multi_agent.py        # Multi-agent league
python -m pytest tests/test_composite_actions.py  # Dict/Tuple/MultiDiscrete actions
python -m pytest tests/test_bc.py                 # Behavioral Cloning
python -m pytest tests/test_ratings.py            # ELO, PFSP matchmaking
python -m pytest tests/test_eval.py               # Evaluation module
python -m pytest tests/test_grpc.py               # gRPC transport
python -m pytest tests/test_config.py             # Config validation
python -m pytest tests/test_distributions.py      # Distributions
python -m pytest tests/test_vec_env.py            # VectorEnv
python -m pytest tests/test_performance.py        # Performance optimizations
python -m pytest tests/test_integration.py        # End-to-end integration

# Standalone integration tests (use multiprocessing.spawn)
python tests/run_pipeline_test.py               # Full pipeline, 1 worker
python tests/run_scaled_test.py                 # 4 workers, scaled
```

---

## Project Structure

```
src/colosseum/
    core/           types.py, config.py, registry.py, action_spec.py
    envs/           base_env.py, vec_env.py
    networks/       base.py, distributions.py, actor_critic.py
    algorithms/     base.py, vtrace.py, appo.py
    worker/         rollout_worker.py
    learner/        learner.py
    coordinator/    coordinator.py, matchmaker.py, agent_pool.py,
                    checkpoint_manager.py, ratings.py
    weight_store/   base.py, shared_memory.py, grpc_store.py
    transport/      base.py, local.py, grpc_transport.py, serialization.py
    bc/             offline_bc.py, kickstart.py
    metrics/        wandb_logger.py
    eval.py
    launcher.py
    cli.py

examples/
    tic_tac_toe/            env.py, networks.py (Discrete actions)
    composite_action/       env.py, networks.py (Dict actions + CompositeDist)
configs/examples/           tic_tac_toe.yaml, tic_tac_toe_multi.yaml, chase.yaml
proto/                      colosseum.proto
deployment/                 Dockerfiles, docker-compose.yaml, k8s/
tests/                      148 tests across 16 test files + helpers.py
```

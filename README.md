# Colosseum

A reusable training framework for competitive bot-programming competitions (Lux AI, Neural MMO, CodeCraft, ...):
behavioural cloning → RL → self-play → league, built on an IMPALA-style asynchronous actor–learner (APPO with V-trace)
in PyTorch, without Ray.

> **Status (SP1 of 6).** Single-machine training is the supported mode: correct, observable and robust for
> 1v1 games with simultaneous or turn-based moves, solo games, multi-agent self-play and a league where all agents meet.
> Read [Status and limitations](#status-and-limitations) before relying on anything else.

## Quick start

```bash
git clone <repo-url> Colosseum && cd Colosseum
scripts/setup-dev.sh            # uv + .venv (Python 3.12) + CPU torch + colosseum[grpc,dev,examples]
source .venv/bin/activate

colosseum validate -c configs/examples/tic_tac_toe.yaml
colosseum train -c configs/examples/tic_tac_toe.yaml --set run.name=ttt-quickstart
```

`scripts/setup-dev.sh --gpu` installs the CUDA build of torch instead of the CPU one.

Training prints its run directory and one progress line per agent every 10 seconds (log lines go to stderr; the
same lines are in `runs/ttt-quickstart/logs/main.log`):

```
Run directory: runs/ttt-quickstart
... [INFO] main colosseum.progress: [agent_0] step 991 |  42.2% budget | 14,372 env-steps/s | loss 0.2190 | entropy 1.251 | return 0.000
...
... [INFO] main colosseum.progress: [agent_0] step 2298 | 100.0% budget | 10,106 env-steps/s | loss 0.0235 | entropy 0.327 | return 0.090 | wr_vs_past 0.95
... [INFO] main colosseum.launcher: Training finished; outputs in runs/ttt-quickstart
```

On an 8-core CPU the run (2 workers × 16 envs, 600,000 env steps) takes about a minute (52–73 s measured). Afterwards the latest
checkpoint, played greedily with the action mask, wins 86–93% of games against a random legal-move player (four
measured runs of 400 games each; `tests/learning/test_ttt_slow.py` requires >= 80%). Re-running with the same
`run.name` is refused; pick another name or delete `runs/ttt-quickstart`.

Evaluate the newest checkpoint against the oldest one still kept:

```bash
NEW=$(ls -d runs/ttt-quickstart/checkpoints/agent_0/ckpt_v* | sort -V | tail -1)
OLD=$(ls -d runs/ttt-quickstart/checkpoints/agent_0/ckpt_v* | sort -V | head -1)
colosseum eval -c configs/examples/tic_tac_toe.yaml -a new=$NEW -a old=$OLD --num-matches 200 --output eval.json
```

Continue training from that run (policy versions, optimizer state and the env-step counter continue):

```bash
colosseum train -c configs/examples/tic_tac_toe.yaml --set run.name=ttt-continued \
  --set training.resume_from=runs/ttt-quickstart --set training.total_timesteps=1200000
```

A two-agent league:

```bash
colosseum train -c configs/examples/tic_tac_toe_multi.yaml --set run.name=ttt-league
```

`runs/` and `eval.json` are git-ignored.

## What a run writes

```
runs/<name>/
  config.resolved.yaml      the config after --set overrides and agent merging
  logs/main.log             main process (also printed to the console)
  logs/learner-<agent>.log  one file per learner
  logs/worker-<i>.log       one file per rollout worker (worker-<i>-env<k>.log for subprocess envs)
  metrics.jsonl             one JSON record per line (below)
  ratings.json              latest ELO / win-rate matrix / wr_vs_past, rewritten periodically and at the end
  checkpoints/<agent>/ckpt_v<N>/{model.pt, trainer_state.pt, meta.json}
```

The run directory is `<run.dir>/<run.name>` (`run.dir` defaults to `runs`, relative to the current directory). Without
`run.name` it is `<config stem>-<YYYYmmdd-HHMMSS>`. Checkpoints always live inside the run directory.

`metrics.jsonl` records (`"kind"` field):

| kind | content |
|---|---|
| `train` | APPO metrics of one agent (`agent`, `train_step`, losses, entropy, `approx_kl`, `explained_variance`, `grad_norm`, `rho_mean`, `rho_clip_frac`, `policy_lag_mean/max`, `lr`, ...) every `metrics.log_interval` train steps |
| `episodes` | per agent since the previous record: `episodes`, `return_mean`, `length_mean`, `wdl` = W/D/L against `latest` (self), `past` (own checkpoints) and `arena` (other agents), `seat_counts` |
| `ratings` | `elo`, `win_rates`, `games`, `wr_vs_past`, `past_games` |
| `system` | `env_steps`, `env_steps_per_sec`, `train_steps_per_sec` per agent, learner `queue_depths`, `parked_buffers`, `workers_reporting` |

`meta.json` of a checkpoint holds `agent_id`, `checkpoint_id`, `policy_version`, `timestamp`, `config_hash`,
`env_steps`, `final` and the agent's `networks` section, so `colosseum eval` can rebuild any checkpoint's architecture.

### WandB

WandB is optional and only a viewer; `metrics.jsonl` stays the source of truth. Install the extra with
`uv pip install --python .venv/bin/python -e ".[wandb]"` and set `metrics.use_wandb: true`
(or `--set metrics.use_wandb=true`). A training run is one WandB run:
- per-agent training metrics are `<agent>/<metric>` on the axis `<agent>/train_step`;
- `ratings/*`, `system/*` and `episodes/*` (nested keys such as `ratings/elo/<agent>`) are on the axis `env_steps`.

Without an interactive `wandb login`, set `WANDB_API_KEY`, or `WANDB_MODE=offline` to log locally (upload later with
`wandb sync`). A WandB failure logs one warning and disables WandB; it never stops training. The distributed roles
(`run-learner`, `run-workers`) have no WandB and no `metrics.jsonl` until SP5.

## Process lifecycle

- `training.total_timesteps` is the global number of env steps over all workers. When it is reached, every learner
  sends a final checkpoint, the main process saves it, all processes stop, and the exit code is 0.
- If any worker or learner dies, training stops with exit code 1 and a message like
  `worker-0 died (exit -9), see runs/<name>/logs/worker-0.log`.
- Ctrl-C (SIGINT) and SIGTERM stop the run: the learners send final checkpoints, which are saved within a shutdown
  grace of 7 seconds; then any remaining child is terminated, so no process is left after 10 seconds. The exit codes
  are 130 (SIGINT) and 143 (SIGTERM). Child processes ignore Ctrl-C themselves (the main process stops them) and die
  with the main process if it is killed.
- An invalid config exits with code 1 and a `Config error: ...` message without a traceback.
- `colosseum eval` uses the same codes (0 / 1 / 130 / 143), plus 2 for bad command-line arguments.

## Writing your own game

### 1. Environment

```python
# my_game/env.py
import gymnasium, numpy as np
from colosseum.envs.base_env import BaseEnv

class MyGameEnv(BaseEnv):
    @property
    def num_players(self) -> int: return 2
    @property
    def observation_space(self): return gymnasium.spaces.Box(0, 1, (16,), np.float32)
    @property
    def action_space(self): return gymnasium.spaces.Discrete(4)   # Dict / Tuple / MultiDiscrete also work

    def reset(self, seed=None):
        obs = {p: np.zeros(16, np.float32) for p in range(2)}
        infos = {p: {"active": p == 0, "action_mask": np.ones(4, bool)} for p in range(2)}
        return obs, infos

    def step(self, actions):            # actions: {player: action}
        ...
        return obs, rewards, terminated, truncated, infos   # every value is a dict keyed by player
```

Per-player `info` keys the framework understands:
- `active` (bool): whose move it is. Only active players run inference and record transitions.
  Rewards that arrive while a player waits are credited to its last move. Without `active`, every player acts every step.
- `action_mask` (bool array): legal actions, applied in training, evaluation and BC. An active player whose mask has
  no legal action is an error.
- `outcome` (float in [0, 1]) or `rank` (int, 1 = best) in the terminal info: the authoritative match result used by
  ratings and eval. Without them, the outcome is derived from episode rewards.

`terminated` ends an episode; `truncated` (time limit) bootstraps with the value of the final observation.

### 2. Model

The default model is `encoder → core → policy head / value head`:

```python
# my_game/networks.py
import torch.nn as nn
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.distributions import CategoricalDist

class Encoder(BaseEncoder):
    def __init__(self, **kwargs):
        super().__init__(); self.net = nn.Sequential(nn.Linear(16, 64), nn.ReLU())
    @property
    def latent_dim(self): return 64
    def forward(self, obs): return self.net(obs)

class Policy(BasePolicy):
    def __init__(self, in_dim: int = 64, **kwargs):     # in_dim = the core's output size
        super().__init__(); self.net = nn.Linear(in_dim, 4)
    def forward(self, features): return CategoricalDist(logits=self.net(features))

class Value(BaseValue):
    def __init__(self, in_dim: int = 64, **kwargs):
        super().__init__(); self.net = nn.Linear(in_dim, 1)
    def forward(self, features): return self.net(features).squeeze(-1)
```

Cores (`colosseum.networks.cores`):
- `null`: no memory;
- `LSTMCore` / `GRUCore` (`hidden_size`, `num_layers`);
- `WindowAttentionCore` (`d_model`, `window`, `num_heads`, `num_layers`): causal attention over the last `window`
  latents of the episode (example: `configs/examples/tic_tac_toe_attention.yaml`).

For anything else, subclass `colosseum.networks.model.PolicyModel` (`initial_state`, `step`, optionally `unroll`) and
set `networks.model_class`. Composite actions (`Dict`, `Tuple`, `MultiDiscrete`) use
`colosseum.networks.distributions.CompositeDist`; see `examples/composite_action` and `configs/examples/chase.yaml`.

### 3. Config

```yaml
# my_game/config.yaml
run: {name: null, dir: runs}            # outputs go to runs/<name>/ (default name: <config stem>-<timestamp>)
env: {env_class: my_game.env.MyGameEnv, num_players: 2}
networks:
  encoder_class: my_game.networks.Encoder
  core: {class: colosseum.networks.cores.LSTMCore, kwargs: {hidden_size: 128}}   # or null
  policy_class: my_game.networks.Policy
  value_class: my_game.networks.Value
training: {phase: self_play, total_timesteps: 2000000}
```

```bash
colosseum validate -c my_game/config.yaml     # schema, env num_players, a dummy step/unroll of the model
colosseum train -c my_game/config.yaml
```

The current directory is put on `sys.path` (also for the spawned worker and learner processes), so run the commands
from the directory that contains `my_game/`.

## Configuration reference

Unknown keys are errors at every level. `--set key=value` accepts YAML values (`null`, numbers such as `1e-4`, lists,
`{}`; quote a value to keep it a string, e.g. `--set run.name='"123"'`) and works for `train`, `validate`,
`run-learner` and `run-workers`:

```bash
colosseum train -c configs/examples/tic_tac_toe.yaml --set run.name=lr-test \
  --set algorithm.learning_rate=1e-3 --set rollout.num_workers=4 --set training.resume_from=null
```

Inside a list, YAML 1.1 rules apply: `--set x=[1e-4,2]` keeps `1e-4` as a string; write `1.0e-4` there.

| section | keys (defaults) |
|---|---|
| `run` | `name` (null → `<config stem>-<YYYYmmdd-HHMMSS>`; an existing explicit name is an error), `dir` (`runs`) |
| `env` | `env_class`, `num_players` (2; must equal the env's), `kwargs` |
| `networks` | `model_class` (null) or `encoder_class` + `core` (`{class, kwargs}` or null) + `policy_class` + `value_class`; `kwargs` (passed to every constructor) |
| `algorithm` | `algorithm_class` (APPO), `gamma` 0.99, `vtrace_lambda` 1.0, `vtrace_rho_bar` 1.0, `vtrace_c_bar` 1.0, `eps_clip` 0.2, `value_loss_coeff` 0.5, `entropy_coeff` 0.01, `max_grad_norm` 0.5, `num_epochs` 1, `minibatch_chunks` 0, `learning_rate` 3e-4, `lr_schedule` (`linear`; also `constant`, `cosine`; follows the share of `total_timesteps` done), `normalize_advantages` (true), `use_amp` (false), `amp_dtype` (`float16` / `bfloat16`), `use_torch_compile` (false). AMP is CUDA-only and has not been run on a GPU yet (see Known limitations) |
| `rollout` | `chunk_length` 256, `num_workers` 4, `envs_per_worker` 8, `torch_threads` 1, `weight_sync_interval_sec` 5, `vec_env` (`sync`/`subprocess`), `subproc_workers`, `match_refresh_interval_sec` 30 |
| `learner` | `device` (`auto`), `batch_chunks` 16 (every update uses exactly this many chunks), `queue_size` 64, `weight_push_interval` 5, `torch_threads` (auto), `pin_memory` (false). A CUDA `device` and `pin_memory` have not been run on a GPU yet (see Known limitations) |
| `training` | `phase` (`self_play` / `league`), `total_timesteps` (global env steps), `seed`, `resume_from`, `kickstart_teacher`, `kickstart_lambda` 1.0, `kickstart_decay_steps` 50000, `kickstart_kl` (`forward` = KL(teacher‖student), or `reverse`) |
| `self_play` | `checkpoint_interval` (train steps), `pool_size` (FIFO per agent), `latest_prob`, `self_play_ratio` (league: share of self-play matches), `pfsp_exponent`, `shuffle_seats` (true) |
| `checkpoint` | `save_optimizer` (true: also save `trainer_state.pt`) |
| `metrics` | `log_interval` (train steps between `train` records), `console_interval_sec` 10, `use_wandb`, `wandb_project`, `wandb_entity` |
| `bc` | `seq_len` 64 (sequence length for stateful models in `colosseum bc`) |
| `transport` | `grpc_max_message_mb` 64 (distributed mode; `mode` and `grpc_port` are not used, ports are command-line flags) |
| `agents` | `{agent_id: {networks: {...}, algorithm: {...}, learner: {...}}}`: partial overrides, deep-merged onto the global sections |

**Agents.** Without an `agents` section there is one agent, `agent_0`. With it, every key is a trainable agent with its
own learner; an override changes only the keys it names:

```yaml
agents:
  alpha: {}
  beta:
    algorithm: {learning_rate: 1.0e-4}                       # every other algorithm key stays global
    networks: {core: {class: colosseum.networks.cores.GRUCore, kwargs: {hidden_size: 64}}}
```

An agent id names directories and metric namespaces: it may contain letters, digits, `_` and `-` (not starting with
`-`), no `.` (so `--set agents.<id>.algorithm.learning_rate=...` always works), and it cannot be `ratings`, `system`,
`episodes` or `train`.

**Matchmaking.** Every env has an owner: the trainable agents take turns by env index, rotating at every match refresh.
- `self_play`: the owner's latest weights in every seat, except that each non-owner seat plays a random own checkpoint
  (not collecting data) with probability `1 - latest_prob`.
- `league`: with probability `self_play_ratio` the match is a self-play match. Otherwise it is an arena match: the owner
  plus N−1 opponents drawn by PFSP, `(1 - win_rate)^pfsp_exponent`, from the other agents, with replacement; every
  arena seat collects data.

**Ratings.** Seats are shuffled. Every pair of seats of different agents scores 1 / 0.5 / 0 by outcome. The win-rate
matrix and ELO use those pairs; all pairs of one match are applied at once with K scaled by `k_scale = 1/(N−1)`. The
scale is per seat, so an agent that fills several seats of one match (an arena with fewer agents than seats) gets
proportionally more ELO exposure from that match. `wr_vs_past` is the latest weights' score against the agent's own
checkpoints over the last 500 pairs.

**Resume.** `training.resume_from` accepts:
- a checkpoint dir (weights + optimizer + versions);
- a previous run dir (each agent takes its latest checkpoint there; an agent without checkpoints starts fresh with a
  warning);
- a `.pt` state dict (weights only, e.g. the output of `colosseum bc`). It is applied to every agent and must match
  each agent's architecture.

An explicit resume is strict: a checkpoint with a missing or malformed `meta.json` in the selected directory is a
config error naming the path, never silently skipped.

## Behavioural cloning and kickstarting

```bash
colosseum bc -c my_game/config.yaml --data path/to/demos/ --output bc.pt --epochs 20
colosseum train -c my_game/config.yaml --set training.resume_from=bc.pt
```

BC data: `.pt` files with `observations`, `actions` and optional `action_masks` and `dones`. The loss is `-log_prob` of
the recorded action for every distribution type. Masks are applied. Stateful models train on sequences of `--seq-len`
steps (default `bc.seq_len`, 64) that reset at `dones`.

Kickstarting adds `lambda * KL(teacher‖student)` to the RL loss, decaying linearly over `kickstart_decay_steps`:
`--set training.kickstart_teacher=bc.pt`. The teacher uses the student's architecture.

## Evaluation

`colosseum eval` plays inference-only matches between checkpoints (`-a name=path`, repeatable; a path is a checkpoint
dir or a `.pt` state dict). It reuses the training semantics: per-seat model state, `active` and action masks.
- Two or more agents: every pair plays `--num-matches` matches (rounded up to an even number) with rotated seats, so
  each agent plays every seat equally often. The report gives W/D/L, the win rate and the score (a draw counts as half)
  with 95% Wilson intervals, a per-seat breakdown, mean returns and episode length.
- One agent, or a 1-player env (solo mode): `--num-matches` episodes per agent, mean return and outcome with 95%
  intervals.
- `--output result.json` writes the machine-readable report; `--deterministic` plays greedily.

A checkpoint's architecture comes from its `meta.json`; a `.pt` file is built from `--config`.

## Distributed mode (limited)

The roles run as separate processes, possibly on separate machines:

```bash
colosseum serve-weight-store --port 50051                                              # machine A
colosseum run-learner -c cfg.yaml --agent agent_0 --traj-port 50052 --weight-store A:50051   # machine B
colosseum run-workers -c cfg.yaml --weight-store A:50051 -l agent_0=B:50052                  # machines C, D, ...
```

`run-learner` writes `runs/<name>-learner-<agent>/` (logs, checkpoints). `run-workers` writes
`runs/<name>-workers-<host>/` (logs), so several worker machines can share one `run.name` on a shared filesystem.

Limitations in SP1 (to be fixed in SP5):
- there is no coordinator: workers play the latest weights of every agent in a fixed round-robin, with no historical
  opponents, no league, no ratings and no `metrics.jsonl` or WandB;
- every `run-workers` host collects `total_timesteps / num_workers` per worker on its own, and the learner's progress is
  `consumed_samples / total_timesteps`;
- `run-learner` cannot resume (`training.resume_from` is not applied);
- no fault tolerance or authentication;
- `deployment/` (Docker, Kubernetes) is untested.

## Status and limitations

SP1 ("foundation and stabilization") is the first of six sub-projects that follow the full review in
[`review/README.md`](review/README.md):

| | Sub-project | Content |
|---|---|---|
| ✔ | SP1 Foundation and stabilization | correct single-machine training, stateful model protocol, run dir, metrics, lifecycle |
| | SP2 Game model | `GameSpec`/multi-agent env API, player elimination, teams, roles, per-unit actions, Dict observations |
| | SP3 Players, league, warm start | scripted, frozen and external players; PFSP over snapshots; per-agent `init`/kickstart/critic warm-up; top-k snapshot storage |
| | SP4 Selection and observability | match log, OpenSkill / Bradley–Terry ratings, `colosseum tournament`, dashboard, snapshot ratings |
| | SP5 Distributed | hub and nodes, wire format, per-machine weight cache, fault tolerance, `max_policy_lag`, K8s images |
| | SP6 Speed and extensions | cuDNN RNN path, fast transformer unroll, GPU inference on workers, inference server, new algorithms |

Known limitations today:
- **Game model:** symmetric players with identical observation and action spaces; a match ends for everyone at once
  (no elimination); no teams or roles.
- **League:** only trainable agents. PFSP picks among agents' latest weights, not snapshots. Checkpoints are a FIFO pool.
  Online ELO is a progress indicator, not a selection-grade rating.
- **Speed:** recurrent and attention cores unroll step by step on the learner. Worker inference is CPU only.
- **Distributed:** see above.
- **GPU:** never run on CUDA during SP1. All CUDA paths (learner device, AMP fp16/bf16, `pin_memory`,
  kickstart/BC on CUDA) are covered only by `gpu`-marked tests that have not been executed yet; see
  [`docs/GPU_CHECKS.md`](docs/GPU_CHECKS.md).

## Tests

```bash
.venv/bin/python -m pytest -m "not gpu and not slow" -q     # full fast suite (CI)
.venv/bin/python -m pytest -m slow -v                       # tic-tac-toe learning test + torch.compile tests (~1.5 min)
.venv/bin/python -m pytest -m gpu -v                        # CUDA machine only, see docs/GPU_CHECKS.md
```

Layout:
- `tests/unit`: pure functions and classes;
- `tests/contract`: the real worker loop feeding the real APPO, in-process;
- `tests/integration`: CLI and multi-process runs (`colosseum train`/`eval`/`bc` subprocesses, gRPC, distributed roles);
- `tests/learning`: does it learn (fast bandit and chain checks, BC; the slow tic-tac-toe run).

Tests write only under pytest's `tmp_path`. Throughput measurements are in [`docs/benchmarks.md`](docs/benchmarks.md).

## Project structure

```
src/colosseum/
  cli.py launcher.py distributed.py eval.py
  core/         config, types, registry, action_spec, outcomes, errors, ipc, run_dir, seat_info, threads
  networks/     model (PolicyModel, act), composed, cores, state, distributions, normalization, base
  algorithms/   appo, vtrace, base
  worker/       rollout_loop (RolloutLoop), slots, rollout_worker (process wrapper)
  learner/      learner process, checkpoint payloads
  coordinator/  coordinator, matchmaker, ratings, checkpoint_manager, agent_pool
  metrics/      jsonl, aggregator, console, hub, wandb_logger
  envs/         base_env, vec_env, subproc_vec_env
  bc/ transport/ weight_store/ utils/
examples/       tic_tac_toe, composite_action (chase), space_miners (Box2D, from the examples extra)
configs/examples/
scripts/        setup-dev.sh, bench_throughput.py
docs/           benchmarks.md, GPU_CHECKS.md
```

# Colosseum

A reusable training framework for competitive bot-programming competitions (Lux AI, Neural MMO, CodeCraft, ...):
behavioural cloning → RL → self-play → league, built on an IMPALA-style asynchronous actor–learner (APPO with V-trace)
in PyTorch, without Ray.

> **Status (SP2 of 6).** Single-machine training is the supported mode for every game structure: solo, 1v1 with
> turn-based or simultaneous moves, one bot controlling many units, team vs team, free-for-all with elimination and
> 2–4 players in one run, asymmetric roles, and cooperative games. Each structure has a demo game that is trained
> end-to-end in the test suite. Read [Status and limitations](#status-and-limitations) before relying on anything else.

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
same lines are in `runs/ttt-quickstart/logs/main.log`). On an 8-core CPU the run (2 workers × 16 envs, 600,000 env
steps) takes about a minute (59–64 s over three seeds). Afterwards the latest checkpoint, played greedily, wins
86–94% of games against a random legal-move player (`tests/learning/test_demo_learning_slow.py` requires >= 80%).
Re-running with the same `run.name` is refused; pick another name or delete `runs/ttt-quickstart`.

Evaluate the newest checkpoint against the oldest one still kept:

```bash
NEW=$(ls -d runs/ttt-quickstart/checkpoints/agent_0/ckpt_v* | sort -V | tail -1)
OLD=$(ls -d runs/ttt-quickstart/checkpoints/agent_0/ckpt_v* | sort -V | head -1)
colosseum eval -c configs/examples/tic_tac_toe.yaml -a new=$NEW -a old=$OLD --num-matches 200 --output eval.json
```

Continue training from that run (policy versions, optimizer state and the env-step counter continue; the new
`total_timesteps` must be above the env steps already done):

```bash
colosseum train -c configs/examples/tic_tac_toe.yaml --set run.name=ttt-continued \
  --set training.resume_from=runs/ttt-quickstart --set training.total_timesteps=1200000
```

Other game structures (each of these trains in under three minutes on 8 cores):

```bash
colosseum train -c configs/examples/tron.yaml --set run.name=tron                    # FFA, 2/3/4 players, elimination
colosseum train -c configs/examples/predator_prey.yaml --set run.name=predator-prey  # asymmetric roles, two agents
colosseum train -c configs/examples/coop_buttons.yaml --set run.name=coop            # cooperative, mixed teammates
colosseum train -c configs/examples/tic_tac_toe_multi.yaml --set run.name=ttt-league # two agents in a league
```

`runs/` and `eval.json` are git-ignored.

## Demo games

| Game | Config | Structure | Contract features |
|---|---|---|---|
| `coin_grid` | `coin_grid.yaml` | solo | Dict observation with a `uint8` grid, masks, step-limit truncation |
| `tic_tac_toe` | `tic_tac_toe.yaml` (also `_attention`, `_multi`) | 1v1 turn-based | one acting seat, masks, rewards to the waiting seat |
| `unit_harvest` | `unit_harvest.yaml` | one bot, many units, simultaneous | `Units`, units born and killed, entity lists with masks |
| `team_tag` | `team_tag.yaml` | 2v2 | local window per bot, `global_state` for a centralized critic, tagged (frozen-in-game) teammates keep team rewards |
| `tron` | `tron.yaml` | FFA 2p/3p/4p | elimination (`terminated`), ranks by elimination order, several layouts in one run |
| `predator_prey` | `predator_prey.yaml` | 1 vs 2 | roles with different spaces, one agent per role |
| `coop_buttons` | `coop_buttons.yaml` | cooperative | one team, `score` outcome, mixed teammates, cross-play table |
| `space_miners` | `space_miners.yaml` | 1v1 (Box2D) | units with a continuous and a discrete component, entity list; reference only |
| `composite_action` | `chase.yaml` | 1v1 | tree action `Dict(direction=Discrete, speed=Box)`; reference only |

The slow learning tests train each of the first seven and check it against random players: coin_grid scores at least
twice the random score; tic-tac-toe, unit_harvest and team_tag win at least 80% (team_tag: the better of two
independent runs, because about 2 of 9 single runs settle into a passive draw-seeking policy); tron wins 80% of 2p
games and takes first place in at least half of 4p games against three random cycles; each predator_prey role beats a
random opponent at least 70%; both coop_buttons agents' homogeneous teams reach `max(5 × random, 0.6 × scripted)`
of the measured baselines. Numbers per seed are in `docs/superpowers/reports/2026-10-09-sp2-acceptance.md`.

## Writing your own game

Read [`docs/ENV_GUIDE.md`](docs/ENV_GUIDE.md) (in Russian): how to write an env for every game type, with the demo
games as examples. In short:

```python
# my_game/env.py
import gymnasium, numpy as np
from colosseum.envs.game import GameSpec, MultiAgentEnv, Outcome, StepResult

class MyGame(MultiAgentEnv):
    def __init__(self):
        obs = gymnasium.spaces.Box(0, 1, (16,), np.float32)
        self.spec = GameSpec.symmetric(2, obs, gymnasium.spaces.Discrete(4))   # layout "2p"

    def reset(self, seed, layout):
        return StepResult(acting={0}, obs={0: np.zeros(16, np.float32)}, action_masks={0: np.ones(4, bool)})

    def step(self, actions):                       # actions: {seat: action} for exactly the acting seats
        ...
        return StepResult(acting={1}, obs={1: ...}, action_masks={1: ...}, rewards={0: 0.0, 1: 0.0})
        # at the end: StepResult(acting=set(), obs={}, rewards=..., episode_over=True,
        #                        outcome=Outcome(team_rank={0: 1.0, 1: 2.0}))
```

- `GameSpec.solo`, `GameSpec.symmetric(n or [2, 3, 4])`, `GameSpec.teams_of([2, 2])` or a hand-written
  `GameSpec(roles=..., layouts=...)` describe roles (observation, action and optional `global_state` spaces), seats
  and teams. The outcome kind follows the number of teams: one → `score`, two → `wdl`, three or more → `rank`.
- `StepResult.acting` says who moves next; `terminated` eliminates seats; `truncated=True` with `final_obs` marks an
  artificial cut (the learner bootstraps from `final_obs`); `outcome` gives team ranks or scores (without it, a team's
  score is the mean of its seats' returns).
- Observations and actions are trees (`Dict`); `uint8` stays `uint8` up to the model. Masks are `bool` arrays. Many
  units per bot: `colosseum.envs.spaces.Units(max_units, per_unit, only_if=None)`.
- A waiting seat gets no model step: whatever a policy must know about what happened between its own turns has to be
  in its next observation.
- Contract violations raise `EnvContractError` with "worker, env, seat, episode step, layout"; run
  `colosseum validate -c my_game/config.yaml` first.

The default model is `encoder → core → policy head / value head` (+ an optional critic encoder over `global_state`):

```python
# my_game/networks.py
import torch.nn as nn
from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue
from colosseum.networks.heads import make_distribution

class Encoder(BaseEncoder):
    def __init__(self, observation_space, **kwargs):
        super().__init__(); self.net = nn.Sequential(nn.Linear(observation_space.shape[0], 64), nn.ReLU())
    @property
    def latent_dim(self): return 64
    def forward(self, obs): return self.net(obs)

class Policy(BasePolicy):
    def __init__(self, in_dim, action_spec, **kwargs):      # in_dim = the core's output size
        super().__init__(); self.action_spec = action_spec; self.net = nn.Linear(in_dim, 4)
    def forward(self, features, aux): return make_distribution(self.action_spec, self.net(features))

class Value(BaseValue):
    def __init__(self, in_dim, **kwargs):
        super().__init__(); self.net = nn.Linear(in_dim, 1)
    def forward(self, features): return self.net(features).squeeze(-1)
```

Constructors receive by name what their signature declares: the encoder, the critic encoder and the policy get
`observation_space`, `action_space`, `global_state_space` and `action_spec`; the policy and value heads get `in_dim`
(the core's output size; for the value head plus the critic encoder's output size); all of them, and
`model_class`, also get `networks.kwargs`. The core gets only `input_dim` (the encoder's `latent_dim`) and its own
`core.kwargs`. The policy returns a `colosseum.networks.dist.Distribution` (use `make_distribution`); the
framework applies the action mask to it. Cores (`colosseum.networks.cores`): `null` (no memory), `LSTMCore` /
`GRUCore` (`hidden_size`, `num_layers`), `WindowAttentionCore` (`d_model`, `window`, `num_heads`, `num_layers`). For
anything else subclass `colosseum.networks.model.PolicyModel` (`step` and `unroll`) and set `networks.model_class`.

```yaml
# my_game/config.yaml
run: {name: null, dir: runs}
env: {env_class: my_game.env.MyGame, kwargs: {}}
networks:
  encoder_class: my_game.networks.Encoder
  core: null
  policy_class: my_game.networks.Policy
  value_class: my_game.networks.Value
training: {total_timesteps: 2000000}
```

```bash
colosseum validate -c my_game/config.yaml
colosseum train -c my_game/config.yaml
```

The current directory is put on `sys.path` (also for the spawned worker and learner processes), so run the commands
from the directory that contains `my_game/`.

## What a run writes

```
runs/<name>/
  config.resolved.yaml      the config after --set overrides
  logs/main.log             main process (also printed to the console)
  logs/learner-<agent>.log  one file per learner
  logs/worker-<i>.log       one file per rollout worker (worker-<i>-env<k>.log for subprocess envs)
  metrics.jsonl             one JSON record per line (below)
  ratings.json              {"env_steps", "layouts": {<layout>: ratings tables}}, rewritten periodically and at the end
  checkpoints/<agent>/ckpt_v<N>/{model.pt, trainer_state.pt, meta.json}
```

The run directory is `<run.dir>/<run.name>` (`run.dir` defaults to `runs`, relative to the current directory). Without
`run.name` it is `<config stem>-<YYYYmmdd-HHMMSS>`. Checkpoints always live inside the run directory.

`metrics.jsonl` records (`"kind"` field):

| kind | content |
|---|---|
| `train` | APPO metrics of one agent every `metrics.log_interval` train steps: `total_loss`, `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction` (per decider) and `clip_fraction_joint`, `rho_mean`, `rho_clip_frac`, `c_clip_frac`, `ess`, `log_rho_abs_mean/p95`, `log_rho_joint_abs_mean/p95`, `deciders_valid_mean/max`, `boot_frac`, `pad_frac`, `explained_variance`, `grad_norm`, `lr`, `policy_version`, `kickstart_loss` / `kickstart_lambda` (with a teacher) |
| `episodes` | per agent since the previous record: `episodes`, `return_mean`, `length_mean`, `wdl` (W/D/L against `latest`, `past` and `arena` opponents), `seat_counts`, and `by_layout.<layout>.<role>` with episodes, returns, lengths, `team_score_mean`, `eliminated_frac` and W/D/L by opponent type |
| `ratings` | `layouts`: per layout `outcome_kind`, `elo`, `win_rates`, `games`, `wr_vs_past`, `past_games`, `scores`, `cross_play`, `role_win_rates` |
| `system` | `env_steps`, `env_steps_per_sec`, `train_steps_per_sec` per agent, learner `queue_depths`, `parked_buffers`, `workers_reporting` |

`meta.json` of a checkpoint holds `agent_id`, `checkpoint_id`, `policy_version`, `env_steps`, `final`, `config_hash`,
the agent's `networks` section, its `roles` and the `role_signature` of their spaces, so `colosseum eval` can rebuild
any checkpoint and resume/eval refuse a checkpoint whose spaces do not match.

### WandB

WandB is optional and only a viewer; `metrics.jsonl` stays the source of truth. Install the extra with
`uv pip install --python .venv/bin/python -e ".[wandb]"` and set `metrics.use_wandb: true`. A training run is one
WandB run. Per-agent training metrics are `<agent>/<metric>` on the axis `<agent>/train_step`;
`ratings/<layout>/*`, `system/*` and `episodes/<agent>/*` are on the axis `env_steps`. Without an interactive
`wandb login`, set `WANDB_API_KEY`, or `WANDB_MODE=offline` to log locally (upload later with `wandb sync`). A WandB
failure logs one warning and disables WandB; it never stops training.

## Process lifecycle

- `training.total_timesteps` is the global number of env steps over all workers. When it is reached, every learner
  sends a final checkpoint, the main process saves it, all processes stop, and the exit code is 0. The run can
  overshoot the budget by about a second of env steps (workers report their counts periodically).
- If any worker or learner dies, training stops with exit code 1 and a message like
  `worker-0 died (exit -9), see runs/<name>/logs/worker-0.log`.
- Ctrl-C (SIGINT) and SIGTERM stop the run: final checkpoints are saved within a shutdown grace of 7 seconds; no
  process is left after 10 seconds. The exit codes are 130 (SIGINT) and 143 (SIGTERM). Child processes ignore Ctrl-C
  themselves (the main process stops them) and die with the main process if it is killed.
- A Ctrl-C always ends `train`, `eval`, `bc`, `validate`, `run-learner` and `run-workers` with exit code 130 and
  `Interrupted` (or `Received SIGINT`), never a traceback, also when it arrives during startup. Rarely, Python loses a
  Ctrl-C that lands inside an import-time callback: `train` and the distributed roles then still stop as on a normal
  Ctrl-C, while `eval`, `bc` and `validate` finish their work first and then exit 130 with `Interrupted`.
  (`serve-weight-store` stops with exit code 0 on Ctrl-C.)
- An invalid config (or an env that breaks the contract at startup) exits with code 1 and a `Config error: ...`
  message without a traceback.
- `colosseum eval` uses the same codes (0 / 1 / 130 / 143), plus 2 for bad command-line arguments; `colosseum bc`
  uses 0 / 1 / 2 / 130.

## Configuration reference

Unknown keys are errors at every level. `--set key=value` accepts YAML values (`null`, numbers such as `1e-4`, lists,
mappings; quote a value to keep it a string) and works for `train`, `validate`, `run-learner` and `run-workers`. A
mapping value replaces the whole mapping (e.g. `env.kwargs`):

```bash
colosseum train -c configs/examples/tron.yaml --set run.name=tron-4p --set 'matchmaking.layouts={4p: 1.0}'
```

Inside a list, YAML 1.1 rules apply: `--set x=[1e-4,2]` keeps `1e-4` as a string; write `1.0e-4` there.

| section | keys (defaults) |
|---|---|
| `run` | `name` (null → `<config stem>-<YYYYmmdd-HHMMSS>`; an existing explicit name is an error), `dir` (`runs`) |
| `env` | `env_class`, `kwargs`, `max_idle_steps` 1000 (more consecutive steps without an acting seat and without `episode_over` is an error) |
| `networks` | `model_class` (null) or `encoder_class` + `core` (`{class, kwargs}` or null) + `policy_class` + `value_class` + `critic_encoder_class` (null; needs a role with `global_state_space`); `kwargs` (passed to the encoder, critic encoder, heads and `model_class`; the core gets only `core.kwargs`) |
| `algorithm` | `name` (`appo`), `algorithm_class` (`colosseum.algorithms.appo.APPO`), `gamma` 0.99, `vtrace_lambda` 1.0, `vtrace_rho_bar` 1.0, `vtrace_c_bar` 1.0, `eps_clip` 0.2, `value_loss_coeff` 0.5, `entropy_coeff` 0.01, `max_grad_norm` 0.5, `num_epochs` 1, `minibatch_chunks` 0 (all chunks), `learning_rate` 3e-4, `lr_schedule` (`linear`; also `constant`, `cosine`), `normalize_advantages` (true), `use_amp` (false), `amp_dtype` (`float16` / `bfloat16`), `use_torch_compile` (false), `ratio_mode` (`auto` = `per_unit` with `Units`, else `joint`), `unit_trace` (`auto` = `joint` with or without `Units`, the units-experiment ruling in `docs/benchmarks.md`; also `geo_mean`, `none`), `entropy_reduction` (`auto` = `sum` for `joint`, `mean_valid` for `per_unit`); with one decider every mode is `joint` (an explicit `unit_trace: none` is kept) |
| `rollout` | `chunk_length` 256 (>= 2), `num_workers` 4, `envs_per_worker` 8, `torch_threads` 1, `weight_sync_interval_sec` 5, `vec_env` (`sync` / `subprocess`), `subproc_workers` (null), `match_refresh_interval_sec` 30 |
| `learner` | `device` (`auto`), `batch_chunks` 16 (every update uses exactly this many chunks), `queue_size` 64, `weight_push_interval` 5, `torch_threads` (null = auto), `pin_memory` (false) |
| `training` | `total_timesteps` 10,000,000 (global env steps), `seed`, `resume_from`, `kickstart_teacher`, `kickstart_lambda` 1.0, `kickstart_decay_steps` 50000, `kickstart_kl` (`forward` / `reverse`) |
| `matchmaking` | `mode` (`self_play` / `league`), `layouts` (`{layout: weight}`; empty = every layout equally), `self_play_ratio` 0.5, `pfsp_exponent` 1.0, `latest_prob` 0.5, `teammates` (`self` / `mixed`), `teammate_self_prob` 0.5, `shuffle_seats` (true) |
| `checkpoint` | `interval` 1000 (train steps), `pool_size` 20 (FIFO per agent), `save_optimizer` (true) |
| `metrics` | `log_interval` 10 (train steps), `console_interval_sec` 10, `use_wandb` (false), `wandb_project` (`colosseum`), `wandb_entity` |
| `bc` | `seq_len` 64 (sequence length for stateful models in `colosseum bc`) |
| `transport` | `grpc_max_message_mb` 64 (distributed mode); `mode` and `grpc_port` are unused until SP5 |
| `agents` | `{agent_id: {roles: [...], networks: {...}, algorithm: {...}, learner: {...}}}`: roles and partial overrides, deep-merged onto the global sections |

**Agents and roles.** Without an `agents` section there is one agent, `agent_0`, playing every role (then all roles
must have the same spaces). With it, every key is a trainable agent with its own learner. `roles` lists the roles an
agent plays (omitted = every role); the roles of one agent must have the same observation, action and `global_state`
spaces. An agent id may contain letters, digits, `_` and `-` (not starting with `-`), no `.`, and cannot be
`ratings`, `system`, `episodes` or `train`.

**Matchmaking.** Every env has an owner: the trainable agents take turns by env index. For each match the matchmaker
picks a layout (`matchmaking.layouts`, only layouts with a seat for the owner's roles), a match type once per match
(self-play with probability `self_play_ratio`, else arena; `mode: self_play` means always self-play), and the owner's
team. The owner's team gets the owner's latest weights as its core. Every other team gets a core: in self-play, if
the owner plays a role of that team, the owner's latest weights (probability `latest_prob`) or one of its
checkpoints; otherwise (arena, or roles the owner does not play) another trainable agent that plays a role of the
team, picked by PFSP, `(1 - win_rate)^pfsp_exponent`, on that layout. The core takes one seat; `teammates: self`
gives it the team's other seats it can play; `mixed` gives each such seat to the core with probability
`teammate_self_prob`, else to another trainable agent's latest weights or a checkpoint of the core. A seat of a role
the core does not play goes to the latest weights of an agent that plays it. All seats with latest weights collect
data; checkpoint seats do not. `shuffle_seats` permutes teams with the same role composition and same-role seats
within a team.

**Ratings** are kept per layout. Two or more teams: ELO over team pairs (each pair is one comparison by rank; its
weight `K / (T - 1)` is split among the counted member pairs of different agents), a fractional win-rate matrix and
`wr_vs_past` (the latest weights against the agent's own checkpoints, last 500 pairs). One team: mean score with EMA
and a 95% interval per agent, and with `teammates: mixed` a cross-play table "team composition → mean score".

**Resume.** `training.resume_from` accepts a checkpoint dir, a previous run dir (each agent takes its latest
checkpoint there) or a `.pt` state dict (weights only, e.g. from `colosseum bc`). An explicit resume is strict: a
missing or malformed `meta.json`, or a role signature that does not match the agent's roles, is a config error naming
the path.

## Behavioural cloning and kickstarting

```bash
colosseum bc -c my_game/config.yaml --agent agent_0 --data path/to/demos/ --output bc.pt --epochs 20
colosseum train -c my_game/config.yaml --set training.resume_from=bc.pt
```

Other `bc` options: `--batch-size` (256), `--lr` (1e-3), `--seq-len` (default `bc.seq_len`). BC data: `.pt` files
with the trees `observations`, `actions` and optional `action_masks` and `dones`. The network and roles come from the
`--agent` (default: the only agent). The loss is `-log_prob` of the recorded action (the mean over valid deciders for
actions with units); masks are applied, and an expert action the mask forbids is a data error; stateful models train
on sequences of `--seq-len` steps that reset at `dones`.

Kickstarting adds `lambda * KL(teacher‖student)` per decider to the RL loss, decaying linearly over
`kickstart_decay_steps`: `--set training.kickstart_teacher=bc.pt`. There is one global teacher: its spaces must match
every agent's roles.

## Evaluation

```bash
colosseum eval -c cfg.yaml -a A=<ckpt-dir|.pt> [-a B=...] [--layout L ...] --num-matches N [--num-envs E] [--output r.json] [--deterministic] [--seed S]
```

Matches run on the same engine as training (`MatchRunner`): per-seat model state, acting seats and masks. An agent's
architecture and roles come from its checkpoint's `meta.json`; a `.pt` is built from the config agent of the same
name, or from the global `networks` with every role. By default every layout the agents can fill is played.
- Two or more teams, two or more agents: for every pair (a, b) match m gives team i to `(a, b)[(i + m) % 2]` where
  the roles allow; teams are filled homogeneously; `--num-matches` is per pair and layout (an odd count is rounded
  up, so every agent plays every side equally often).
- Two or more teams, one agent: all teams are that agent.
- One team: each agent in a homogeneous team, plus mixed compositions for cross-play.

Reports per layout: `wdl` — W/D/L, win rate and score with 95% Wilson intervals, per side, mean returns and episode
length; `rank` — mean rank with a 95% interval, first-place share, a pairwise "who placed higher" matrix; `score` —
mean score with a 95% interval per team composition (the cross-play table); with one agent, mean return per seat.
Mean returns are also broken down by role. `--output` writes JSON.

The in-process API `colosseum.eval.play_lineups(env_fn=..., models=..., lineups=...)` plays any `PolicyModel`
instances in explicit lineups (the learning tests use it with a random legal-move player).

## Distributed mode (limited)

```bash
colosseum serve-weight-store --port 50051                                                     # machine A
colosseum run-learner -c cfg.yaml --agent agent_0 --traj-port 50052 --weight-store A:50051    # machine B
colosseum run-workers -c cfg.yaml --weight-store A:50051 -l agent_0=B:50052                   # machines C, D, ...
```

Trajectory chunks travel as numpy trees over gRPC. Limitations until SP5: games where every agent plays every role
only (each worker env draws its layout from `matchmaking.layouts` once, every seat plays the latest weights); no
coordinator, league, ratings, `metrics.jsonl` or WandB; every `run-workers` host collects
`total_timesteps / num_workers` per worker, and the learner's progress is `consumed_samples / total_timesteps`
(decisions, not env steps); with `training.seed` set, worker machines start with identical per-env seeds;
`run-learner` cannot resume; no fault tolerance or authentication; `deployment/` (Docker, Kubernetes) is untested.

## Status and limitations

| | Sub-project | Content |
|---|---|---|
| ✔ | SP1 Foundation and stabilization | correct single-machine training, stateful model protocol, run dir, metrics, lifecycle |
| ✔ | SP2 Game model | `GameSpec` / `MultiAgentEnv`, elimination, teams, roles, layouts, `Units`, Dict observations, bootstrap on the learner, centralized critic (awaiting the owner's acceptance) |
| | SP3 Players, league, warm start | scripted, frozen and external players; PFSP over snapshots; per-agent `init`/kickstart/critic warm-up; top-k snapshot storage |
| | SP4 Selection and observability | match log, OpenSkill / Bradley–Terry ratings, `colosseum tournament`, dashboard, snapshot ratings |
| | SP5 Distributed | hub and nodes, wire format, per-machine weight cache, fault tolerance, `max_policy_lag`, K8s images |
| | SP6 Speed and extensions | cuDNN RNN path, fast transformer unroll, GPU inference on workers, inference server, new algorithms |

Known limitations today:
- **Players:** only trainable agents; no scripted, frozen or external players in training (SP3). PFSP picks among
  agents' latest weights, not snapshots. Online ELO is a progress indicator, not a selection-grade rating.
- **Game model:** one agent cannot play roles with different spaces; no per-unit rewards or values; no built-in
  autoregression between action components; no PettingZoo adapter.
- **Speed:** the worker handles observation trees in Python per seat (hundreds of seats per env, as in Neural MMO,
  need SP6's vectorized path); recurrent and attention cores unroll step by step on the learner; worker inference is
  CPU only.
- **Distributed:** see above.
- **GPU:** not run on CUDA yet. All CUDA paths are covered only by `gpu`-marked tests; see
  [`docs/GPU_CHECKS.md`](docs/GPU_CHECKS.md).

## Tests

```bash
.venv/bin/python -m pytest -m "not gpu and not slow" -q     # full fast suite (CI), 1243 tests
.venv/bin/python -m pytest -m slow -v                       # learning tests of every demo game (12–16 min) + torch.compile
.venv/bin/python -m pytest -m gpu -v                        # CUDA machine only, see docs/GPU_CHECKS.md
```

Layout:
- `tests/unit`: pure functions and classes;
- `tests/contract`: the real `MatchRunner` / `RolloutLoop` feeding the real APPO in-process, the demo games;
- `tests/integration`: CLI and multi-process runs;
- `tests/learning`: does it learn (fast bandits; the slow demo-game runs).

Tests write only under pytest's `tmp_path`. Throughput and the units experiment are in
[`docs/benchmarks.md`](docs/benchmarks.md).

## Project structure

```
src/colosseum/
  cli.py launcher.py distributed.py eval.py
  core/         config, types (chunk v2, lineups, results), tree, specs (ObsSpec, ActionSpec, masks), roles,
                registry, validation, outcomes, errors, ipc, run_dir, threads
  envs/         game (GameSpec, MultiAgentEnv, StepResult), spaces (Units), contract (EpisodeTracker), vector
  networks/     model (PolicyModel, act), composed, base, heads (UnitsHead), dist/, cores, state, normalization
  algorithms/   appo, vtrace, base
  worker/       match_runner (MatchRunner), rollout_loop (RolloutLoop), buffers, rollout_worker
  learner/      learner process
  coordinator/  coordinator, matchmaker (lineups), ratings (per layout), checkpoint_manager, agent_pool
  metrics/      jsonl, aggregator, console, hub, wandb_logger
  transport/ weight_store/ bc/ utils/
examples/       coin_grid, tic_tac_toe, unit_harvest, team_tag, tron, predator_prey, coop_buttons,
                space_miners (Box2D), composite_action (chase)
configs/examples/
scripts/        setup-dev.sh, bench_throughput.py, units_experiment.py, measure_global_state.py
docs/           ENV_GUIDE.md, benchmarks.md, GPU_CHECKS.md
deployment/     Dockerfiles, docker-compose, k8s (untested)
```

# Colosseum

A reusable training framework for competitive bot-programming competitions (Lux AI, Neural MMO, CodeCraft, ...):
behavioural cloning → RL → self-play → league, built on an IMPALA-style asynchronous actor–learner (APPO with V-trace)
in PyTorch, without Ray.

> **Status (SP3 of 6).** Single-machine training is the supported mode for every game structure: solo, 1v1 with
> turn-based or simultaneous moves, one bot controlling many units, team vs team, free-for-all with elimination and
> 2–4 players in one run, asymmetric roles, and cooperative games. Each structure has a demo game that is trained
> end-to-end in the test suite. SP3 adds scripted and frozen players, a configurable league (latest weights, PFSP
> over stored snapshots, other agents, scripted/frozen anchors; shares, schedules, per-agent overrides, a custom
> matchmaker), snapshot retention, per-agent warm start (`init`, critic warm-up, kickstart from a neural or scripted
> teacher) and `colosseum record`. Read [Status and limitations](#status-and-limitations) before relying on anything else.

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
steps) takes about a minute (52–64 s over the six measured runs). Afterwards the latest checkpoint, played greedily, wins
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

### Competition pipeline: bot → BC → league

The game's scripted bot plays itself, its decisions become BC data, the BC weights start RL with a critic warm-up,
the bot itself keeps teaching (DAgger kickstart) and plays as an anchor. One config serves every step: copy
`configs/examples/unit_harvest_league.yaml` to `harvest.yaml` and name the BC output in it before it exists:

```yaml
agents:
  main:
    init: {from: bc_net, critic_warmup_steps: 30}   # start RL from the BC weights
    kickstart: {teacher: greedy}                     # the bot keeps teaching (DAgger labels)
  bc_net: {kind: frozen, path: bc.pt}                # written by step 2
  # greedy and random as in the example
```

```bash
colosseum record -c harvest.yaml --player greedy --num-matches 300 --output data/greedy --seed 0
colosseum bc -c harvest.yaml --agent main --data data/greedy --output bc.pt --epochs 5 --seed 0
colosseum train -c harvest.yaml --set run.name=harvest-league
NEW=$(ls -d runs/harvest-league/checkpoints/main/ckpt_v* | sort -V | tail -1)
colosseum eval -c harvest.yaml -a trained=$NEW -a greedy -a random -a bc_net --num-matches 100 --deterministic
```

`record`, `bc` and `eval` check only what they use (the env, the trainable agents' models, the scripted / frozen
agents they seat), so `bc_net` and `init.from` may point at a `bc.pt` that does not exist yet; `validate` and `train`
check everything. The example config itself can be used unchanged as well, with the warm start given on the
command line: `colosseum train -c configs/examples/unit_harvest_league.yaml --set agents.main.init.from=bc.pt
--set agents.main.init.critic_warmup_steps=30 --set agents.main.kickstart.teacher=greedy` (`record`, `bc` and
`eval` take `--set` too).

On an 8-core CPU recording takes about 17 s, BC about 9 s and training about 126 s (wall clock with process start-up;
measured once, 2026-10-10). The trained agent beats `random` in 100% of games and scores 0.50 against `greedy` (win 1,
draw 0.5; on this game a good policy mostly draws with the bot: all 100 games were draws). Recipes for leagues,
anchors, asymmetric games and custom matchmakers: [`docs/LEAGUE_GUIDE.md`](docs/LEAGUE_GUIDE.md) (Russian). `data/`
and `bc.pt` are yours to delete.

## Demo games

| Game | Config | Structure | Contract features |
|---|---|---|---|
| `coin_grid` | `coin_grid.yaml` | solo | Dict observation with a `uint8` grid, masks, step-limit truncation |
| `tic_tac_toe` | `tic_tac_toe.yaml` (also `_attention`, `_multi`) | 1v1 turn-based | one acting seat, masks, rewards to the waiting seat |
| `unit_harvest` | `unit_harvest.yaml` | one bot, many units, simultaneous | `Units`, units born and killed, entity lists with masks |
| `unit_harvest` league | `unit_harvest_league.yaml` | the same game, SP3 pipeline | agent `main`, scripted anchors `greedy` (`examples/unit_harvest/bots.py`) and `random` |
| `team_tag` | `team_tag.yaml` | 2v2 | local window per bot, `global_state` for a centralized critic, tagged (frozen-in-game) teammates keep team rewards; a `RandomBot` anchor (`anchors` 0.2) |
| `tron` | `tron.yaml` | FFA 2p/3p/4p | elimination (`terminated`), ranks by elimination order, several layouts in one run |
| `predator_prey` | `predator_prey.yaml` | 1 vs 2 | roles with different spaces, one agent per role |
| `coop_buttons` | `coop_buttons.yaml` | cooperative | one team, `score` outcome, mixed teammates, cross-play table |
| `space_miners` | `space_miners.yaml` | 1v1 (Box2D) | units with a continuous and a discrete component, entity list; reference only |
| `composite_action` | `chase.yaml` | 1v1 | tree action `Dict(direction=Discrete, speed=Box)`; reference only |

The slow learning tests train each of the seven games from `coin_grid` to `coop_buttons` and check it against random
players: coin_grid scores at least twice the random score; tic-tac-toe, unit_harvest and team_tag win at least 80%
(team_tag in a single run since SP3: its `RandomBot` anchor keeps self-play from settling into the passive
draw-seeking policy that about 2 of 9 SP2 runs found); tron wins 80% of 2p games and takes first place in at least
half of 4p games against three random cycles; each predator_prey role beats a random opponent at least 70%; both
coop_buttons agents' homogeneous teams reach `max(5 × random, 0.6 × scripted)` of the measured baselines. A further
slow test runs the whole pipeline above on `unit_harvest_league.yaml` (win against `random` ≥ 0.95, score against
`greedy` ≥ 0.40 and against the BC network ≥ 0.45). Numbers per seed are in
`docs/superpowers/reports/2026-10-09-sp2-acceptance.md` and `docs/benchmarks.md`.

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
| `train` | APPO metrics of one agent every `metrics.log_interval` train steps: `total_loss`, `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_fraction` (per decider) and `clip_fraction_joint`, `rho_mean`, `rho_clip_frac`, `c_clip_frac`, `ess`, `log_rho_abs_mean/p95`, `log_rho_joint_abs_mean/p95`, `deciders_valid_mean/max`, `boot_frac`, `pad_frac`, `explained_variance`, `grad_norm`, `lr`, `policy_version`, `critic_warmup` (1 during the critic warm-up), `kickstart_loss` / `kickstart_lambda` (with a teacher), `kickstart_label_frac` (scripted teacher: share of ACT slots with a DAgger label) |
| `episodes` | per agent since the previous record: `episodes`, `return_mean`, `length_mean`, `wdl` (W/D/L against `latest`, `past`, `arena` and `anchor` opponents), `seat_counts`, `by_layout.<layout>.<role>` with episodes, returns, lengths, `team_score_mean`, `eliminated_frac` and W/D/L by opponent type, and `opponents.<layout>` (the agent as data owner: `teams`, the played share of each opponent category `latest` / `snapshots` / `rivals` / `anchors` / `fallback`, and of each anchor) |
| `ratings` | `layouts`: per layout `outcome_kind`, `elo`, `win_rates`, `games`, `wr_vs_past`, `past_games`, `scores`, `cross_play`, `role_win_rates` (scripted and frozen agents are entities of their own), and `pfsp` (the owner's latest against each player: EMA score and games) |
| `system` | `env_steps`, `env_steps_per_sec`, `train_steps_per_sec` per agent, learner `queue_depths`, `parked_buffers`, `dropped_reward_episodes` (seat-episodes whose rewards were dropped because the seat never acted, summed over the latest counts of the workers that reported in the last few seconds, so it can drop when a worker stalls and read 0 at shutdown; the reliable signal is the WARNING logged on a worker's first drop, see ENV_GUIDE), `workers_reporting` |

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
- A Ctrl-C always ends `train`, `eval`, `bc`, `validate`, `run-learner`, `run-workers` and `serve-weight-store` with
  exit code 130 and `Interrupted` (or `Received SIGINT`), never a traceback, also when it arrives during startup.
  Rarely, Python loses a Ctrl-C that lands inside an import-time callback: `train` and the distributed roles then still
  stop as on a normal Ctrl-C, while `eval`, `bc` and `validate` finish their work first and then exit 130 with
  `Interrupted` (`serve-weight-store` keeps serving until the next Ctrl-C, then exits 130).
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
| `training` | `total_timesteps` 10,000,000 (global env steps), `seed`, `resume_from` (SP2's `kickstart_*` keys are translated into `kickstart` with one warning) |
| `matchmaking` | `opponents` (`latest` 0.7, `snapshots` 0.2, `rivals` 0, `anchors` 0.1; each a number or a schedule `{env_step: share}`), `anchors` (null = every scripted and frozen agent; a list of names or `{name: weight-or-schedule}`), `pfsp` (`weighting` `hard` / `balanced` / `uniform`, `exponent` 2.0, `halflife_games` 200), `layouts` (`{layout: weight}`; empty = every layout equally), `teammates` (`self` / `mixed`), `teammate_self_prob` 0.5, `shuffle_seats` (true), `matchmaker_class` (null; a `colosseum.league.BaseMatchmaker` subclass). SP2's `mode`, `self_play_ratio`, `latest_prob`, `pfsp_exponent` are translated with one warning (the league still behaves differently from SP2: new default mix, PFSP over snapshots, the opponent's snapshots in asymmetric games; the full list is in `docs/LEAGUE_GUIDE.md` §16) |
| `checkpoint` | `interval` 1000 (train steps), `keep_last` 20, `keep_every` 10 (every N-th snapshot kept for good; 0 = off), `save_optimizer` (true); the final snapshot is never deleted; SP2's `pool_size` = `keep_last` |
| `init` | `from` (null; `.pt`, checkpoint dir, run dir or a frozen agent's name), `strict` (true; false loads tensors with matching names and shapes), `critic_warmup_steps` 0 (train steps that update only the value path) |
| `kickstart` | `teacher` (null; a frozen or scripted agent's name, a checkpoint dir, or a `.pt` built with the student's architecture), `lambda` 1.0, `decay_steps` 50000 (train steps after the warm-up), `kl` (`forward` / `reverse`, neural teachers) |
| `metrics` | `log_interval` 10 (train steps), `console_interval_sec` 10, `use_wandb` (false), `wandb_project` (`colosseum`), `wandb_entity` |
| `bc` | `seq_len` 64 (sequence length for stateful models in `colosseum bc`) |
| `transport` | `grpc_max_message_mb` 64 (distributed mode); `mode` and `grpc_port` are unused until SP5 |
| `agents` | `{agent_id: {kind: trainable (default) / scripted / frozen, ...}}`: trainable — `roles`, partial overrides `networks`, `algorithm`, `learner`, `matchmaking`, `init`, `kickstart` (deep-merged onto the global sections); scripted — `class`, `kwargs`, `roles`; frozen — `path` (plus `networks` and `roles` for a `.pt`) |

**Agents, players and roles.** Every key of `agents` is an agent: `trainable` (the default kind) has its own learner;
`scripted` is a `colosseum.players.ScriptedBot` subclass (`colosseum.players.RandomBot` is built in); `frozen` plays
fixed weights from a checkpoint dir or a `.pt`. Without any trainable agent the config gets an implicit `agent_0` with
the global settings. `roles` lists the roles an agent plays (omitted = every role; a trainable or frozen agent's roles
must share their spaces). Only seats with a trainable agent's latest weights collect data. An agent id may contain
letters, digits, `_` and `-` (not starting with `-`), no `.`, and cannot be `ratings`, `system`, `episodes` or `train`.

**Matchmaking.** Every env has an owner: the trainable agents take turns by env index. For each match the built-in
matchmaker picks a layout (`layouts`, only layouts with a seat for the owner's roles), the owner's team (its latest
weights on one seat of its role), and for every other team independently a core category by the `opponents` shares:
the owner's latest weights (or, in an asymmetric game, the latest weights of an agent that plays that team), a stored
snapshot of an agent that plays the team (PFSP), another trainable agent's latest weights (`rivals`, PFSP) or an anchor
(by weight). Shares of categories that are empty right now go to the others. The team's other seats follow
`teammates`. `docs/LEAGUE_GUIDE.md` has the full rules and recipes; `colosseum validate` prints each agent's effective
mix.

**Ratings** are kept per layout. Two or more teams: ELO over team pairs (each pair is one comparison by rank; its
weight `K / (T - 1)` is split among the counted member pairs of different agents), a fractional win-rate matrix and
`wr_vs_past` (the latest weights against the agent's own checkpoints, last 500 pairs). One team: mean score with EMA
and a 95% interval per agent, and (two or more seats) a cross-play table "team composition → mean score" (mixed
compositions appear with `teammates: mixed`). Scripted and frozen agents are entities of their own in the ELO and
win-rate tables; snapshots count for their agent. `ratings.json` also holds the PFSP table per layout, and
`metrics.jsonl` the shares of played episodes per opponent category and anchor.

**Resume.** `training.resume_from` accepts a checkpoint dir, a previous run dir (each agent takes its latest
checkpoint there) or a `.pt` state dict (weights only, e.g. from `colosseum bc`). An explicit resume is strict: a
missing or malformed `meta.json`, or a role signature that does not match the agent's roles, is a config error naming
the path. A resume from a run dir also carries the stored snapshots over into the new run (hard links, else
copies), so the opponent pool survives the resume; `training.resume_from` takes precedence over `init`.

## Recording, behavioural cloning and warm start

```bash
colosseum record -c cfg.yaml --player <scripted|frozen agent | name=path> [--against <player> ...] [--layout L ...] \
  --num-matches N --output data/x [--num-envs E] [--seed S] [--deterministic] [--set k=v ...]
colosseum bc -c cfg.yaml --agent main --data data/x [--data more.pt ...] --output bc.pt --epochs 20 [--seed S] \
  [--set k=v ...]
```

`record` plays the player on the eval engine and writes its decisions per role (`<output>/<role>/part-NNNNN.pt`,
every seat-episode contiguous) plus `record.json`. Without `--against` the player takes every seat; with it, each
opponent forms a pair with the player as in `eval`, and only the player's seats are recorded. `bc` reads `.pt` files,
directories of them and `record` output directories (the folders of the agent's roles), `--data` repeatable; other
options: `--batch-size` (256), `--lr` (1e-3), `--seq-len` (default `bc.seq_len`), `--seed` (initial weights and
minibatch order). Both commands, like `eval`, check only what they use, so the config may already name the BC output
(`init.from`, a frozen agent with `path: bc.pt`) before it exists. The loss is `-log_prob` of the
recorded action (the mean over valid deciders for actions with units); masks are applied, and an expert action the
mask forbids is a data error.

Warm start is per agent (`init`, `kickstart`; global sections are the defaults): `init.from` loads weights only
(strict, or partial with `strict: false`), `critic_warmup_steps` first trains the value path alone while the learner's
policy stays bit-identical (the statistics of the observation normalizers are frozen; the critic's global-state
normalizer keeps learning; the workers act with their own initial weights until the first weight sync, as in SP2),
and `kickstart` adds `lambda * KL(teacher‖student)` per decider (a neural teacher: a frozen agent or a checkpoint dir
brings its own architecture, a bare `.pt` path is built with the student's) or `lambda * -log pi(teacher's action)` (a
scripted teacher, DAgger labels written by the workers), decaying linearly over `decay_steps` after the warm-up.
`training.resume_from` takes precedence over `init`.

## Evaluation

```bash
colosseum eval -c cfg.yaml -a A=<ckpt-dir|.pt> [-a <scripted or frozen agent>] ... [--layout L ...] --num-matches N [--num-envs E] [--output r.json] [--deterministic] [--seed S] [--set k=v ...]
```

Matches run on the same engine as training (`MatchRunner`): per-seat model state, acting seats and masks. An agent's
architecture and roles come from its checkpoint's `meta.json`; a `.pt` is built from the config agent of the same
name, or from the global `networks` with every role. By default every layout the agents can fill is played.
A bare name (`-a greedy`) plays a scripted or frozen agent of the config; a trainable agent needs `name=path`.
- Two or more teams, two or more agents: for every pair (a, b) match m gives team i to `(a, b)[(i + m) % 2]` where
  the roles allow; teams are filled homogeneously; when only one orientation of the pair can fill the layout
  (hunter and prey, or an agent that plays only some roles), every match uses it; `--num-matches` is per pair and
  layout (an odd count is rounded up, so every agent plays every side equally often).
- Two or more teams, one agent: all teams are that agent.
- One team: each agent in a homogeneous team, plus mixed compositions for cross-play.

Reports per layout: `wdl` — W/D/L, win rate and score with 95% Wilson intervals, per side, mean returns and episode
length; `rank` — mean rank with a 95% interval, first-place share, a pairwise "who placed higher" matrix; `score` —
mean score with a 95% interval per team composition (the cross-play table); with one agent, mean return per seat.
Mean returns are also broken down by role. `--output` writes JSON.

The in-process API `colosseum.eval.play_lineups(env_fn=..., models=..., lineups=...)` plays any `PolicyModel`
instances and scripted players (`colosseum.worker.match_runner.ScriptedPlayer`) in explicit lineups (the learning
tests use it with `RandomBot`).

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
SP3 features are local only: scripted and frozen agents, `init`, critic warm-up, a custom matchmaker, per-agent
`matchmaking` or `kickstart` are refused with a config error pointing to SP5, and any opponent mix is reduced to latest
weights with one warning.

## Status and limitations

| | Sub-project | Content |
|---|---|---|
| ✔ | SP1 Foundation and stabilization | correct single-machine training, stateful model protocol, run dir, metrics, lifecycle |
| ✔ | SP2 Game model | `GameSpec` / `MultiAgentEnv`, elimination, teams, roles, layouts, `Units`, Dict observations, bootstrap on the learner, centralized critic (accepted and merged 2026-10-09) |
| ✔ | SP3 Players, league, warm start | scripted and frozen players, opponent mixes with PFSP over snapshots and anchors, snapshot retention, per-agent init / critic warm-up / kickstart (neural or DAgger), `colosseum record` (implemented on `sp3-league`; acceptance report in docs/superpowers/reports) |
| | SP4 Selection and observability | match log, OpenSkill / Bradley–Terry ratings, `colosseum tournament`, dashboard, snapshot ratings, top-k snapshot storage |
| | SP5 Distributed | hub and nodes, wire format, per-machine weight cache, fault tolerance, `max_policy_lag`, K8s images, distributed league, scripted/frozen players and SP3 warm start across machines |
| | SP6 Speed and extensions | cuDNN RNN path, fast transformer unroll, GPU inference on workers, inference server, new algorithms |

The table is a summary; the full scope of each sub-project and the parked items are in [`CLAUDE.md`](CLAUDE.md) («Roadmap»), which is authoritative.

Known limitations today:
- **Players:** external players are frozen agents (`.pt` or checkpoint dir); no top-k snapshot storage or snapshot
  ratings yet (SP4); online ELO is a progress indicator, not a selection-grade rating; scripted bots run one Python
  call per seat (no batched bot API).
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
.venv/bin/python -m pytest -m "not gpu and not slow" -q     # full fast suite (CI), 1689 tests
.venv/bin/python -m pytest -m slow -v                       # 10 tests, about 15 min: the 7 demo-game learning tests,
                                                            # the unit_harvest pipeline test, 2 torch.compile checks
.venv/bin/python -m pytest -m gpu -v                        # 22 tests, CUDA machine only, see docs/GPU_CHECKS.md
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
  cli.py launcher.py distributed.py eval.py record.py
  core/         config, types (chunk v2, lineups, results), tree, specs (ObsSpec, ActionSpec, masks), roles,
                registry, validation, outcomes, errors, ipc, run_dir, threads
  envs/         game (GameSpec, MultiAgentEnv, StepResult), spaces (Units), contract (EpisodeTracker), vector
  networks/     model (PolicyModel, act), composed, base, heads (UnitsHead), dist/, cores, state, normalization
  algorithms/   appo, vtrace, base
  worker/       match_runner (MatchRunner), rollout_loop (RolloutLoop), buffers, rollout_worker
  learner/      learner process, factory (algorithm, init, kickstart teacher per agent)
  coordinator/  coordinator, ratings (per layout), checkpoint_manager (keep_last / keep_every, run-dir import)
  players/      scripted bots (ScriptedBot, RandomBot), fixed-player registry (scripted / frozen agents)
  league/       matchmaker interface (BaseMatchmaker, MatchmakerContext), mixture matchmaker, PFSP, schedules,
                lineup checks
  metrics/      jsonl, aggregator, console, hub, wandb_logger
  transport/ weight_store/ bc/ utils/
examples/       coin_grid, tic_tac_toe, unit_harvest, team_tag, tron, predator_prey, coop_buttons,
                space_miners (Box2D), composite_action (chase)
configs/examples/
scripts/        setup-dev.sh, bench_throughput.py, units_experiment.py, measure_global_state.py,
                team_tag_anchors.py, pipeline_vs_scratch.py
docs/           ENV_GUIDE.md, LEAGUE_GUIDE.md, benchmarks.md, GPU_CHECKS.md
deployment/     Dockerfiles, docker-compose, k8s (untested)
```

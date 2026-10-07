# R5 — Config, registry/plugins, deployment, metrics/monitoring, docs, DX

Scope: `core/config.py`, `core/registry.py`, `cli.py`, `metrics/wandb_logger.py` and every place that produces metrics, `launcher.py`/`distributed.py` (wiring), `deployment/**`, `configs/**`, `examples/**`, `pyproject.toml`, `README.md`, `CLAUDE.md`.
Environment: torch 2.14.1+cpu venv, WANDB_MODE disabled/offline, scratch dir `scratchpad/r5_config/`. Docker and kubectl are not usable in this WSL distro, so deployment findings come from reading the files. No repo files were modified.

Evidence tags: **RUN** = verified by running; **READ** = verified by reading code (surrounding code re-checked).

---

## Findings

### Metrics / monitoring

**R5-01 — critical — metrics/bug — `launcher.py:127-193`, `launcher.py:58-124` (spawn targets), `launcher.py:669`**
What's wrong: `logging.basicConfig` runs only in the main process (`run_training`). Learner and worker processes are started with `spawn` and never set up logging, so every `logger.info` they emit is dropped: "Learner [x]: step=…, loss=…", "starting", "finished", "sent checkpoint", "Worker N: finished…". Only WARNING and above get through, via Python's lastResort handler.
Evidence: RUN. The same 2-agent config was run twice. Without a logging hook, `sp2.log` has no learner or worker lines at all. With a `sitecustomize.py` that calls `basicConfig`, `sp2b.log` shows "Learner [a]: starting…", "step=10…" and so on.
Impact: every example config sets `use_wandb: false` (R5-03). In that case a local run shows nothing about training progress: no loss, no step count, no episode stats. The user only sees the startup banner and "All processes stopped".
Fix: add a `setup_logging(level, run_dir)` helper and call it first thing in every process target (`_learner_target`, `_worker_target`, `_dist_worker_target`, the subproc env workers). Use a QueueHandler or per-process log files under the run dir.

**R5-02 — critical — metrics/bug — `metrics/wandb_logger.py:47-50`, `launcher.py:570-576`**
What's wrong: all agents log into one WandB run with `step=<that agent's train_step>`. WandB needs `step` to increase monotonically within a run, so a lagging agent's points are silently dropped.
Evidence: RUN (offline WandB). Logged alpha@100, beta@95, beta@99, alpha@110. The run history contains only `agent_alpha/loss`; nothing from `agent_beta` was recorded.
Impact: in league or multi-agent training, effectively only the fastest learner's curves survive. This hits exactly the scenario the owner cares about (comparing agents).
Fix: log without `step=` (or with a global monotonic counter). Put per-agent x-axes in the metrics (`agent/train_step`, `agent/env_steps`) and register them with `wandb.define_metric("agent_beta/*", step_metric="agent_beta/train_step")`. Alternatively use one WandB run per agent with a shared `group=`.

**R5-03 — high — metrics/design — `worker/rollout_worker.py:620-663`, `coordinator/coordinator.py:139-187`, `launcher.py:558-564`**
What's wrong: episode return, episode length, outcome vs. each opponent, ELO, the win-rate matrix and checkpoint events are computed (worker → `results_queue` → `Coordinator.report_match_result`) but never logged anywhere. `Coordinator.get_ratings_summary()` has no callers. ELO and win rates live only in memory and are lost when the process exits. Checkpoint `meta.json` always has `metrics: {}` because the launcher never passes metrics.
Evidence: READ (grep shows `get_ratings_summary` is only defined, never called; `maybe_save_checkpoint` is called without `metrics`).
Impact: the owner cannot see reward curves, win rates against historical, frozen or other agents, or ELO over time, and has no data for picking the best agents. These are the core signals for self-play and league training.
Fix: in `_monitor_loop`, aggregate results every N seconds and log them: `agent/ep_return_mean`, `ep_len_mean`, `winrate_vs/<opp>`, `elo/<agent>`, a `wandb.Table`/heatmap for the win-rate matrix, and `checkpoint/saved` events. Persist `ratings.json` (ELO + WR matrix + match counts) to the run dir on each refresh and at shutdown. Write ELO and recent win rate into each checkpoint's `meta.json`.

**R5-04 — high — metrics/design — `distributed.py:222-233` (`metrics_queue=None`), `distributed.py:288-304` (no `results_queue`)**
What's wrong: distributed mode has no metrics path at all. `run-learner` passes `metrics_queue=None`, and nothing in `distributed.py` creates a WandB run. `run-workers` passes no `results_queue` and there is no coordinator, so no episode or outcome data is collected anywhere.
Evidence: READ.
Impact: the setup the owner most wants (CPU boxes plus GPU boxes) gives zero monitoring beyond learner stdout lines every 10 steps.
Fix: have the learner log directly to WandB (one run per role/agent, `group=<run_id>`, `job_type=learner|workers`). Have workers aggregate episode stats locally and either log them to their own WandB run or ship them to a small coordinator/metrics service (see the Assessment).

**R5-05 — high — metrics — `algorithms/appo.py:235-250, 288-319`, `learner/learner.py:140-148`**
What's wrong: standard async-RL diagnostics are missing:
- grad norm: the return value of `clip_grad_norm_` is thrown away
- explained variance and value/target statistics
- policy lag: `policy_version` minus `chunk.behavior_policy_version`, even though the chunk already carries it
- learner throughput (train steps/s, samples/s), time spent waiting for data vs. computing, queue depth
- worker env SPS, inference time, chunks/s per worker
- weight-sync age on workers
- AMP scaler scale and V-trace rho clip fraction
- GPU memory
Evidence: READ.
Impact: you cannot tell whether a learner is starved or a bottleneck, whether V-trace is fighting large lag, or whether the value function is learning. You also cannot size the CPU-box vs. GPU-box split, which is the owner's top priority.
Fix: add these to `train_step` and learner metrics (cheap: they are scalars that are already available). Add a periodic `worker_stats` message (SPS, episodes, mean return) sent over the existing results queue, or over gRPC in distributed mode.

**R5-06 — medium — metrics/design — `metrics/wandb_logger.py` (whole module)**
What's wrong: WandB is the only sink. There is no TensorBoard, CSV or JSONL fallback. Every run gets the same name (`colosseum_agent_0`), with no group, tags, `resume` or run id. WandB is a hard dependency in `pyproject.toml`. `log_config` logs the post-override config (good), but only when WandB is enabled.
Evidence: READ.
Impact: offline machines or competitions without WandB get no metrics. Runs collide visually and nothing ties a checkpoint to a run.
Fix: add a `MetricsSink` interface with `JsonlSink` always on (writing `run_dir/metrics.jsonl`), plus optional `WandbSink` and `TensorBoardSink`. Generate `run_id = <timestamp>-<name>` and use it for the WandB name/group and for the run dir.

### Configuration

**R5-07 — high — ux/bug — `core/config.py` (all models use the pydantic default `extra="ignore"`)**
What's wrong: unknown keys are ignored at every level, including inside `agents`.
Evidence: RUN. All of the following were accepted silently: `rollout.num_worker: 64` (effective `num_workers` stayed 2), top-level `rollot:`, `agents.x.algoritm:`, `training.kickstart_teachr:`, `agents.beta.training.resume_from` (not an override field). `colosseum validate -c typo.yaml` printed "Config is valid."
Impact: a typo in a hyperparameter or in `kickstart_teacher` silently trains the wrong experiment, possibly for hours.
Fix: set `model_config = ConfigDict(extra="forbid")` on a shared base model. Also validate `learner.device` (regex `auto|cpu|cuda(:\d+)?|mps`), `amp_dtype: Literal["float16","bfloat16"]` and `recurrent_type: Literal["lstm","gru"] | None`. All three accepted garbage in the run (`device: gpu`, `amp_dtype: garbage`, `recurrent_type: transformer`). For `amp_dtype` this is worse than a validation miss: `appo.py:47` uses `getattr(torch, name, float16)`, so a bad value silently becomes float16.

**R5-08 — high — bug/design — `core/config.py:297-317` (`get_agent_config`)**
What's wrong: a per-agent override replaces the whole section instead of deep-merging into the global one. Any field the override doesn't set falls back to the model default, not to the global value.
Evidence: RUN. With global `lr_schedule: constant` and `agents.beta.algorithm: {learning_rate: 1e-4}`, beta ends up with `lr_schedule: linear`. `agents.gamma.learner: {batch_chunks: 64}` resets gamma's `queue_size` to 64 (global was 32). A partial `networks` override (for example only `recurrent_type: lstm`) is rejected because `encoder_class`/`policy_class`/`value_class` are required. The README's own example (`agent_beta: algorithm: learning_rate: 1e-4  # uses different LR`) silently changes other hyperparameters too.
Impact: league comparisons between "same agent, different LR" are not what the user thinks they are.
Fix: store overrides as raw `dict[str, Any]` and deep-merge them into `self.model_dump()` before `model_validate`. Validate override keys against the section schema, failing on unknown keys.

**R5-09 — high — design — `core/config.py:262-268` (`AgentConfig`), `launcher.py:399-402`, `launcher.py:267-307`**
What's wrong: `AgentConfig` only allows `networks`, `algorithm` and `learner` overrides. As a result:
- `training.resume_from` and `kickstart_teacher` are global and applied to every agent. Two agents with different architectures plus one `resume_from` .pt will fail in `load_state_dict` for one of them.
- There is no way to declare a frozen-checkpoint agent or a scripted bot in config. `AgentPool.register_frozen` and `register_scripted` exist but nothing calls them, and workers cannot run scripted policies.
- There is no per-agent `role` (main/exploiter), no `trainable: false`, and no per-agent `checkpoint_interval`.
Evidence: READ (plus RUN showing `agents.beta.training` is silently ignored).
Impact: two owner scenarios cannot be expressed: warm-start variants (A from BC, B from scratch, C from run X checkpoint 500) and the final arena (bots with different weights and architectures plus scripted bots).
Fix: make the agent the unit of config, for example:
```yaml
agents:
  main:     {type: trainable, init_from: bc.pt, kickstart: {teacher: bc.pt, lambda: 1.0}, algorithm: {...}}
  old_best: {type: frozen, checkpoint: runs/x/agent_0/ckpt_v900, networks: {...}}
  rules:    {type: scripted, class: my_game.bots.GreedyBot, kwargs: {...}}
```
Teach the worker to run a `ScriptedPolicy` interface (`act(obs, info) -> action`) in the same slot machinery.

**R5-10 — medium — bug — `cli.py:14-32`, `launcher.py:674-694`, `distributed.py:382-393`**
What's wrong: `--set` overrides have several problems:
- (a) They are applied after the global seeding. `run_training` seeds from the YAML's `training.seed` and only then applies overrides, so `--set training.seed=…` does not seed the main process.
- (b) An unknown leaf key is silently ignored (RUN: `rollout.num_worker=16` → OK, `num_workers` stays 2).
- (c) An unknown section raises a raw `KeyError: 'rollot'` traceback (RUN).
- (d) Overriding inside a null agent section fails: `agents.agent_alpha.algorithm.learning_rate=1e-4` → `TypeError: 'NoneType' object does not support item assignment` (RUN).
- (e) `null`/`none` becomes the string `"null"` (RUN: `training.resume_from=null` → `'null'`, which is then treated as a file or checkpoint id).
- (f) Lists and dicts cannot be set.
- (g) The override logic is copy-pasted in two places.
Fix: one `apply_overrides(cfg, list[str])` in `core/config.py` that parses values with `yaml.safe_load(value)` (handles null, bools, numbers, lists), creates missing dicts, validates with `extra="forbid"`, and runs before seeding.

**R5-11 — medium — bug — `launcher.py:356-361`**
What's wrong: `total_train_steps` is computed from the global `cfg.learner.batch_chunks` and passed to every learner. An agent with a `learner.batch_chunks` override gets the wrong LR-schedule horizon and a wrong stop point. The distributed path (`distributed.py:192`) correctly uses `acfg`.
Evidence: READ (override verified to take effect via RUN).
Fix: compute this per agent from `agent_configs[aid].learner.batch_chunks`.

**R5-12 — medium — design/bug — reproducibility: `launcher.py:127-193`, `coordinator/checkpoint_manager.py:80-130`**
What's wrong:
- (a) Learner processes are never seeded. Only the main process and the workers are, so network initialization differs between runs with the same `training.seed`.
- (b) The resolved config (after overrides) is never written to disk. It only goes to WandB, and only when WandB is enabled.
- (c) Checkpoints are not self-describing. `meta.json` has no network class paths, no network kwargs, no config hash and no metrics, so to load a checkpoint you must already know which YAML produced it. This is exactly what an arena of mixed architectures needs.
- (d) There is no run directory. Everything goes into `checkpoint.dir`.
Evidence: READ.
Fix: create `runs/<run_id>/` containing `config.resolved.yaml`, `metrics.jsonl`, `ratings.json` and `checkpoints/<agent>/ckpt_vN/{model.pt, meta.json with networks section + env class + policy_version + elo}`. Seed learners with `seed + 10_000 + agent_index`.

**R5-13 — high — bug — `configs/examples/*.yaml` (`checkpoint.dir: ./checkpoints`), `coordinator/checkpoint_manager.py:59-75`, `coordinator/coordinator.py:83-92`**
What's wrong: `CheckpointManager` scans the existing directory at startup, and the self-play matchmaker immediately uses whatever is there as opponents. Every example uses `./checkpoints` and `agent_0`.
Evidence: RUN (`t_stale.py`). Saved a tic-tac-toe checkpoint, then started the chase config with the same dir. The slot map was `[['latest','ckpt_v50'], …]` and `load_state_dict` raised "Error(s) in loading state_dict" (the worker would crash at startup).
Impact: running a second example, or changing the architecture and rerunning, crashes. Worse, rerunning the same config silently trains against the previous run's checkpoints.
Fix: default to a per-run dir (R5-12). Store an architecture fingerprint in `meta.json` and skip or raise a clear error on mismatch. Reusing earlier checkpoints should be explicit (`opponents_from: runs/x`).

**R5-14 — medium — design — dead or misleading config fields**
What's wrong: `transport.mode` and `transport.grpc_port` are never read (`grep`: no uses). `algorithm.name` is only printed. `training.phase: bc` is accepted but behaves exactly like `self_play` (BC is the separate `colosseum bc` command). The README documents all of them as working. `transport.mode: grpc` is the key piece of the README's multi-machine recipe, and it does nothing.
Evidence: READ (grep) and RUN (accepted).
Fix: remove them or make them functional. For example, `transport.mode=grpc` in `train` could fail with "use run-learner/run-workers", and `phase` could become `matchmaking: {type: self_play|league|…}`.

**R5-15 — high — bug — multiple agents with `phase: self_play` hangs forever — `launcher.py:420-422`, `coordinator/matchmaker.py:53-136`**
What's wrong: matches are always generated for `trainable_agents[0]`. In `self_play`, every slot goes to that agent, so the other agents' learners never receive a chunk. The monitor loop waits for all learners to exit. The worker then blocks forever in `tq.put` on the finished agent's full queue.
Evidence: RUN (`sp_two_agents.yaml`, timeout 170 s). "Learner [a]: finished. Total train_steps=11"; learner b never logged a step; the process was still alive at timeout, while the 1-agent baseline finished in about 50 s.
Fix: at config load, reject `len(agents)>1 && phase==self_play`, or run self-play per agent by distributing envs across agents. Also add a starvation watchdog that warns when a learner has had no chunk for X seconds.

**R5-16 — medium — ux — `core/registry.py:106-164` (`validate_config`) and `cli.py:143-154`**
What's wrong: `validate` misses the most common shape bugs:
- `env.num_players` (config) vs. `env.num_players` (env property) mismatch. RUN: tic-tac-toe with `num_players: 3` → "Config is valid.", then the worker crashed with `IndexError: index 2 is out of bounds for axis 0 with size 2`.
- Policy head with the wrong number of logits (10 vs. `Discrete(9)`). RUN: passes validation.
- Value head returning `[B,1]` instead of `[B]`. RUN: passes, because only `values.shape[0]==1` is checked.
- It never calls `env.reset()`/`step()`, so obs dtype/shape from the real env, the info `action_mask` shape and the dict-keyed API contract are not checked.
Errors also surface as full tracebacks rather than a one-line message.
Fix: check `env.num_players == cfg.env.num_players`, `values.shape == (B,)`, and that the distribution's `action_dim`/`n` matches the `ActionSpec`. Run `reset()` plus a few random `step()` calls and check the mask shape against the logits. Catch `ValueError` in the CLI and print it cleanly. Better still, derive `num_players` from the env and drop the config field.

**R5-17 — medium — bug/ux — `launcher.py:534-539`, `launcher.py:665-697`**
What's wrong: `colosseum train` exits with code 0 when all workers crashed (it logs "All worker processes exited unexpectedly" and returns normally).
Evidence: RUN (num_players mismatch run: rc=0).
Impact: scripts, CI and k8s Jobs cannot detect a failed run.
Fix: track failure state and `sys.exit(1)` when learners or workers die with a non-zero exitcode or before reaching their target.

### Registry / plugin model

**R5-18 — medium — design — `core/registry.py:68-103`**
What's wrong:
- (a) The same `networks.kwargs` dict goes to encoder, policy and value constructors, so every class must accept every key (`ChaseEncoder.__init__(self)` takes none). RUN: `kwargs: {hidden: 256}` → "unexpected keyword argument 'hidden'".
- (b) Heads are not given their input dim (`latent_dim` or `recurrent_hidden_size`), and no network gets `observation_space`/`action_space`. Users must hardcode `nn.Linear(64, 9)`, so changing `recurrent_hidden_size` in YAML requires a code change.
- (c) Only `algorithm_class` is pluggable. The matchmaker, weight store, transport, scripted bots and outcome function are hardcoded.
- (d) User code reaches remote machines only through the convention that the process cwd contains the package (`sys.path.insert(0, ".")` in worker/learner targets).
Fix: `build_network` should pass `obs_space=`, `action_space=` and `input_dim=` (computed), with per-component `encoder_kwargs`/`policy_kwargs`/`value_kwargs` and a shared `kwargs` filtered by signature. Add `matchmaker_class` and `scripted` agent classes to config. Add `--code-dir` / `COLOSSEUM_USER_PATH` that is prepended to `sys.path` in every process, or recommend `pip install -e my_game/` and document it.

**R5-19 — high — bug/ux — console-script entry point cannot import user/example code — `pyproject.toml:[project.scripts]`, `core/registry.py:119`**
What's wrong: the `colosseum` console script does not put cwd on `sys.path`, so `validate_config`/`build_network` in the main process cannot import `examples.*` or a user's `my_game.*`. Only `python -m colosseum …` or `PYTHONPATH=.` works. The child processes add `"."` themselves, which is inconsistent.
Evidence: RUN. From the repo root, `colosseum validate -c configs/examples/*.yaml` → `ModuleNotFoundError: No module named 'examples'` for all 4 configs, and `colosseum train -c configs/examples/tic_tac_toe.yaml` (the README Quick Start) fails the same way. `python -m colosseum validate` works (space_miners fails on the undeclared `Box2D`, see R5-31).
Impact: the very first command in the README fails for every new user.
Fix: in `cli.main()`, `sys.path.insert(0, os.getcwd())` (plus an optional `--code-dir`). Better: add a `code_paths:` config key applied in every process.

### Deployment

**R5-20 — critical — deploy/bug — `deployment/k8s/learner.yaml:17-58`, `deployment/k8s/worker.yaml:6-36`, `distributed.py:119-238`**
What's wrong: run-to-completion processes are deployed as `Deployment`s (restartPolicy Always).
- When the learner reaches `total_train_steps` it exits, k8s restarts it, and it starts training from scratch. The distributed learner never resumes: `resume_from` is ignored and no `resume_state` is passed. It then pushes `policy_version` 1, 2, … to the weight store.
- Workers keep `_last_version` from the old learner. `GRPCWeightSource.get_nowait` (`distributed.py:92-100`) ignores any version ≤ last, so workers keep acting with stale weights while sending chunks to a learner training a fresh network.
- The same happens on any learner crash or OOM (for example `GRPCWeightSink.put` raising when the weight store is briefly down, `distributed.py:110-111`: uncaught `RpcError` kills the learner).
- Workers also exit after their own `total_timesteps // num_workers` and restart in a loop. Each pod or replica counts its budget independently.
Evidence: READ.
Impact: k8s training never terminates cleanly, can silently throw away learned weights, and leaves workers on stale policies after any restart.
Fix: run learners as `Job`s (or `restartPolicy: OnFailure` StatefulSets) that auto-resume from the latest checkpoint in the PVC. Give the weight store an epoch or learner-run-id so workers accept a version reset. Workers should run until told to stop (coordinator or learner "done" flag in the store) rather than counting their own budget.

**R5-21 — high — deploy — `deployment/k8s/worker.yaml:38-56` (HPA)**
What's wrong: the HPA scales on CPU utilization. Rollout workers are busy loops, so per-pod CPU is always at or above the 70 % target, and the HPA goes straight to `maxReplicas: 16`. Nothing reflects learner capacity. When the learner queue is full, `TrajectoryServicer` drops chunks after a 5 s timeout (`transport/grpc_transport.py:44-49`), so extra pods add only waste and staler data.
Evidence: READ.
Fix: use a fixed `replicas` per experiment, or scale on a custom metric that the code would need to export (learner queue fill or `samples_dropped`, via a Prometheus endpoint). Simplest option: drop the HPA and document "set replicas = desired worker machines".

**R5-22 — high — deploy — `deployment/Dockerfile.base`**
What's wrong:
- (a) One image for every role installs PyPI `torch`. On x86_64 Linux that is the CUDA build with its NVIDIA wheels (several GB), even for CPU worker boxes and the weight store.
- (b) Source is copied before `pip install`, so every code or config edit reinstalls torch. There is no layer caching.
- (c) It runs as root.
- (d) `build-essential` stays in the final image.
- (e) There is no `.dockerignore`, so runs, checkpoints, `.venv` and `wandb/` in the repo get sent as build context.
- (f) `grpcio-tools`, a codegen-only package, is installed at runtime.
- (g) There is no GPU story: compose has no `deploy.resources.reservations.devices` for the learner, the k8s GPU request is commented out, and there is no nodeSelector or toleration. `device: auto` silently falls back to CPU.
Evidence: READ.
Fix: use two targets in one multi-stage Dockerfile. `cpu` installs torch from `https://download.pytorch.org/whl/cpu` and is used by workers and the weight store; `cuda` uses a `pytorch/pytorch:*-cuda*-runtime` base or the cu12x index and is used by learners. Install dependencies from `pyproject` first (copy only `pyproject.toml` plus a stub package), then `COPY src`. Run as a non-root `USER`. Add `.dockerignore`. Add a GPU reservation in compose and an `nvidia.com/gpu: 1` request with a nodeSelector in `learner.yaml`. Use `tini` (or `docker run --init`) so SIGTERM reaches Python. `run-workers` installs no SIGTERM handler, and Python as PID 1 ignores SIGTERM, so each scale-down or rollout waits the 30 s grace period.

**R5-23 — high — deploy — config and user-code delivery: `deployment/Dockerfile.base:14-15`, all `k8s/*.yaml`, `docker-compose.yaml`**
What's wrong:
- Configs and game code are baked into the image (`COPY examples/ configs/`). The k8s and compose manifests hardcode `configs/examples/tic_tac_toe.yaml`.
- There is no ConfigMap, no mounted config volume and no `COLOSSEUM_CONFIG` env var, and nothing documents how to add your own game package. Every hyperparameter change means rebuilding and pushing the image.
- Manifests use `image: colosseum-base:latest` with no registry and no `imagePullPolicy`. For `:latest` the default is `Always`, so a real cluster hits ImagePullBackOff. Nothing documents building or pushing it.
- `kubectl apply -f deployment/k8s/` processes files alphabetically, so `learner.yaml` comes before `namespace.yaml` and the first apply fails with "namespace colosseum not found".
Evidence: READ.
Fix: mount configs from a ConfigMap (or a volume) and pass `--config /config/run.yaml`. Use a user Dockerfile `FROM colosseum:cpu` + `COPY my_game/ && pip install -e`. Rename to `00-namespace.yaml` or add a `kustomization.yaml` with image and config generators. Document `docker build -t <registry>/colosseum:<tag>`.

**R5-24 — medium — deploy — `deployment/Dockerfile.worker:3-4`, `deployment/Dockerfile.weight-store`**
What's wrong: `Dockerfile.worker` runs `colosseum train`, the full single-machine pipeline, not `run-workers`. Neither `Dockerfile.worker` nor `Dockerfile.weight-store` is used by compose or k8s; both use `Dockerfile.base` with explicit commands.
Impact: these images are misleading leftovers.
Fix: delete them or turn them into targets of a single Dockerfile.

**R5-25 — medium — design/deploy — distributed mode lacks the coordinator: `distributed.py:18-22, 331-337`**
What's wrong: there is no coordinator service, so multi-machine runs have no historical-checkpoint opponents, no PFSP/league, no ELO or win rates, and no frozen or scripted opponents. Every distributed match is latest-vs-latest. `slot_agent_map` is a fixed round-robin; with 2 agents and 2 players, `(e*2+p)%2 == p`, so agent A is always seat 0 and B always seat 1 (seat bias).
Evidence: READ.
Impact: the owner's main deployment target (CPU boxes plus GPU boxes) supports only plain self-play. Cross-ref R3/R4.
Fix: see the Assessment, a coordinator gRPC service (or simply the existing `Launcher` monitor loop exposed over gRPC/HTTP) that workers poll for `WorkerCommand`s and report results to.

**R5-26 — medium — deploy/ux — no simple bare-metal path, and the README's multi-machine recipe is wrong — `README.md:416-436`**
What's wrong: the README says to run `serve-trajectory` on the learner box and `colosseum train --set transport.mode=grpc` on worker boxes. `transport.mode` is unused (R5-14), so this runs a full local training and the trajectory server receives nothing. The working commands are `serve-weight-store`, then `run-learner --agent … --traj-port … --weight-store host:port`, then `run-workers --weight-store … --learner agent=host:port …`. They are undocumented in the README (only in docstrings and compose). There is no SSH or launch script, no cluster file, and workers crash at startup if the weight store isn't up yet (uncaught `RpcError` in `GRPCWeightSource.get_nowait` → `get_version`).
Evidence: READ.
Fix: document the three-command recipe. Add `colosseum launch --cluster cluster.yaml` (role→host map, spawns via SSH, or just prints the commands). Make the gRPC clients wait for readiness (`grpc.channel_ready_future(...).result(timeout)` with retry).

**R5-27 — low — security — `deployment/docker-compose.yaml:15-16`, `weight_store/grpc_store.py:86`, `transport/grpc_transport.py:72`**
What's wrong: gRPC servers bind `[::]` with insecure channels, and compose publishes 50051 on all host interfaces. Anyone on the LAN can `PutWeights` (weight poisoning) or send 64 MB messages. Remote code execution is mitigated by `weights_only=True` everywhere (good).
Fix: bind to a configurable host, don't publish ports in compose (the internal network is enough), and add an optional shared-token metadata interceptor or mTLS for cross-machine runs.

### Other bugs and DX

**R5-28 — medium — bug — `cli.py:94`, `launcher.py:285`, `cli.py:124`**
What's wrong: `colosseum bc` on a GPU saves `network.state_dict()` with CUDA tensors. `_resolve_resume_state` and `colosseum eval` call `torch.load(path, weights_only=True)` without `map_location`, so on a CPU-only box (the typical worker or eval machine in the owner's setup) loading fails with "Attempting to deserialize object on a CUDA device". The kickstart teacher load correctly uses `map_location`.
Evidence: READ.
Fix: save CPU tensors (`{k: v.cpu()}`) and always load with `map_location="cpu"`.

**R5-29 — medium — ux — `cli.py:99-140` (`eval`), `eval.py:83-122`**
What's wrong: the eval CLI has these problems:
- (a) It builds every agent from the single config's `networks`, so mixed architectures cannot be compared. The library supports `network_factories`, but the CLI doesn't expose it.
- (b) It has no scripted or random baseline.
- (c) Output is a printed table only, with no `--out results.json/csv` and no WandB.
- (d) `-a ttt.pt` without `name:` gives a raw `ValueError: not enough values to unpack` (RUN).
- (e) A single `-a` prints an empty table with no warning (RUN).
- (f) The `--num-matches` help says "Matches per agent pair", but it is the total match count.
Fix: let `-a` accept a checkpoint dir whose `meta.json` holds the architecture (R5-12), or `name:path:config.yaml`, plus `scripted:<class>`. Add `--out`, `--baseline random`, and a final "rank by ELO / mean WR" table.

**R5-30 — medium — ux/perf — heterogeneous machine sizing — `config.py:120-149`, `worker/rollout_worker.py` (no `torch.set_num_threads`)**
What's wrong: worker processes never call `torch.set_num_threads(1)`, so N workers each start a full intra-op thread pool. That oversubscribes CPU boxes and, under a k8s CPU limit, causes heavy throttling. `num_workers` is a fixed number (default 4) with no `auto` (= cpu_count) setting. Per-machine settings exist only as `--set` flags on a shared YAML.
Evidence: READ, and suspected from RUN: a 62 s single-agent tic-tac-toe run with 1 worker + 1 learner used 2 m 53 s of user CPU.
Fix: add `rollout.torch_threads: 1` (applied in the worker), `num_workers: auto`, and `learner.torch_threads`. Document "one YAML for the experiment, `--set rollout.num_workers=…` / `--device` per box", or add a per-role `machine:` section.

**R5-31 — low — ux — assorted**
- `agents: {alpha: null}` is rejected; you must write `{}` or `{networks: null, …}` (RUN).
- `run-learner --agent agnet_alpha` (typo) silently trains with the global config (RUN: `get_agent_config` of an unknown id returns the global config).
- Agent ids containing `:` are accepted (RUN) but break `Coordinator._base_agent` parsing.
- `examples/space_miners` needs `Box2D`, which is not declared in any extra and not in the Docker image (RUN: `ModuleNotFoundError: Box2D`).
- `ChaseEncoder.__init__` takes no `**kwargs` (copy-paste trap).
- There is no `.gitignore`.
- compose uses the obsolete `version:` key and builds the same Dockerfile three times under three image names (should be `image: colosseum:local` + one `build`).
- The learner Deployment uses an RWO PVC with the default RollingUpdate strategy, so a reschedule to another node deadlocks. Use `strategy: Recreate`. A second learner manifest copied as instructed also collides on `checkpoints-pvc`.

**R5-32 — medium — design — `coordinator/checkpoint_manager.py:114-121`**
What's wrong: checkpoint retention is FIFO only. There is no "keep best by ELO", "keep every Kth", or pinned milestone, so with `pool_size: 20` the strongest historical agent is eventually deleted from disk. There is no `colosseum leaderboard`/"select best" command.
Evidence: READ.
Impact: works against the owner's goal of selecting the best agents and building a final arena.
Fix: add `checkpoint.keep_every: N` (archive, never evicted, kept outside the opponent pool) and `keep_top_k_by_elo`, plus a `colosseum leaderboard --run runs/x` command that reads `ratings.json` and checkpoint metas.

---

## Metrics inventory

| Metric | Produced where | Logged where | Gaps |
|---|---|---|---|
| total_loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction | `appo.py:239-246`, averaged over minibatches in `train_step` | Local: learner → `metrics_queue` → monitor → WandB `"<agent>/<key>"` every `log_interval` steps. Distributed: **nowhere**. | Multi-agent WandB drops the lagging agents (R5-02); nothing without WandB (R5-06) |
| kickstart_loss, kickstart_lambda | `appo.py:247-249` (only if a teacher is set) | same as above | — |
| policy_version, learning_rate | `appo.py:317-318` | same | — |
| train_step, chunks_received | `learner.py:143-144` | same, plus learner stdout every 10 steps (**invisible** in local mode, R5-01) | no rates (steps/s, samples/s) |
| episode return / length per slot | `rollout_worker.py:620-663` → `MatchResult` | **not logged**; only fed to the coordinator (local mode); distributed: not produced | R5-03 |
| per-player outcome (win/draw/loss) | `core/outcomes.py`, worker | **not logged**; feeds ELO/WR in memory | — |
| ELO per agent | `coordinator.py:169-174` | **not logged, not persisted** | no ELO over time, nothing ranks agents |
| pairwise win-rate matrix | `ratings.WinRateTracker` | **not logged**; `get_ratings_summary()` unused | — |
| win rate vs historical checkpoints / vs self | — | — | same-agent pairs are deliberately skipped (`coordinator.py:162`), so "beats its own past" is **never measured** |
| checkpoint saved events | `launcher.py:553-554` | main-process log line only | not in WandB, metrics in `meta.json` empty |
| eval results | `eval.py` → `EvalMatrix.summary()` | stdout only | no JSON/CSV/WandB |
| BC loss | `offline_bc.py:147-148` | log line only (BC CLI sets up logging) | no WandB, no val split |
| grad norm | — (`clip_grad_norm_` return ignored, `appo.py:291/296`) | — | missing |
| explained variance, value/return stats, V-trace rho stats | — | — | missing |
| policy lag (learner version − behavior version) | data present in `TrajectoryChunk.behavior_policy_version` | — | missing |
| env SPS / per-worker throughput / inference latency | — | worker logs only "finished, steps=…" (invisible) | missing |
| learner queue depth, data-wait vs compute time, dropped chunks | — (gRPC drop is logged as a WARNING per chunk) | — | missing |
| weight-sync age on workers | — | — | missing |
| system / GPU | WandB system metrics of the **main** process only (local mode) | WandB | none for remote learners or workers |
| active workers / learners / agent pool view | — | — | no dashboard, no status command |

---

## Doc mismatches

**README.md**
1. Line 34, Quick Start `colosseum train --config configs/examples/tic_tac_toe.yaml` fails with `ModuleNotFoundError: examples` when run through the console script (R5-19). Only `python -m colosseum` or `PYTHONPATH=.` works.
2. Lines 416-436, the multi-machine recipe (`serve-trajectory` + `train --set transport.mode=grpc`) does not work. `transport.mode` and `grpc_port` are unused. The real commands `run-learner` and `run-workers` are not mentioned anywhere in the README.
3. Lines 553-559, the transport table presents `mode` and `grpc_port` as functional. They are dead.
4. Lines 218-224, the "kickstarting" section shows `resume_from`. Kickstarting is actually `training.kickstart_teacher` / `kickstart_lambda` / `kickstart_decay_steps`, none of which appear in the reference tables.
5. Lines 288 and 563-573, "Unset (null) fields inherit from the global config" / "override specific fields". Wrong: a partial section override resets the remaining fields to defaults (R5-08). Partial `networks` overrides are rejected.
6. Line 522, `phase: "bc"` is listed as a valid phase but has no effect.
7. Line 525, `resume_from` is described as "Checkpoint ID to resume from". It also accepts a `.pt` path, is global (not per agent), and is ignored by `run-learner`.
8. The rollout table (499-506) is missing `vec_env`, `subproc_workers` and `match_refresh_interval_sec`. The training table is missing `kickstart_*`.
9. Line 262, "ELO and pairwise win rates tracked automatically". They are tracked in memory but never output or persisted.
10. Line 450, "Worker pods auto-scale via HPA based on CPU utilization". This is technically what the manifest does, but it scales to max immediately and does not add useful throughput (R5-21).
11. Lines 7, 628 and 681: "148 tests", "16 test files". Actual: **174 tests collected** across **19** `test_*.py` files (RUN, `pytest --collect-only`).
12. Project structure (657-681) omits `distributed.py`, `envs/subproc_vec_env.py`, `networks/normalization.py`, `core/outcomes.py`, `examples/space_miners`, `configs/examples/space_miners.yaml` and `k8s/learner.yaml`. The test list omits `test_distributed`, `test_review_fixes`, `test_subproc_vec_env`, `test_milestone2` and the `run_*_test.py` scripts.
13. Undocumented features: `colosseum validate`, `run-learner`, `run-workers`, the `info["outcome"]` / `info["rank"]` outcome convention (`core/outcomes.py`), `NormalizeObs`, `rollout.vec_env: subprocess`.
14. `--num-matches` help text says "Matches per agent pair". It is the total number of matches.
15. Nothing says `examples/space_miners` needs `pip install Box2D`, and the Docker images do not include it.
16. k8s: the README omits building and pushing `colosseum-base:latest`, and the namespace ordering problem.

**CLAUDE.md**
1. "148 tests passing, 2 skipped". Actual: 174 collected.
2. CLI list: `train`, `bc`, `eval`, `serve-weight-store`, `serve-trajectory`. Missing `validate`, `run-learner` and `run-workers`.
3. Eval command `colosseum eval --agents A_ckpt_100 B_ckpt_200 --num_matches 1000 --env my_game`. The real syntax is `-c config.yaml -a name:path.pt … --num-matches`; there is no `--env`.
4. The Milestone 1 table says `LocalTransport` and the `InMemory`/`SharedMemoryWeightStore` are used by the pipeline. The launcher uses raw `mp.Queue`s. `LocalTransport` and `SharedMemoryWeightStore` are unused (dead code), and `InMemoryWeightStore` is used only inside the gRPC server.
5. "Single-machine fallback: shared_memory for weights (zero-copy), shared memory ring buffer for trajectories". Actual: pickled `mp.Queue` for both.
6. Deployment: "Weight Store as a StatefulSet with persistent volume". Actual: a Deployment, in-memory. "Coordinator as a single-replica Deployment": there is no coordinator service. "HPA based on queue depth": actually CPU.
7. Milestone 6 lists Docker images "Base, worker, weight-store" and k8s "Namespace, weight-store, worker + HPA". There is also `learner.yaml`. `Dockerfile.worker` runs `train`, not a worker role.
8. Metrics section: "Per-agent: reward curves, ELO over time… Win rate matrix… Dashboard". Only loss/entropy/KL/LR are logged. No reward curves, ELO or WR matrix, and no dashboard.
9. "Checkpoint = state_dict + optimizer_state + training_step + metrics snapshot". The metrics snapshot is always empty.
10. "Dynamic: add/remove agents and machines at any time without stopping". Agents are fixed at launch in both modes. Only distributed workers can be added.
11. "Mode B, Online BC / DAgger: scripted expert provides actions at inference time". Only kickstarting from a teacher network exists; there is no scripted expert path.
12. Agent pool lists `ScriptedAgent` / `FrozenCheckpoint` as usable. They are registered types only, with no config surface and no worker support.
13. Remaining open items are stale. #3 (distributed launcher) is partially done in `distributed.py`. #6 (SubprocessVectorEnv) is done. #11 (observation normalization) is done as opt-in `NormalizeObs`.
14. The file list omits `distributed.py`, `subproc_vec_env.py`, `normalization.py`, `outcomes.py`, `examples/space_miners` and `k8s/learner.yaml`.

---

## DX walkthrough (clone → first training on a new game)

1. `pip install -e .` downloads the CUDA torch build (GBs) even on CPU boxes. Nothing explains how to use the CPU index.
2. `colosseum train -c configs/examples/tic_tac_toe.yaml` fails (R5-19). The user has to work out `python -m colosseum` or `PYTHONPATH=.`.
3. Once it runs, the console shows only the banner, then silence for about a minute, then "All processes stopped" (R5-01, R5-06). The user cannot tell whether it learned anything.
4. For a new game you write a custom dict-keyed `BaseEnv` (not PettingZoo or Gymnasium multi-agent), three network classes with hardcoded input dims (R5-18), and a YAML file. `colosseum validate` catches latent-dim mismatches with a decent message (RUN), but not num_players, action-count or value-shape errors (R5-16), and not typos (R5-07).
5. A second experiment in the same directory may crash because of stale checkpoints (R5-13).

A realistic first good run takes about 1-2 hours of friction for a framework author. A competitor working under time pressure will hit R5-01, R5-07, R5-13 and R5-19 in their first session.

---

## Assessment

**What's good**
- The Pydantic schema is readable, has good `description=` texts and mostly sensible defaults. `ge`/`le` bounds exist on numeric fields.
- `colosseum validate` exists and gives actionable messages for latent-dim and recurrent-dim mismatches.
- `torch.load(weights_only=True)` is used everywhere.
- Algorithm selection via `algorithm_class` is a clean extension point.
- The distributed roles (`serve-weight-store` / `run-learner` / `run-workers`) are a reasonable, minimal decomposition. Compose wires them correctly, with healthchecks and `depends_on`.
- `core/outcomes.py` (the env-provided `outcome`/`rank` convention) and `NormalizeObs` (stats in buffers, so they sync with weights) are good designs.

**What's weak**
- Monitoring is the biggest gap relative to the owner's priorities. Local child processes are silent, multi-agent WandB loses data, no episode, win-rate or ELO data is ever surfaced, there is nothing in distributed mode, and there is no non-WandB sink.
- Config is permissive in the wrong places (typos accepted, overrides that replace whole sections) and too narrow where the owner needs it (agents can't be frozen or scripted, warm start is global, no run dirs).
- Deployment artifacts look complete but don't work in practice. Run-to-completion jobs are deployed as Deployments, HPA uses CPU, the single image is GPU-sized, configs and code are baked in, and the README recipe is wrong. The distributed mode has no coordinator, so league training is single-machine only.

**Recommended target design, in priority order (simplicity first)**

1. **Run directory and observability (1-2 days, highest value).** Each `train` / `run-learner` / `run-workers` gets a `run_id` (CLI `--run-id` or generated) and writes `runs/<run_id>/{config.resolved.yaml, metrics.jsonl, ratings.json, logs/<role>.log, checkpoints/}`.
   - A `setup_logging()` call goes into every process target.
   - A `MetricsSink` interface has an always-on JSONL sink plus optional WandB (one run per role/agent, `group=run_id`, `define_metric` per agent) and optional TensorBoard.
   - Log, per agent, everything in the inventory gaps: episode return and length, win rate vs. each opponent class (latest, historical, other agents, scripted), ELO, policy lag, grad norm, explained variance, SPS, and queue fill/drops.
   - Add `colosseum status runs/<id>`, which prints a pool/ELO/throughput table from these files. That is the dashboard with zero infrastructure.
2. **Config hardening (≈1 day).**
   - `extra="forbid"` on all models.
   - Literal or regex validation for device, amp_dtype and recurrent_type.
   - Deep-merge agent overrides.
   - One `apply_overrides()` using `yaml.safe_load` for values, applied before seeding.
   - Reject multiple agents with `phase: self_play`.
   - `validate` checks num_players, action count, value shape and env reset/step plus the mask.
   - Remove dead fields (`transport.mode`, `grpc_port`, `phase: bc`) or make them meaningful.
   - Add `extends:` (a single-level YAML merge) for experiment variants. This is a tiny feature that covers most "warm-start variant" configs.
3. **Agent-centric config (the owner's scenarios).**
   - `agents.<id>.type: trainable|frozen|scripted`.
   - Per-agent `init_from`, `kickstart`, `networks`, `algorithm`, `learner`, and `checkpoint` for frozen agents (or read the architecture from checkpoint `meta.json`).
   - `scripted.class` plus kwargs, with a `ScriptedPolicy` interface run by workers in the normal slot machinery.
   - Self-describing checkpoints (meta contains the networks section and env class), so `eval`/arena can mix architectures from paths alone.
   - Checkpoint retention with `keep_every` and `keep_top_k_by_elo`, plus `colosseum leaderboard`.
4. **Distributed = the same Launcher, split by role.** Expose the existing coordinator monitor loop as a small gRPC (or HTTP/JSON) service: `GetAssignment(worker_id)` returns a `WorkerCommand` plus checkpoint ids, `ReportResults(batch)`, and `Status`. Workers fetch checkpoint weights from a shared directory or the weight store by id. Then league training, PFSP, ELO and metrics work identically across machines.
   - Add `colosseum launch --cluster cluster.yaml`. The cluster file maps roles to hosts, e.g. `coordinator+weight_store: box0`, `learners: {agent_a: gpu1:cuda:0, agent_b: gpu1:cuda:1}`, `workers: {cpu1: 32, cpu2: 16}`. The command either SSH-starts the roles or prints the exact commands. This is the simple bare-metal path the owner asked for; k8s becomes optional.
   - gRPC clients wait for readiness and retry. The weight store carries an epoch so workers accept a learner restart. Learners auto-resume from the latest checkpoint in the run dir.
5. **Deployment simplification.**
   - One multi-stage Dockerfile with `cpu` and `cuda` targets, dependencies installed before `COPY src`, non-root user, `tini`, `.dockerignore`.
   - User images: `FROM colosseum:cpu` + `COPY my_game` + `pip install -e`. Configs mounted at `/config`.
   - k8s via kustomize: namespace first, ConfigMap generator for the run config, `image:` override.
   - Learners as StatefulSets or Jobs with `restartPolicy: OnFailure`, auto-resume, a GPU request and a nodeSelector. Workers as a Deployment with fixed replicas (drop the CPU HPA until a learner-queue metric exists). Coordinator and weight store as single Deployments/Services.
   - compose: one `build`/`image`, GPU reservation on the learner, `WANDB_API_KEY` passthrough, and no host-published ports by default.
6. **DX polish.**
   - Insert cwd plus `code_paths` into `sys.path` in the CLI.
   - `colosseum init my_game`, which scaffolds env, networks and YAML with dims wired from spaces.
   - Pass `obs_space`, `action_space` and `input_dim` into network constructors.
   - Per-component kwargs.
   - `torch.set_num_threads(1)` in workers and `num_workers: auto`.
   - Exit code ≠ 0 on failure.
   - Fix README and CLAUDE.md against the mismatch list above. Generate the config reference table from the Pydantic schema so it can't drift again.

Scratch artifacts used for verification are in `scratchpad/r5_config/`: `t_config.py`, `t_stale.py`, `badnet/`, `*.yaml`, and logs `sp1.log`, `sp2*.log`.

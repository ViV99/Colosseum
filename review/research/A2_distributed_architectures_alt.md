# Distributed RL architecture research for Colosseum (W1)

Date of research: 2026-10-07. Method: web search + primary-source fetches (arXiv PDFs converted locally with pdftotext, official docs).
Convention: **[V]** = verified in a source I actually read (URL given). **[I]** = my inference / engineering judgement, not measured. **[U]** = not verified in this session (from memory or only seen in a low-quality summary). No local benchmarks were run (the sandbox has no pip), so every throughput number below is from the cited papers, not from this machine.

NOTE ON OUTPUT PATH: the requested scratchpad path was reported as unavailable by the harness mid-task, so this file was written to the session directory instead.

---------------------------------------------------------------------------
## 0. One-paragraph answer

Colosseum's core design (IMPALA/APPO-style async, CPU workers with local inference, per-agent learners, pull-based weights) is well supported by the literature for **small/medium models and moderately expensive envs**, which is the bot-competition regime. The closest prior art is **TLeague/TStarBot-X (Tencent, 2020)** and **SRL (ICLR 2024)**: both are league-capable, Kubernetes- or socket-based, non-Ray, and both offer an *optional* GPU inference server next to inline CPU inference. The main things the evidence says Colosseum should change or add: (a) keep local CPU inference as default but add an optional batched remote inference mode for big models; (b) do not rely on Python gRPC *streaming* for the trajectory data plane (grpc.io says Python streaming RPCs are much slower than unary); consider ZeroMQ / raw framed TCP or asyncio gRPC and raw-buffer serialization; (c) make weight sync per-machine, version-checked and pull-based (RLBoost, Ape-X, IMPALA, SRL all pull); (d) track and bound policy lag explicitly in policy *versions*; (e) deploy as "stable core + stateless dial-out workers"; Kubernetes/Ray are optional, not required. Building on PufferLib/Sample Factory does not satisfy the owner's #1 priority (machine-level decoupling); RLlib can, but with Ray-cluster constraints and multi-agent/league friction. A custom framework is justified, provided it borrows from TLeague/SRL rather than reinventing.

---------------------------------------------------------------------------
## 1. Landscape of actor-learner systems and where inference runs

| System | Year | Inference placement | Transport / sync | Multi-machine | League / multi-agent | Source |
|---|---|---|---|---|---|---|
| IMPALA | 2018 | On actors (CPU); actors pull params at the start of every trajectory | Actor->learner trajectories; V-trace for lag | Yes (thousands of machines claimed) | Multi-task, not league | arXiv 1802.01561 [V] |
| Ape-X | 2018 | On actors (CPU); params pulled periodically; actors compute initial priorities | Batched comms to central replay; "actors and learners could run in different data-centers without limiting performance" | Yes | No | arXiv 1803.00933 [V] |
| OpenAI Five / Rapid | 2018-19 | **Central Forward-Pass GPUs** (batches ~60); rollout machines run only the game engine | Params in Redis "controller"; optimizers publish every 32 grad steps; rollouts send every 256 steps (~34 s of game); NCCL allreduce among up to 1536 optimizer GPUs | Yes (128k CPU, 256 GPU) | Self-play, 80% latest / 20% old | arXiv 1912.06680 [V] |
| AlphaStar | 2019 | TPU "actor tasks" do inference (16 TPU-v3x8 actor tasks and 16,000 concurrent matches per agent) | - | Yes | PFSP league | secondary summary of Nature paper [U-ish] |
| SEED RL | 2019 | **Central accelerator inference**; actors only step envs; streaming gRPC | gRPC streaming + batching; unix sockets when co-located | Yes | No | arXiv 1910.06591 [V] |
| Sample Factory v2 | 2020/2023 | **Local to one machine**: rollout workers + policy workers (GPU forward) + one learner per policy; shared-memory tensors, signals carry buffer IDs | Shared memory; param update on policy workers <1 ms | **Single machine by design** (paper: "optimized for a single-machine setting") | Multi-agent, self-play, PBT, multi-policy | arXiv 2006.11751, samplefactory.dev [V] |
| Podracer Anakin/Sebulba | 2021 | Anakin: env+agent on the accelerator; Sebulba: actor threads on TPU cores, inference on accelerator | TPU interconnect | TPU pods | No | arXiv 2104.06272 [V abstract only] |
| Cleanba | 2023 | Sebulba-style (JAX, EnvPool), inference on accelerator | Bounded queue (size 1) -> deterministic 1-step staleness | Multi-GPU | No | arXiv 2310.00036 [V] |
| moolib | 2022 | Peer-to-peer RPC library, batched inference possible | Custom RPC | Yes | No | Meta paper (not read in full) [U] |
| TLeague | 2020 | **Inline or optional GPU InfServer** per actor; ModelPool serves params | ZeroMQ RPC with Python-native messages; Horovod/NCCL among learners; k8s | Yes (hybrid CPU/GPU cluster) | **League (CSP-MARL), LeagueMgr, payoff matrix** | arXiv 2011.12895 [V] |
| TStarBot-X (on TLeague) | 2020 | GPU InfServer; "reduces 34% of CPU usage while maintaining the same data generation speed" | k8s on Tencent Cloud, 144 V100 + 13,440 CPU cores | Yes | League with exploiters | arXiv 2011.13729 [V] |
| SRL | ICLR 2024 | **Configurable**: inline CPU inference in actor workers OR separate GPU policy workers; dynamic batching | Sockets (remote), shared memory (local), LZ4; **NFS-based parameter server** | Yes, >15k CPU cores, 32 A100 | Multi-agent/PBT, hide-and-seek | arXiv 2306.16688 [V] |
| MALib | JMLR 2023 | Actor-Evaluator-Learner; centralized task dispatcher for heterogeneous policy combinations | Ray-based | Yes | **Population-based MARL (PSRO etc.)** | arXiv 2106.07551 [V via search summary] |
| DI-engine / DI-star | 2021- | Collector/Learner/Coordinator/League modules; DI-orchestrator k8s operator | - | Yes | League (AlphaStar-style) | DI-engine docs [V via search] |
| Acme + Launchpad + Reverb | 2020- | Actor-Learner(-InferenceServer) building blocks; Launchpad defines topology of nodes | Reverb replay service | Yes | Not league | DeepMind blog, TLeague related-work [V] |
| RLlib (Ray >= 2.40 default new API stack) | 2024- | EnvRunners (remote actors, CPU) do inference locally; LearnerGroup (GPU or CPU) trains | Ray object store; `broadcast_interval` etc. | Yes (Ray cluster) | Multi-agent via MultiRLModule | Ray docs [V] |
| TorchRL | 2023- | Collectors (Ray / RPC / torch.distributed); **new InferenceServer (0.13) with auto-batching and Threading/Slot/MP/Ray/Monarch transports + WeightSyncScheme** | - | Yes | No league | TorchRL docs [V] |
| PufferLib | 2024-26 | See section 7 | - | **No machine-level distribution found** | Native multi-agent emulation | puffer.ai/docs [V] |

Key point: **every large system that serves complex models (OpenAI Five, AlphaStar, TStarBot-X, SEED) moved inference to accelerators; every system targeting cheap envs with small nets (IMPALA, Ape-X, Sample Factory, Cleanba-IMPALA, SRL inline mode) keeps it on CPU.** TLeague and SRL expose both.

### LLM-RL systems (transferable patterns only)
- Colocated vs disaggregated: veRL (HybridFlow) supports colocation with resharding; OpenRLHF/NeMo-RL separate rollout and training pools. Disaggregation gives independent scaling, elastic rollout, fault tolerance and heterogeneous hardware, at the cost of weight transfer (search summary of several sources; [V-weak]).
- **AReaL** (NeurIPS 2025): fully async, rollout never waits; up to 2.77x speedup vs sync with matched accuracy; staleness cap eta (4 for code, 8 for math) and a decoupled PPO objective (behavior vs proximal policy). Without the decoupled objective performance collapses at eta > 1; with it eta <= 8 has minimal impact. arXiv 2505.24298 [V].
- **RLBoost** (2025, NSDI'26): fixed on-demand trainer + elastic pool of preemptible rollout GPUs; **pull-based weight transfer** (trainer copies weights to a CPU buffer, rollout instances pull peer-to-peer; no RDMA, TCP over 50 Gbps vNIC); partial-work migration on preemption; 1.51-1.97x throughput and 28-49% better cost efficiency vs on-demand only. arXiv 2510.19225 [V].
- Weight transfer scale: NCCL broadcast for a 1T-param model took ~53 s vs ~7.2 s with P2P RDMA; LMSYS lists NCCL's weaknesses as redundancy, idle ranks, and **rigid fixed communication groups that prevent dynamic scaling**. lmsys.org/blog/2026-04-29-p2p-update [V]. Relevance: NCCL-style collectives are the wrong tool for elastic, heterogeneous CPU workers.

---------------------------------------------------------------------------
## 2. Local CPU inference vs central GPU inference vs hybrid

### Evidence table (all [V])
| Evidence | Numbers | Source |
|---|---|---|
| SEED cost vs IMPALA, DMLab, 1B frames | IMPALA $90 (default net) / $128 / $236 (large) vs SEED $25 / $35 / $54; cost ratio 3.6-4.4x. IMPALA's cost grows with net size because CPU inference dominates; SEED's inference ~30% of its cost | SEED paper Table 4 |
| SEED cost, Google Football (heavier env) | IMPALA $553/$681/$899 vs SEED $345/$365/$369; ratio only 1.68-2.7x. "As the env is more expensive... training and inference costs have relatively smaller impact" | SEED Table 5 |
| SEED is NOT always faster on same hardware | DMLab, 1 P100: IMPALA 30K fps vs SEED 19K fps (0.63x). ALE, 1 V100: R2D2 85K vs SEED 67K (0.79x). Gains appear with TPUs (2.5x at 2 cores, 80x at 64 cores = 2.4M fps) | SEED Table 1 |
| SEED latency | End-to-end inference latency IMPALA 18-49 ms vs SEED 11-15 ms (DMLab, default->large net); Football 12.6-34 vs 6.5-11 ms | SEED Table 13 |
| SEED actor design | CPU sits idle waiting for inference, so each SEED actor runs 12-16 envs on a 4-CPU machine to recover utilization | SEED Sec 4.4.1 |
| SEED bandwidth example | 100k obs/s of 96x72x3 bytes + traj 20 + 30 MB model: IMPALA-style needs 148 GB/s total (mostly params), obs transfer is 2 GB/s | SEED footnote 1 |
| SRL, single machine | Atari: SRL 124K fps vs Sample Factory 96K vs SeedRL 27K; DMLab 46K vs 42K vs 14K | SRL Table 6 |
| SRL, distributed (32 trainers), Atari | SEED-style (central inference) 169K fps; IMPALA-style (inline CPU inference) 477K; SRL full decoupling 453K. Configs: 480 actor workers per trainer with CPU inference, 120 with GPU inference (env ring size 20) | SRL Table 7 |
| SRL vs OpenAI Rapid on hide-and-seek | 3x faster with CPU inference, 5x with GPU inference | SRL abstract/results |
| TStarBot-X | GPU InfServer cuts CPU usage 34% at same data rate; 96 of 144 GPUs train, 48 do inference | arXiv 2011.13729 |
| OpenAI Five cost split | ~30% optimizer GPUs, ~30% forward-pass GPUs, ~30% rollout CPUs, ~10% overhead (huge LSTM) | arXiv 1912.06680 App. |
| Sample Factory masks latency locally | Double-buffered sampling: while one half of an env vector steps, the policy worker computes actions for the other half; "completely mask the communication overhead" when tuned | SF paper Sec 3 |
| TorchRL doc note | multiprocessed collectors have lower IO overhead; parallel envs execute policies faster through vectorization | TorchRL distributed collectors docs |

### When each wins [I, grounded in the table]
- **Local CPU inference (current Colosseum design) wins when**: model <= ~10M params (MLP/small CNN/GRU), env step time is comparable to or larger than a batch-of-envs forward pass, machines are heterogeneous or remote (spot, WAN, home LAN), and you want trivial elasticity. SRL's Atari result (CPU inline 477K vs SEED-style 169K) and SEED's own "0.63x/0.79x on GPUs" result support this. Per-step RPC to a remote GPU has a latency floor (SEED: 6-15 ms on datacenter network + TPU) that is only hidden by running 12-20 envs per actor; over typical office/cloud links (RTT 1-50 ms) it would dominate cheap-env stepping.
- **Central GPU inference wins when**: forward cost on CPU dominates (big CNN/ResNet/transformer/large LSTM; OpenAI Five, AlphaStar, TStarBot-X), envs are cheap, CPUs are expensive/scarce relative to GPUs, and the GPU box is on the same LAN (>=10 GbE, sub-ms RTT) or the same machine (unix sockets). Gains: SEED 3.6-4.4x cost ratio for DMLab, TStarBot-X -34% CPU.
- **Hybrid** (TLeague, SRL): make inference placement a per-agent/per-machine config: `inference: local | server`. Co-locate an inference server on the learner machine or on a LAN-local GPU box; workers on remote spot instances stay local-CPU.
- Colosseum-specific twist [I]: league training has several agents with different networks. Workers already group inference by (agent_id, network_id); a remote server mode would multiply the number of servers (one per trainable agent or one multi-model server) and frozen-opponent inference could be served from the same server. Local CPU inference for frozen opponents is cheap but multiplies CPU load, so batched inference for opponents is the first place a server pays off.
- JAX/GPU environments change the picture: Lux AI Season 3 ships a **GPU-parallelized JAX environment** designed for "fast training/evaluation on a single GPU" (neurips.cc/virtual/2024/competition/84798 [V]). For such envs Anakin-style (env + policy + learner on one accelerator) beats any CPU-actor design. Colosseum should be able to run a "colocated GPU env" worker on the learner box [I].

---------------------------------------------------------------------------
## 3. Weight synchronization

### Observed practices [V]
| System | Pull/push | Frequency | Store |
|---|---|---|---|
| IMPALA | Pull | At the beginning of each trajectory (unroll) | Learner |
| Ape-X | Pull | Periodically (actors in separate datacenters possible) | Learner |
| Sample Factory | Shared memory; policy workers update immediately | Param update <1 ms; avg policy lag 5-10 SGD steps in all experiments | Shared memory |
| OpenAI Five | Pull by forward-pass GPUs from Redis controller | Optimizers publish every 32 gradient steps; ~1 version/min; targeted staleness 0-1 | Redis |
| SRL | Pull | - | **NFS** parameter server, scaled to 15k cores |
| TLeague | Pull from ModelPool (in-memory, up to M replicas load balanced, random replica per request) | At each episode/period | ModelPool service |
| RLlib IMPALA/APPO | Push-ish via `broadcast_interval` (training-step calls between broadcasts); "doesn't always sync back the weights right after a new model version" | configurable | Ray |
| RLBoost | **Pull**, p2p from CPU buffer; designed so new instances can join without blocking others | per training step | transfer agents |
| vLLM/LLM RL | NCCL broadcast (fixed groups) or P2P RDMA | 53 s -> 7.2 s for 1T params | - |

### Cost model for Colosseum [I - arithmetic, not measured]
- Typical bot-competition policy: 0.25M-12M params -> 1-50 MB fp32.
- Per-machine pull of 10 MB over 1 GbE (~110 MB/s) ~ 90 ms; 50 MB ~ 0.45 s; over 10 GbE ~ 10-50 ms.
- Aggregate load on the weight server = (#machines x size) / interval, if **one puller per machine** fans out to local processes via shared memory. 20 machines x 50 MB / 5 s = 200 MB/s (1.6 Gbit/s). If every worker *process* pulled independently (e.g., 20 machines x 16 procs) it would be 16x that. => implement a per-machine weight daemon.
- Cheap tricks: version integer check before download (HEAD-like call); bf16/fp16 cast on the wire (2x); only agents actually scheduled on that machine; push-interval decoupled from checkpoint interval; zstd/lz4 gives little on float weights [U, expectation: near 1.0-1.1x]; delta weights not worth it for small nets [I].
- Frozen opponents/checkpoints are immutable: cache by (agent, version) and never re-pull.
- A shared filesystem (SRL's NFS) or an HTTP object endpoint is a legitimate "simplest thing" for the weight store and gives free multi-reader semantics [SRL, V].
- Avoid NCCL/collective broadcast for CPU workers: fixed groups, lock-step, hang on a slow receiver (LMSYS) [V].
- Reproducibility option (Cleanba): bounded 1-step staleness through a size-1 queue gives deterministic data composition [V].

---------------------------------------------------------------------------
## 4. Trajectory transport and serialization

### What the literature actually reports [V]
- Sample Factory: at >1 GB/s of experience even the fastest serialization would hinder throughput, hence shared-memory buffers + signals carrying IDs. => on one machine never serialize.
- SEED: streaming gRPC (C++), keep connection open and send metadata once, batching module; unix domain sockets when co-located.
- SRL: sockets for remote, shared memory for local, LZ4 on image data.
- TLeague: ZeroMQ RPC with Python-native messages; one DataServer + ReplayMem embedded in each Learner; actors associated to a specific learner (MA x ML x MG topology).
- OpenAI Five: rollouts push every 256 steps (~once/min each); Redis for params and metadata; data directly to optimizer experience buffers.
- Ape-X: batch all communications with replay, trade latency for throughput.
- **grpc.io performance guide: "Python... streaming RPCs create extra threads, making them much slower than unary RPCs"; asyncio API performs better; reuse channels; streams cannot be load balanced once started.** (grpc.io/docs/guides/performance) [V]. This directly questions Colosseum's plan of persistent gRPC streams for the data plane (open item #7) when using the sync Python API.

### Comparison [mostly I; no measured head-to-head benchmark found]
| Option | Pros | Cons | Verdict for Colosseum |
|---|---|---|---|
| gRPC unary (bytes field) | Simple, typed, deadlines, TLS, LB, good tooling | Default 4 MB message cap [U]; HTTP/2 + protobuf copy overhead for large bytes; chunks (~300 KB) are fine | OK for control plane and chunks of 0.1-4 MB |
| gRPC streaming (Python sync) | Fewer handshakes | Extra threads, "much slower" per grpc.io | Avoid with sync API; test asyncio |
| ZeroMQ PUSH/PULL (pyzmq) | Brokerless, built-in fan-in/queues, multipart zero-copy frames, reconnect, used by TLeague | No built-in auth/typing; you own framing | Strong candidate for data plane |
| Raw framed TCP / asyncio streams | Max control, fewest copies | You own reconnect/backpressure | Fine, but ZMQ already does it |
| Ray object store | Zero-copy on node, spill | Requires Ray cluster | N/A (no Ray) |
| Redis (strings/streams) | Easy, durable, OpenAI Five used it for params/metadata | Single-threaded, memory-bound; big values hurt; 512 MB value cap [U] | Good for metadata/registry, not bulk trajectories |
| NATS/JetStream | Great for control messaging | Default max payload ~1 MB [U]; JetStream ~50k msg/s per search summary [V-weak] | Control plane only |
| Shared memory ring buffer | Zero copy | Single machine | Keep for local mode (already planned) |

Search-summary throughput numbers (Redis Streams ~100k msg/s, NATS JetStream ~50k msg/s, ZeroMQ millions msg/s for small messages) are **small-message** figures and don't transfer to 300 KB chunks [V-weak]. At 300 KB/chunk the bottleneck is bytes/s (NIC, memcpy, compression CPU), not msg/s.

### Serialization [V unless marked]
- `torch.save` goes through pickle + zip container: avoid on the hot path; also a security concern (hence `weights_only`) [I/U].
- Pickle protocol 5 (PEP 574) exists specifically to avoid copies of large arrays via out-of-band buffers [V]. Best practice: a tiny header (msgpack or protobuf: agent_id, version, shapes, dtypes, lag info) + raw contiguous numpy/torch bytes as separate frames (ZMQ multipart or gRPC bytes), reconstruct with `np.frombuffer`/`torch.frombuffer` [I].
- safetensors: zero-copy, safe (no pickle), simple header + raw bytes; good for weight payloads and checkpoints [V: HF docs].
- Compression (Silesia corpus, i7-9700K): lz4 1.10: ratio 2.10, 675 MB/s compress, 3850 MB/s decompress; zstd -1: ratio 2.90, 510 MB/s, 1550 MB/s; zstd --fast=4: 2.15, 665/2050 [V: github.com/facebook/zstd]. On a 1 GbE link (110 MB/s) both are far faster than the wire, so compression is net positive only if the data compresses (grid/one-hot/sparse observations: yes; float activations/weights: little) [I]. Make compression per-field and adaptive (skip if ratio < ~1.2) [I]. On 10 GbE+ it may be net negative.
- Arrow/flatbuffers: no evidence they beat "header + raw buffer" for dense tensors; skip [I].
- Colosseum's own sizing: ~300 KB per chunk; 100 workers x 10 chunks/s = 300 MB/s. That saturates 1 GbE ~3x and needs >= 10 GbE or compression/obs-dtype shrinking (uint8/bool observations; store only what the learner needs; do not send both obs and next_obs) [I from CLAUDE.md].
- League fan-out: an arena match with k trainable agents sends k chunk streams (one per learner), multiplying bandwidth by k [I].

---------------------------------------------------------------------------
## 5. Policy lag

### Numbers [V]
| Source | Finding |
|---|---|
| Sample Factory paper | Lag defined in policy versions; lag in experiments averaged **5-10 SGD steps**, "stable training"; reduce lag by smaller rollout T or larger minibatch; recommends larger batch with many cores. Docs: `max_policy_lag` default 1000 (effectively off), `async_rl` default True, `rollout` default 32 |
| OpenAI Five | Staleness = M - N versions. **~8 versions of extra staleness caused significant slowdowns** (a few minutes in a multi-month run). Early version sending whole episodes -> data hours old, "thousands of gradient steps", gradients "often useless or destructive". Final: send every 256 steps, params ~every minute, target staleness 0-1 |
| AReaL | Without decoupled PPO objective: collapse for eta > 1. With decoupled objective: eta <= 8 minimal impact; eta=4 code / 8 math suggested; unbounded eta worse than eta=0 |
| IMPALA paper | V-trace vs 1-step IS: nearly the same when lag is negligible; V-trace better as lag grows (e.g., with replay); lag "can be several updates" |
| IMPACT (ICLR 2020) | PPO-style loss + target network + circular buffer + truncated IS: higher reward and up to 30% less wall time than IMPALA on discrete tasks |
| PPO-EWMA / batch-size invariance (NeurIPS 2022) | Decoupling proximal policy from behavior policy makes PPO batch-size invariant and more tolerant of stale data |
| Cleanba | Deterministic 1-step staleness (learner trains on the second-latest policy's data) matches sync PPO quality, parallelizes actor and learner, reproducible across 1 vs 8 GPUs; moolib IMPALA curves vary with hardware speed |
| SRL | Data reuse configs (PPO: 50 bootstrap steps, 5x data reuse) |
| TStarBot-X | Each sample used ~5 times (consume 210 fps vs receive 43 fps per GPU) |

### Recommendations [I]
- Measure lag in **learner policy versions** (version at learner minus version stamped on chunk), per chunk, per agent; log mean/p95/max.
- Targets: mean <= ~5-10 (SF), p95 <= ~16, hard drop above ~32 for APPO with V-trace. Plain PPO (no V-trace) should be held to <= ~2-4. These are heuristics synthesized from SF (5-10 OK), Five (8 hurts plain PPO) and AReaL (eta<=8 OK only with decoupled objective).
- Chunk length T vs sync interval: a chunk of T steps is generated under up to T/steps_per_pull policy versions if workers pull mid-chunk. Pull weights **at chunk boundaries** and stamp version in the chunk; also record the behavior log-probs (already done) so V-trace/decoupled-PPO is exact.
- Learners that update fast (small nets, GPU) produce more versions/s than workers can follow: lag grows with (learner step rate x (chunk time + transport + pull interval)). With remote spot workers, worker->learner delay is the dominant term. Add a `max_version_lag` filter and a "slow-worker throttle" (learner tells coordinator to reduce push frequency, or learner batch >= k chunks).
- Optional Cleanba-style "bounded lag mode" for debugging/reproducibility.
- Consider PPO-EWMA/decoupled proximal policy (AReaL style) as the loss variant when lag is large, instead of vanilla clipped ratio vs behavior policy [I].

---------------------------------------------------------------------------
## 6. Elasticity, fault tolerance, deployment for a small team

### Facts [V]
- RLlib: `restart_failed_env_runners=True` restarts an EnvRunner as an identical copy, "set this to True when training on SPOT instances"; `ignore_env_runner_failures` continues with the rest (docs.ray.io fault_tolerance config). New-stack docs call EnvRunners "fully fault tolerant" and support CPU runners + GPU learners.
- Ray on-prem: nodes must share a network; ports 6379; with NAT the address printed by the head won't work from outside the subnet; head node must not be spot (Ray community docs); cluster launcher uses SSH (docs.ray.io on-premises).
- RLBoost: rollout preemption handled by migrating partial work; trainer on reserved instances.
- SkyPilot managed jobs: spot "70-90% cheaper", auto recovery from preemption/node failure, needs checkpointing to persistent storage, works over clouds, Kubernetes and **SSH node pools** (docs.skypilot.ai managed-jobs).
- TLeague: all modules as k8s resources; example: "56 Learners and 8 InfServers, each Learner corresponds to 16 actors, 1 GPU per learner, 4 CPU cores per actor; every 7 Learners and 1 InfServer co-located on one GPU machine"; ModelPool replicated for load.
- OpenAI Five: Redis controller stores all metadata so runs can stop/restart.
- DI-engine ships DI-orchestrator (k8s operator) + coordinator for similar roles.

### Simplest robust pattern for a few heterogeneous machines [I]
1. **Stable core** (one on-demand/owned box, ideally the GPU box): coordinator (agent pool, matchmaker, ratings), weight store, per-agent learners + their trajectory receivers, checkpoint dir, metrics. State is checkpoints + a small registry file/SQLite; everything else is reconstructible. Learner restarts from last checkpoint (OpenAI Five/TLeague pattern).
2. **Stateless workers dial out**: `colosseum worker --coordinator host:port --token X` (in a container or a plain venv). On start: register, receive env/agent spec (git SHA or image tag for env code), pull weights, produce chunks, heartbeat. Outbound-only connections avoid NAT problems that Ray has; works over WireGuard/Tailscale-style overlay if machines are on different networks [Tailscale: U, not researched]. Scale = start/stop processes; no scheduler needed.
3. Worker leases: coordinator hands out match configs with a lease/TTL; crashed worker's matches simply expire; chunks are independent so losing in-flight data is harmless (on-policy-ish data is cheap to regenerate). Make results (win/loss) idempotent by match_id.
4. **Backpressure instead of queues**: learner receiver has bounded buffer; when full, it tells workers to slow or drops oldest chunks (freshest data matters more; Five/AReaL results). Workers must tolerate learner unavailability (retry with backoff) so that learner restarts do not kill the fleet.
5. Spot nodes: run **workers only** on spot; never learners/coordinator. Use SkyPilot (clouds, k8s, SSH node pools) *or* a simple docker-compose/systemd + restart policy; use k8s only if the team already operates it. HPA on queue depth is a nice-to-have, not a requirement; on a handful of machines "docker run" with `--restart=always` is the least moving parts.
6. Version skew: stamp every message with protocol/env/agent-config hash; coordinator refuses workers with mismatching env version (competition envs change mid-contest).
7. Observability: per-machine heartbeat with CPU util, env steps/s, chunk send latency, lag stats; a dashboard of "chunks received/s per agent" is the one signal that matters.

---------------------------------------------------------------------------
## 7. Build vs buy

### Per-framework findings
**PufferLib** (repo main = "5.0"; PyPI latest = 3.0.0 released 2025-06-23)
- Docs (puffer.ai/docs) [V]: "Up to 60,000,000 step/second training in only ~10k lines of CUDA C"; environments (Ocean) written in C; FAQ: "Where did all the Python/third-party stuff go? It was all 100x+ slower than PufferLib is now."; "As of 5.0, we have a solid --cpu eval mode but no CPU training option"; supports "synchronous and 1-epoch asynchronous training as in CleanBa"; multi-GPU hyperparameter sweeps (`--sweep.gpus=4`); multi-agent mentioned only as "massively multiagent sims" in C envs; default MinGRU recurrent net.
- Blog titles [V]: 2.0 = 1M sps, 3.0 = "Better RL at 4M sps", 4.0 posts exist; 3.0 on PyPI still has Python vectorization backends (Serial/Multiprocessing/Ray) and Gymnasium/PettingZoo emulation (GitHub 3.0 tag fetch + PyPI; summary-level [V-weak]).
- Neural MMO 2023 competition: baseline trained with "Clean PuffeRL"; top solution reached 4x baseline within 8 h on a single RTX 4090 (arXiv 2311.03736 via search [V-weak]).
- Gaps vs Colosseum: no actor/learner machine split, no remote workers, no league/PFSP/self-play framework found, no heterogeneous learners per agent; v5 needs NVIDIA GPU to train and envs in C (bot competitions ship Python/JS/Kaggle envs); strengths: fastest single-box throughput, good vectorization/multi-agent emulation (v3). Verdict: **cannot satisfy priority #1**; useful as inspiration for vectorization and as a possible single-box backend for very fast envs.

**Sample Factory v2**
- [V] Async APPO, rollout workers + policy workers + learner per policy, shared memory, multi-agent, self-play, PBT, multi-policy on one or more GPUs, serial mode, HF hub integration; paper: single-machine focus; knobs: `policy_workers_per_policy`, `worker_num_splits`, `max_policy_lag`.
- Conflicting: one GitHub-summary fetch claimed "distributed/multi-node training" but paper/docs and other sources describe single machine; a third-party fork ("multi-sample-factory") exists for clusters [U]. Treat as single-node.
- Gaps: no remote workers (shared-memory IPC), league/PFSP matchmaking and scripted opponents need custom code (self-play via policy sampling per agent exists), per-agent different architectures/algorithms only within its policy abstraction. Verdict: excellent reference design for the single-machine mode (double-buffered sampling, shared memory), not a distributed solution.

**RLlib (Ray >= 2.40 new API stack)**
- [V] EnvRunners (remote, fault tolerant), LearnerGroup (GPU or CPU, multi-learner DDP), CPU runners + GPU learners, async IMPALA/APPO (circular buffer, `learner_queue_size`, `broadcast_interval`), multi-agent with MultiRLModule, `algorithm_config_overrides_per_module`, `policies_to_train` allows scripted/non-learning policies, league example exists (older AlphaStar-style `self_play_league_based_with_open_spiel`).
- Documented limitation: "multi-agent setups are not vectorizable yet" (one multi-agent env per EnvRunner) [V, docs], hurting throughput for cheap envs.
- One Algorithm/LearnerGroup trains all trainable modules; truly independent learners with different algorithms and cadences per agent, plus arena matches feeding several learners, are not first-class [I from docs; per-module overrides exist but "single algorithm class" per config].
- SRL reports up to 21x higher throughput vs RLlib in distributed setting (SRL's own benchmark; old RLlib version) [V, caveat], MALib 5x vs RLlib on 32 cores [V-weak].
- Ops: Ray cluster must share a network and has NAT/head-node constraints; autoscaler is cloud-centric; fault tolerance for EnvRunners is good.
- Verdict: the only "buy" option that meets machine-decoupling; costs: Ray as a mandatory runtime, multi-agent vectorization gap, league/PFSP/ELO coordinator still custom, heavy API surface that changes between releases (new API stack migration in 2.10-2.40) [I].

**TorchRL**
- [V] Distributed collectors (Ray, RPC, torch.distributed backends; sync/async), WeightSyncScheme, replay buffers, and in 0.13 an InferenceServer with auto-batching and Threading/Slot/MP/Ray/Monarch transports; AsyncBatchedCollector.
- Gaps: building blocks only; no league/matchmaking/ELO, no multi-agent orchestration across heterogeneous agents, no deployment story beyond Ray/submitit; fast-moving API (DistributedDataCollector -> DistributedCollector deprecations) [V]. Verdict: could supply inference-server ideas or components, not the system.

**Closest prior art (not mainstream "buy" options, but worth reading/borrowing from)**
- **TLeague** (open source 2020): league manager, ModelPool, per-learner DataServer, InfServer, ZeroMQ, k8s, hybrid CPU/GPU cluster. Throughput table: ViZDoom 1,152 CPU cores + 32 GPUs: 6.0K receiving fps / 8.2K consuming fps; Pommerman 100 cores + 2 GPUs: 2.9K / 20K. Probably TensorFlow-era, maintenance unknown [U].
- **SRL** (open source: github.com/openpsi-project/srl): dataflow abstraction, inline vs GPU inference, NFS param server, multi-agent; academic-grade.
- **MALib**: population/PSRO, Ray-based, centralized task dispatcher.
- **DI-engine/DI-star**: league + coordinator + k8s operator; broad but heavy.

### Gap summary that justifies a custom framework
| Requirement | PufferLib | Sample Factory | RLlib | TorchRL | TLeague/SRL/MALib |
|---|---|---|---|---|---|
| CPU boxes + GPU box on different machines | No | No (single machine) | Yes (Ray) | Yes (Ray/RPC) | Yes |
| Add/remove workers without restart | No | No | Yes (Ray autoscale/restart) | partial | yes/yes/Ray |
| No Ray/heavy runtime | yes | yes | **No** | optional | yes (TLeague, SRL) |
| League/PFSP + ELO + frozen/scripted agents | No | partial (self-play, PBT) | custom example | No | TLeague/MALib yes |
| One learner per agent, different arch/algo | n/a | partial | limited (one algo config) | DIY | TLeague yes |
| Python envs (Kaggle/Lux) first-class | v3 only; v5 C-only | yes | yes | yes | yes |
| Active maintenance (2025-26) | very active | active (push 2026-01) | active | active | academic/unclear |

Conclusion [I]: a custom framework is justified **if** the owner's #1 priority is truly heterogeneous, dynamic, multi-machine league training with minimal ops. It is *not* justified by throughput alone (PufferLib/Sample Factory are faster on one box), nor by missing algorithms. To limit risk: keep the single-machine path competitive with Sample Factory's design (shared memory, double-buffered sampling), and borrow TLeague/SRL's multi-machine decisions.

---------------------------------------------------------------------------
## 8. Concrete recommendations for Colosseum (ranked)

1. **Keep local CPU inference as default; add `inference: server` as a pluggable mode** (TLeague InfServer, SRL policy workers, TorchRL InferenceServer). Trigger when profiled forward time on worker CPU > ~30-50% of per-step time or model > ~10-20M params. Evidence: SEED Tables 4-5, SRL Table 7, TStarBot-X 34% CPU saving. [I for the threshold]
2. **Data plane: stop at "unary/async + raw buffers", benchmark before building persistent streams.** grpc.io says Python streaming is much slower than unary; use asyncio gRPC or ZeroMQ PUSH/PULL (TLeague precedent) for chunks; keep gRPC for control plane. Replace `torch.save` with header + raw buffers; compress per-field adaptively (lz4). First task: a 1-day benchmark harness comparing (unary gRPC, asyncio gRPC stream, ZMQ multipart, raw TCP) at 300 KB x {1,10,50} chunks/s from {1,8,32} senders over loopback and over the real LAN.
3. **Weights: per-machine puller + shared-memory fan-out; version-gated pulls; cached immutable checkpoints; optionally NFS/HTTP-backed store** (SRL used NFS to 15k cores). Pull at chunk boundaries; stamp versions on chunks.
4. **Policy-lag instrumentation and control**: version-based lag metric, `max_version_lag` drop, bounded buffers, dashboards; evaluate decoupled-PPO/PPO-EWMA objective for high-lag remote workers; offer Cleanba-style bounded-lag mode.
5. **Deployment = stable core + dial-out stateless workers** with leases, heartbeats, backoff, env/protocol version stamping; learners checkpoint to disk for restart. Document three recipes: single box, docker-compose/SSH fleet, k8s (optional HPA). Use SkyPilot or plain restart policies for spot **workers only**.
6. **Support colocated GPU-env workers** (JAX envs like Lux S3) as a first-class worker type on the learner box (Anakin-style).
7. **League fan-out accounting**: arena matches with k trainable agents create k chunk streams; route chunks directly to each learner's receiver (TLeague pattern), not through the coordinator.
8. Don't adopt PufferLib 5/Sample Factory/RLlib as the base; do reuse ideas: SF double-buffered sampling + serial debug mode; PufferLib emulation/vectorization and sweep culture (Protein); RLlib per-module overrides and fault-tolerant restart semantics.

---------------------------------------------------------------------------
## 9. Gaps, caveats, unverified items
- No local benchmarks; no authoritative head-to-head gRPC vs ZeroMQ vs raw TCP benchmark for 100 KB-4 MB messages was found. The comparison in section 4 is reasoning, not measurement.
- Default gRPC 4 MB message limit, NATS 1 MB payload default and Redis 512 MB value limit are from memory [U].
- AlphaStar numbers came from a search summary of the Nature paper, not the paper itself.
- PufferLib 5.0 self-play/league support was not found in docs (absence of evidence, not proof). PyPI vs main-branch version mismatch (3.0.0 vs 5.0) should be rechecked before any decision.
- Sample Factory multi-node: conflicting signals (paper says single machine; one summarizer claimed multi-node). Check the repo/issues if this matters.
- RLlib limitations (multi-agent vectorization, single algorithm per config) come from current docs and a search snippet; behaviour changes across Ray releases.
- TLeague/SRL/DI-engine/MALib maintenance status not verified.
- Some sources are from 2026 (vLLM/LMSYS blogs) and were read via fetch summaries.

---------------------------------------------------------------------------
## 10. Source list
- IMPALA: https://arxiv.org/abs/1802.01561
- Ape-X: https://arxiv.org/abs/1803.00933
- SEED RL: https://arxiv.org/abs/1910.06591
- Sample Factory paper: https://arxiv.org/abs/2006.11751 ; docs https://www.samplefactory.dev/ , https://www.samplefactory.dev/06-architecture/overview/ , https://www.samplefactory.dev/02-configuration/cfg-params/
- Podracer (Anakin/Sebulba): https://arxiv.org/abs/2104.06272
- Cleanba: https://arxiv.org/abs/2310.00036
- SRL: https://arxiv.org/abs/2306.16688
- OpenAI Five: https://arxiv.org/abs/1912.06680
- TLeague: https://arxiv.org/abs/2011.12895 ; TStarBot-X: https://arxiv.org/abs/2011.13729
- MALib: https://arxiv.org/abs/2106.07551
- DI-engine league docs: https://di-engine-test.readthedocs.io/en/latest/feature/league_overview_en.html
- IMPACT: https://arxiv.org/abs/1912.00167 ; PPO-EWMA: https://arxiv.org/abs/2110.00641
- AReaL: https://arxiv.org/abs/2505.24298 ; RLBoost: https://arxiv.org/abs/2510.19225
- LMSYS P2P weight update: https://lmsys.org/blog/2026-04-29-p2p-update/ ; vLLM weight transfer: https://docs.vllm.ai/en/stable/training/weight_transfer/
- RLlib: https://docs.ray.io/en/latest/rllib/rllib-new-api-stack.html , https://docs.ray.io/en/latest/rllib/rllib-algorithms.html , https://docs.ray.io/en/latest/rllib/multi-agent-envs.html , fault tolerance config https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.fault_tolerance.html
- Ray on-prem: https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html
- TorchRL: https://docs.pytorch.org/rl/main/reference/collectors_distributed.html , https://docs.pytorch.org/rl/0.13/reference/modules_inference_server.html
- PufferLib: https://puffer.ai/docs.html , https://github.com/PufferAI/PufferLib , https://pypi.org/project/pufferlib/ , paper https://arxiv.org/abs/2406.12905
- Neural MMO 2.0 / competition: https://arxiv.org/abs/2311.03736 , https://arxiv.org/abs/2508.12524
- Lux AI S3: https://neurips.cc/virtual/2024/competition/84798
- grpc.io performance: https://grpc.io/docs/guides/performance/
- PEP 574: https://peps.python.org/pep-0574/ ; zstd/lz4 benchmarks: https://github.com/facebook/zstd ; safetensors: https://huggingface.co/docs/safetensors/index
- SkyPilot managed jobs: https://docs.skypilot.ai/en/latest/examples/managed-jobs.html

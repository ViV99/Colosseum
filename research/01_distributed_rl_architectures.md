# Distributed Reinforcement Learning: A Comprehensive Technical Guide

> Research date: 2026-03-18
> For competitive bot programming and efficient agent training

---

## Table of Contents

1. [Classic Distributed RL Architectures](#1-classic-distributed-rl-architectures)
2. [Modern Distributed RL Architectures (2022–2025)](#2-modern-distributed-rl-architectures-2022-2025)
3. [Key Concepts and Techniques](#3-key-concepts-and-techniques)
4. [Hardware Considerations](#4-hardware-considerations)
5. [Performance Benchmarks and Comparisons](#5-performance-benchmarks-and-comparisons)
6. [Practical Recommendations for Competitive Bot Programming](#6-practical-recommendations-for-competitive-bot-programming)

---

## 1. CLASSIC DISTRIBUTED RL ARCHITECTURES

### 1.1 GORILA (General Reinforcement Learning Architecture) — 2015

**How it works:**

GORILA, proposed by Nair et al. at Google DeepMind in 2015, was the first large-scale distributed deep reinforcement learning system. It extends DQN to a distributed setting using a centralized parameter server architecture borrowed from DistBelief.

The architecture consists of four components:
- **Actors:** Each actor has a replica of the Q-network and generates experiences by interacting with the environment. These experiences are stored in both a local replay memory and a shared global replay memory.
- **Learners:** Each learner also has a replica of the Q-network. It samples experiences from the replay memory, computes gradients, and sends them to the parameter server.
- **Parameter Server:** The central hub that maintains the canonical Q-network parameters. The parameter vector is split disjointly across multiple machines, each responsible for applying gradient updates to a subset of parameters using asynchronous stochastic gradient descent.
- **Replay Memory:** A combination of local per-actor replay memories and a global consolidated replay memory.

**Key innovations:**
- First demonstration that DQN-style training could be massively parallelized with distributed actors and learners.
- Showed that sharing experiences (rather than just gradients) between actors and a replay buffer could lead to significant speedups.
- Achieved a 10x reduction in training duration on most Atari games compared to single-machine DQN.

**Pros:**
- Flexible distribution granularity (can scale actors and learners independently).
- Demonstrated the actor-learner paradigm that became foundational for all subsequent work.

**Cons:**
- High communication overhead due to the centralized parameter server.
- The parameter server becomes a bottleneck at scale.

**When to use:** Historical interest only. All subsequent architectures improve upon GORILA's design.

**Reference:** Nair et al., [Massively Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1507.04296) (2015).

---

### 1.2 A3C (Asynchronous Advantage Actor-Critic) — 2016

**How it works:**

A3C uses multiple worker agents running in parallel on a single machine. Each worker interacts with its own copy of the environment, computes gradients locally using a local copy of the network, and then pushes those gradients asynchronously to update a shared global network. Workers periodically pull the latest global parameters.

The architecture:
- **Global Network:** Maintains shared parameters for both the policy network π(s) and value network V(s).
- **Worker Agents:** Each worker maintains a local copy of the network, runs an episode (or partial episode) in its own environment instance, computes the advantage function A(s,a) = R - V(s), calculates gradients, and asynchronously pushes them to the global network.

**Key innovations:**
- **Eliminated the replay buffer:** By running multiple workers in parallel with different environment seeds, A3C naturally decorrelates training samples without needing experience replay. This makes it suitable for on-policy algorithms.
- **Single-machine parallelism:** Unlike GORILA, A3C was designed to run on a single multi-core CPU machine.

**Pros:**
- Simple to implement and understand.
- No replay buffer needed, reducing memory requirements.
- Works on a single machine with multiple CPU cores.

**Cons:**
- Asynchronous gradient updates can lead to stale gradients and instability.
- CPU-only training limits throughput compared to GPU-based approaches.
- Does not scale efficiently beyond a single machine.

**When to use:** Good for quick prototyping on a single machine. For serious training, PPO with vectorized environments is generally preferred.

**Frameworks:** PyTorch implementations ([ikostrikov/pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c)); RLlib supports A2C/A3C.

**Reference:** Mnih et al., [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) (2016).

---

### 1.3 IMPALA (Importance Weighted Actor-Learner Architecture) — 2018

**How it works:**

IMPALA separates acting and learning into distinct processes. Instead of workers computing and sending gradients, actors send entire trajectories of experience (sequences of states, actions, and rewards) to a centralized learner. The learner has access to a GPU and performs updates on mini-batches of trajectories.

The architecture:
- **Multiple Actors (on CPUs):** Each actor has a local copy of the policy, interacts with the environment, and generates trajectories sent to the learner.
- **Centralized Learner (on GPU):** Receives trajectories from actors, computes loss and gradients on mini-batches using GPU, updates the model, and periodically broadcasts updated parameters back to actors.

Because actors' policies lag behind the learner's policy, the data is effectively off-policy. IMPALA corrects for this with **V-trace**.

**V-trace off-policy correction:**

The n-step V-trace target at time s is:

```
v_s = V(x_s) + sum_{t=s}^{s+n-1} γ^{t-s} (prod_{i=s}^{t-1} c_i) * δ_t
```

where:
- `δ_t = ρ_t * (r_t + γ * V(x_{t+1}) - V(x_t))` is a TD-error weighted by a truncated importance ratio
- `ρ_t = min(ρ̄, π(a_t|x_t) / μ(a_t|x_t))` — clipped importance ratio
- `c_i = min(c̄, π(a_i|x_i) / μ(a_i|x_i))` — trace-cutting coefficients

The clipping constants ρ̄ and c̄ control bias-variance tradeoff.

**Key innovations:**
- **Trajectory sharing instead of gradient sharing:** Higher throughput because communication of trajectory data is less latency-sensitive than gradient sharing.
- **V-trace off-policy correction:** Enables stable learning even when actors are many updates behind the learner.
- **GPU-efficient learning:** The learner can batch trajectories from many actors for efficient GPU utilization.
- **Multi-task learning:** IMPALA demonstrated effective positive transfer across 30 DMLab tasks and 57 Atari games with a single agent.

**Pros:**
- Extremely high throughput: 250,000 frames per second, 30x faster than single-machine A3C.
- Scales to thousands of machines without sacrificing data efficiency.
- V-trace provides principled off-policy correction.

**Cons:**
- Complexity of the V-trace implementation.
- Still requires significant CPU resources for actors.
- Communication overhead between actors and the learner can be a bottleneck.

**When to use:** When you need high-throughput training and can tolerate some off-policy-ness. Excellent for multi-task training. Particularly suited for environments where CPU-based simulation is the bottleneck. **Strong choice for competitive bot training.**

**Frameworks:**
- [TorchBeast](https://github.com/facebookresearch/torchbeast) (Facebook/Meta): PyTorch IMPALA (MonoBeast for single machine, PolyBeast for distributed).
- [Moolib](https://github.com/facebookresearch/moolib): Successor to TorchBeast with better performance.
- [Cleanba](https://github.com/vwxyzjn/cleanba): CleanRL-style JAX implementation with IMPALA and PPO variants.
- [SEED RL](https://github.com/google-research/seed_rl): Google's implementation with centralized inference.
- [RLlib](https://docs.ray.io/en/latest/rllib/index.html): Ray's implementation supports IMPALA.
- [Stoix](https://github.com/EdanToledo/Stoix): JAX-based framework with both Anakin and Sebulba variants.

**Reference:** Espeholt et al., [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561) (ICML 2018).

---

### 1.4 Ape-X (Distributed Prioritized Experience Replay) — 2018

**How it works:**

Ape-X decouples acting from learning with a shared prioritized experience replay buffer as the central data structure. Unlike IMPALA (which uses a queue/stream of trajectories), Ape-X stores individual transitions in a large shared replay memory that the learner samples from based on priority.

The architecture:
- **Multiple Actors (typically 256+, on CPUs):** Each actor has its own environment instance and a local copy of the Q-network. Actors generate experience, compute initial priorities for transitions (using their local TD-error), and insert these transitions into the shared replay memory.
- **Shared Prioritized Replay Memory:** A large centralized buffer (typically 2 million transitions) that stores experience with associated priorities. Sampling is done proportionally to priority (priority exponent 0.6, importance sampling exponent 0.4).
- **Single Learner (on GPU):** Samples mini-batches from the replay memory based on priority, updates the Q-network, and updates the priorities of replayed transitions. The learner periodically pushes updated parameters to the actors.

For exploration, different actors use different epsilon-greedy exploration rates (e.g., actor i uses ε_i = ε^{1 + i/(N-1) * α}), ensuring diverse experience collection.

**Key innovations:**
- **Prioritized replay at distributed scale:** Ensures that the learner focuses on the highest-value transitions from a massive stream of experience.
- **Exploration through diversity:** Different epsilon values per actor provide a natural curriculum of exploration behaviors.
- **Generality:** The Ape-X framework can be applied to any off-policy algorithm (demonstrated with DQN and DPG).

**Pros:**
- State-of-the-art performance on Atari and continuous control tasks at the time.
- Very sample-efficient due to prioritized replay.
- Scales naturally with more actors (more diverse experience generation).
- Applicable to both discrete (DQN) and continuous (DPG/DDPG) action spaces.

**Cons:**
- Requires a large replay buffer (memory-intensive).
- Only applicable to off-policy algorithms.
- The centralized replay memory can become a bottleneck at extreme scale.

**When to use:** When you need an off-policy method with maximum sample efficiency. Best for environments where experience is expensive to generate.

**Frameworks:**
- [PyTorch Ape-X](https://github.com/younggyoseo/Ape-X)
- [Uber Research Ape-X](https://github.com/uber-research/ape-x) (optimized for single machine with 2 GPUs)
- RLlib supports distributed DQN with prioritized replay

**Reference:** Horgan et al., [Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933) (ICLR 2018).

---

### 1.5 R2D2 (Recurrent Replay Distributed DQN) — 2019

**How it works:**

R2D2 builds on the Ape-X architecture by adding recurrent (LSTM) networks and addressing the challenges of using recurrent networks with experience replay in a distributed setting.

Architecture specifics:
- **Q-network:** Uses a dueling network architecture with an LSTM layer after the convolutional stack.
- **Replay storage:** Instead of storing individual transitions, R2D2 stores fixed-length sequences of 80 timesteps (s, a, r), with adjacent sequences overlapping by 40 timesteps. Sequences never cross episode boundaries.
- **Recurrent state handling:**
  1. **Stored state:** Store the LSTM hidden state at the beginning of each sequence and use it to initialize replay.
  2. **Burn-in:** Use a burn-in period (the first portion of the replayed sequence) to allow the LSTM to "warm up" without contributing to the loss. This mitigates representational drift.
- **Distributed actors:** Typically 256 actors with different exploration rates, similar to Ape-X.
- **Additional components:** Double Q-learning, n-step TD returns (n=5), value function rescaling.

**Key innovations:**
- **Recurrent networks + distributed replay:** Showed how to effectively combine LSTMs with experience replay and distributed training.
- **Burn-in technique:** Sacrifices some data to allow the recurrent state to warm up, significantly improving training stability.

**Pros:**
- First agent to exceed human-level performance on 52 of 57 Atari games.
- The LSTM component is crucial for environments with partial observability or long-term dependencies.

**Cons:**
- Higher memory requirements due to sequence storage (80-step sequences with overlap).
- Recurrent networks are slower to train than feed-forward networks.

**When to use:** When your environment has partial observability or requires memory — **common in competitive bot programming.** If your game requires the agent to remember past states, plan over long horizons, or reason about opponent behavior history, R2D2's recurrent architecture is essential.

**Frameworks:**
- [PyTorch R2D2](https://github.com/ZiyuanMa/R2D2)
- SEED RL includes an R2D2 implementation
- Acme (DeepMind) includes R2D2

**Reference:** Kapturowski et al., [Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX) (ICLR 2019).

---

### 1.6 SEED RL (Scalable, Efficient Deep RL) — 2020

**How it works:**

SEED RL's key innovation is moving neural network inference from the actors to the learner. In IMPALA, each actor runs inference locally on CPU to select actions and then sends trajectories to the learner. In SEED RL, the actors only run the environment and send observations to the learner, which performs both inference (action selection) and training on specialized hardware (GPU/TPU).

The architecture:
- **Remote Environment Workers:** Run on CPUs and only execute environment steps. At each step, they send the observation to the learner and wait for an action.
- **Centralized Learner (on GPU/TPU):** Receives observations from all actors, batches them for efficient GPU inference, computes actions, sends actions back to actors, accumulates trajectories, and performs training updates.
- **Communication Layer:** Uses gRPC with asynchronous streaming RPCs to minimize latency.

**Key innovations:**
- **Centralized inference:** By moving inference to the learner, SEED avoids the cost of running neural networks on CPU and eliminates the need to broadcast model parameters.
- **Cost reduction:** 40-80% cost reduction compared to IMPALA for the same training speed.
- **TPU utilization:** SEED is designed to fully utilize TPUs, achieving 2.4 million frames per second with 64 Cloud TPU cores on DeepMind Lab (80x improvement over IMPALA).

**Pros:**
- Massively faster and cheaper than IMPALA at scale.
- Eliminates parameter synchronization overhead.
- Supports both on-policy (V-trace) and off-policy (R2D2) algorithms.

**Cons:**
- Introduces latency for each environment step (observation must be sent to learner, action returned).
- Requires low-latency networking between actors and learner.
- Google's implementation is in TF2 and is now archived.

**When to use:** When you have access to powerful GPUs/TPUs and need maximum throughput. Particularly valuable when the neural network is large (making CPU inference slow).

**Frameworks:**
- [SEED RL (Google)](https://github.com/google-research/seed_rl): Official TF2 implementation (archived).
- Conceptually similar to Sebulba architecture in Podracer/Cleanba/Stoix.

**Reference:** Espeholt et al., [SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference](https://arxiv.org/abs/1910.06591) (ICLR 2020).

---

## 2. MODERN DISTRIBUTED RL ARCHITECTURES (2022–2025)

### 2.1 Podracer Architectures: Anakin and Sebulba — 2021

**How they work:**

DeepMind's Podracer paper proposes two complementary architectures designed specifically for TPU Pods, implemented in JAX.

#### Anakin (Fully On-Device)

In Anakin, everything — the environment, action selection, and learning — runs on the accelerator (TPU/GPU). This requires the environments to be implemented as pure JAX functions so they can be compiled by XLA.

Data flow:
1. The environment is defined by `initial_state()` and `step(state, action) -> (new_state, reward)` functions, both implemented in JAX.
2. A single JAX function encompasses: stepping the environment, computing the policy output, selecting actions, computing the loss, and updating parameters.
3. This entire computation is vectorized across environments using `jax.vmap` and replicated across accelerator cores using `jax.pmap`.
4. Gradient synchronization across cores uses `jax.pmean`/`jax.psum`.

The result is a self-contained, deterministic training loop that can be JIT-compiled into a single XLA program. There is zero Python overhead during training, zero CPU-GPU data transfer, and perfect reproducibility.

#### Sebulba (Split Actor-Learner)

Sebulba handles environments that cannot run on the accelerator (e.g., Atari games that must run on CPU). It splits the available TPU/GPU cores into two sets:
- **Actor cores:** Handle inference. Python threads step batches of environments on the CPU host, send observations to actor TPU cores for batched inference, and receive actions back.
- **Learner cores:** Receive batched rollout data from actor cores and perform parameter updates using `jax.pmap` for data-parallel training.

**Key innovations:**
- **End-to-end JAX compilation (Anakin):** Eliminates all overhead from Python, CPU-GPU transfer, and serialization.
- **Flexible device allocation (Sebulba):** Can tune the ratio of actor-to-learner cores based on the relative cost of environment simulation vs. network updates.
- **Linear scaling:** Both architectures demonstrate near-linear scaling as the number of accelerator cores increases.

**Performance:**
- Anakin: 5 million steps/second for small networks on grid-world environments.
- Sebulba: Completed training of a 200-million-frame Atari game in one hour on an 8-core TPU.

**Frameworks:**
- [Cleanba](https://github.com/vwxyzjn/cleanba): Sebulba implementation with PPO and IMPALA in JAX.
- [Stoix](https://github.com/EdanToledo/Stoix): Both Anakin and Sebulba, many algorithms.
- [Mava](https://github.com/instadeepai/Mava): Multi-agent RL with both Podracer architectures.
- [InstaDeep Sebulba](https://github.com/instadeepai/sebulba): Reference implementation.

**Reference:** Hessel et al., [Podracer architectures for scalable Reinforcement Learning](https://arxiv.org/abs/2104.06272) (2021).

---

### 2.2 Sample Factory — 2020 (Updated Through 2024)

**How it works:**

Sample Factory targets maximum throughput on a **single machine**. It implements Asynchronous Proximal Policy Optimization (APPO) with a carefully designed architecture that minimizes idle time across all computation.

The architecture consists of three component types:
- **Rollout Workers:** Interact with environments to generate experience. Multiple workers run in separate processes, each managing multiple environment instances.
- **Policy Workers:** Run neural network inference on GPU to produce actions for the rollout workers. By separating inference from rollout, the GPU can be kept busy batching inference requests from multiple rollout workers.
- **Learners:** Consume experience from rollout workers and update the neural network on GPU.

These components communicate via **shared memory** and a fast queuing protocol. Shared memory avoids the overhead of serialization and inter-process data copying.

**Performance:**
- Achieves 130,000+ frames per second on a single-GPU commodity PC for 3D environments.
- Can train populations of agents on billions of environment transitions on commodity hardware.

**Pros:**
- Extremely high throughput on a single machine (no cluster needed).
- Works with any Gymnasium-compatible environment.
- Open source, well-documented, actively maintained.
- No distributed infrastructure complexity (no gRPC, no parameter servers).
- Excellent for "democratizing" large-scale RL.

**Cons:**
- Limited to single-machine scaling.
- Asynchronous PPO introduces some off-policy-ness.
- PyTorch-only.

**When to use:** When you want the highest throughput on a single machine without dealing with distributed infrastructure. **Ideal for competitive bot programming where you have a single powerful workstation.**

**GitHub:** [sample-factory](https://github.com/alex-petrenko/sample-factory)

**Reference:** Petrenko et al., [Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS](https://arxiv.org/abs/2006.11751) (ICML 2020).

---

### 2.3 PureJaxRL — 2023–2024

**How it works:**

PureJaxRL takes the end-to-end JAX approach to its extreme: the entire RL training loop (environment stepping, action selection, loss computation, parameter updates) is written as a single JAX function that is JIT-compiled and executed entirely on a GPU or TPU. There is literally **zero Python overhead** during training.

Using `jax.vmap`, you can vectorize across thousands of environments; using `jax.vmap` again to train multiple agents in parallel. Training 2048 agents on CartPole takes about half the time it takes CleanRL to train a single agent.

**Performance:**
- On CartPole-v1: Trains 2048 agents in about the same time it takes CleanRL to train 1 agent.
- On MinAtar: Achieves 1000x+ speedups over standard implementations.
- 4000x+ speedups over traditional implementations in many cases.

**Pros:**
- Fastest possible training for JAX-compatible environments.
- Perfect reproducibility and determinism.
- Enables training many agents in parallel for statistically significant results.
- Simple, readable, single-file implementations.

**Cons:**
- Requires environments to be written in pure JAX.
- Limited environment ecosystem compared to Gymnasium (though growing).
- JIT compilation can be slow for the first call.

**When to use:** When your environment exists in or can be ported to JAX, and you want maximum iteration speed. Exceptional for board games, card games, simple grid-world games.

**Frameworks:**
- [PureJaxRL](https://github.com/luchris429/purejaxrl): The original implementations.
- [Stoix](https://github.com/EdanToledo/Stoix): Research-friendly JAX RL with many algorithms.
- [Rejax](https://github.com/keraJLi/rejax): Pure JAX RL algorithms with jit/vmap/pmap support.
- [JaxMARL](https://github.com/FLAIROx/JaxMARL): Multi-agent RL in JAX (up to 12,500x speedups).

**Reference:** Chris Lu et al., [Achieving 4000x Speedups and Meta-Evolving Discoveries with PureJaxRL](https://chrislu.page/blog/meta-disco/) (2023).

---

### 2.4 PufferLib — 2024

**How it works:**

PufferLib addresses the "impedance mismatch" between complex environments and RL libraries. It provides one-line environment wrappers that handle compatibility issues and fast vectorization. Ships with PuffeRL, a training algorithm that achieves millions of steps per second in a single ~1000-line script.

**Competition-proven results:**
- NeurIPS 2023 Neural MMO competition (no other RL library could handle Neural MMO 2.0 natively).
- Pokemon Red: 7,000 steps/second (2-3x faster than the original SB3 project).
- Neural MMO: Enabled training of competent policies in 8 hours on a single desktop.

**Pros:**
- Extremely practical for complex game environments.
- Works with familiar libraries (CleanRL, SB3).
- Easy to set up and iterate with.
- Good documentation and community support.

**When to use:** When you are working with a complex game environment for a competition and need things to "just work." **If you are participating in competitions like Neural MMO, Lux AI, or similar game-based challenges, PufferLib should be your first stop.**

**GitHub:** [PufferLib](https://github.com/PufferAI/PufferLib)

**Reference:** Suarez, [PufferLib: Making Reinforcement Learning Libraries and Environments Play Nice](https://arxiv.org/abs/2406.12905) (2024).

---

### 2.5 Cleanba — 2023

**How it works:**

Cleanba implements DeepMind's Sebulba Podracer architecture with a focus on reproducibility and transparency. It provides distributed variants of PPO and IMPALA implemented in JAX with EnvPool for fast environment execution.

**Key finding:** In the Sebulba architecture, IMPALA (with V-trace correction) benefits from asynchronous actor-learner execution without losing data efficiency. PPO, however, suffers because its 16 gradient updates per rollout amplify the staleness of the data. This means **IMPALA is the better algorithm choice for distributed Sebulba-style training.**

**Benchmark results:**
- Cleanba's IMPALA and PPO achieve about 165% median Human Normalized Score (HNS) on Atari with sticky actions.
- Under 1 GPU 10 CPU: Cleanba's IMPALA is 6.8x faster than MonoBeast (TorchBeast) and 1.2x faster than Moolib.
- Under max spec (8 GPU 40 CPU): Cleanba's IMPALA is 5x faster than MonoBeast and 2x faster than Moolib.

**GitHub:** [Cleanba](https://github.com/vwxyzjn/cleanba)

**Reference:** Huang et al., [Cleanba: A Reproducible and Efficient Distributed Reinforcement Learning Platform](https://arxiv.org/abs/2310.00036) (2023).

---

### 2.6 GPU-Accelerated Environments Ecosystem

#### NVIDIA Isaac Gym / Isaac Lab

Isaac Gym runs physics simulation on the GPU alongside neural network training, eliminating CPU-GPU data transfer entirely. It supports tens of thousands of simultaneous environments on a single GPU.

**Performance highlights (single NVIDIA A100):**
- Ant: Performant locomotion in 20 seconds
- Humanoid: 4 minutes
- ANYmal: Under 2 minutes
- Shadow Hand cube rotation: 35 minutes

**Note:** Isaac Gym is now legacy software. NVIDIA recommends [Isaac Lab](https://isaac-lab.github.io/), built on Isaac Sim.

**Reference:** Makoviychuk et al., [Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning](https://arxiv.org/abs/2108.10470) (NeurIPS 2021).

#### Google Brax

Brax is a differentiable physics engine written entirely in JAX. Reaches hundreds of millions on a TPU Pod. 100-1000x faster training compared to traditional MuJoCo CPU setups.

**GitHub:** [google/brax](https://github.com/google/brax)

#### Gymnax

Classic RL environments (CartPole, bsuite, MinAtar, etc.) implemented in JAX, enabling `jit` and `vmap`/`pmap` for massive vectorization on GPU.

**GitHub:** [RobertTLange/gymnax](https://github.com/RobertTLange/gymnax)

#### Pgx (Hardware-Accelerated Game Simulators)

Pgx provides high-performance board game and card game simulators written in JAX. 10-100x faster than Python-based alternatives like PettingZoo and OpenSpiel. **Particularly relevant for competitive bot programming involving board games.**

**Reference:** [Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning](https://ar5iv.labs.arxiv.org/html/2303.17503)

#### EnvPool

EnvPool is a C++-based parallel environment execution engine that accelerates CPU-based environments.

**Performance:**
- 1 million FPS on Atari, 3 million FPS on MuJoCo (on 256 CPU cores / NVIDIA DGX-A100).
- 14.9x / 19.6x faster than `gym.vector_env` on high-end hardware.
- 3.1x / 2.9x faster than `gym.vector_env` on a typical 12-core PC.

**Limitation:** Environments must be translated to C++ (Python environments cannot be accelerated).

**GitHub:** [sail-sg/envpool](https://github.com/sail-sg/envpool)

**Reference:** [EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine](https://arxiv.org/abs/2206.10558) (NeurIPS 2022).

---

### 2.7 Key Frameworks for Distributed RL

#### Ray RLlib (v2.54.0, February 2026)

Ray RLlib is the most production-ready distributed RL framework. Its architecture centers on:
- **Algorithm:** The central runtime class.
- **EnvRunnerGroup:** N EnvRunner actors for sample collection.
- **LearnerGroup:** M Learner actors for computing gradients and updating models.
- **RLModules:** Framework-specific neural network wrappers.

RLlib supports multi-GPU training on a single node and multi-node clusters. It implements PPO, DQN, IMPALA, SAC, and many more algorithms. It natively supports multi-agent RL with independent learning, collaborative training, and adversarial self-play.

**When to use:** When you need a production-grade, well-supported framework that can scale from a laptop to a cluster.

**Reference:** [RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html).

#### DeepMind Acme

Acme is a modular framework where agents are composed from interchangeable acting, learning, and replay components. Single-process agents can be trivially scaled to distributed versions using the same core components. Uses Reverb for experience replay.

**When to use:** When you want clean, modular code that can easily transition from single-process debugging to distributed execution.

**Reference:** Hoffman et al., [Acme: A Research Framework for Distributed Reinforcement Learning](https://arxiv.org/abs/2006.00979).

#### DeepMind Reverb

Reverb is a purpose-built experience replay server supporting FIFO, LIFO, priority queues, uniform sampling, and prioritized sampling. It scales to thousands of concurrent clients with minimal overhead and is the backbone of Acme's distributed agents.

**GitHub:** [google-deepmind/reverb](https://github.com/google-deepmind/reverb)

**Reference:** Cassirer et al., [Reverb: A Framework For Experience Replay](https://arxiv.org/abs/2102.04736).

---

## 3. KEY CONCEPTS AND TECHNIQUES

### 3.1 On-Policy vs. Off-Policy in Distributed Settings

The fundamental tension in distributed RL is between throughput and learning signal quality.

**On-policy distributed (e.g., distributed PPO, Anakin):**
- Actors and learners are synchronized: the learner only trains on data generated by the current policy.
- Requires synchronization barriers, creating idle time.
- More stable gradients and simpler implementation.
- Lower hardware utilization due to synchronization waits.
- Lower sample efficiency: data is used once and discarded.

**Off-policy distributed (e.g., IMPALA, Ape-X, R2D2):**
- Actors and learners are decoupled: the learner can train asynchronously.
- Higher throughput because no synchronization barriers.
- Requires off-policy correction (V-trace, importance sampling).
- Better hardware utilization due to asynchronous execution.

**The Cleanba finding:** IMPALA (off-policy with V-trace) benefits from the Sebulba asynchronous architecture without losing data efficiency, while PPO (on-policy) suffers reduced data efficiency. **If you choose to use asynchronous distributed training, IMPALA with V-trace is likely a better choice than PPO.**

---

### 3.2 Experience Replay in Distributed Settings

- **Uniform replay (DQN, GORILA):** Simple but wastes capacity on low-value transitions.
- **Prioritized replay (Ape-X, R2D2):** Samples proportional to TD-error. In Ape-X, actors compute initial priorities at no extra cost (using their local Q-network).
- **Sequence-based replay (R2D2):** Stores fixed-length sequences (80 steps, 40-step overlap) instead of individual transitions. Essential for recurrent architectures.
- **FIFO queues (IMPALA):** On-policy-like training without a replay buffer, using a queue of recent trajectories.

---

### 3.3 Policy Lag and Correction Methods

Policy lag is the number of learner updates that have occurred between when an actor generated its data and when the learner processes it.

- **V-trace (IMPALA):** The most widely used correction. Uses truncated importance sampling ratios to downweight stale data. The clipping ensures samples are never upweighted (only downweighted), so stale trajectories gradually lose impact.
- **PPO clipping:** PPO's clipped objective naturally limits the impact of off-policy data by constraining how much the policy can change in a single update.
- **Retrace (Munos et al., 2016):** An off-policy correction for multi-step returns used in Q-learning settings (Ape-X, R2D2).
- **Adaptive Actor Policy Synchronization (AAPS, 2025):** A divergence-triggered update mechanism that synchronizes actor policies with the learner only when the KL divergence between them exceeds a threshold.

---

### 3.4 Vectorized Environments vs. Distributed Environments

**Decision framework for competitive bot programming:**

1. If a JAX version of your game exists (Pgx for board games, JaxMARL for multi-agent): **Use GPU-vectorized environments with PureJaxRL or Stoix Anakin.** This is the fastest option by far.
2. If the environment can be ported to C++ but not GPU: **Use EnvPool** for fast CPU vectorization with Cleanba or CleanRL.
3. If the environment is a complex Python game: **Use PufferLib or Sample Factory** for optimized CPU-based vectorization with a single machine.
4. If you need massive scale or have cluster access: **Use Ray RLlib** or a Sebulba-based framework.

---

## 4. HARDWARE CONSIDERATIONS

### 4.1 Single Machine Multi-GPU

For reinforcement learning, single-machine multi-GPU setups are often the most cost-effective option because:
1. RL is typically more bottlenecked by environment simulation (CPU) than by neural network training (GPU).
2. Communication latency between GPUs on the same machine is minimal (NVLink, PCIe).
3. Most RL neural networks are small enough to fit on a single GPU.

**Data Parallelism for RL:** Each GPU holds a complete copy of the model. The batch of experience is split across GPUs, each computes gradients on its portion, and gradients are averaged via all-reduce. PyTorch's Distributed Data Parallel (DDP) achieves 4.5x the throughput of DataParallel on 4 GPUs because DDP uses one process per GPU.

**Practical recommendation:** For competitive bot programming, a single machine with 1-2 high-end GPUs (RTX 4090, A100, H100) and many CPU cores (16-64) is the sweet spot.

**Estimated costs (2024-2025):**
- RTX 4090 workstation: $3,000-5,000 (one-time). Can train competitive agents for most bot programming competitions.
- Cloud equivalent: AWS p3.2xlarge (V100) at ~$3/hr or p4d.24xlarge (A100x8) at ~$32/hr.

---

### 4.2 Cloud Training Optimization

**Cost optimization strategies for competitive bot programming:**

1. **Use spot/preemptible instances for actors, reserved for learners (RLBoost approach):** Rollout accounts for up to 73% of overall training time. Rollout workers are stateless and can tolerate preemption.
2. **Instance diversification:** Spread across 10-15 instance types, multiple availability zones and regions.
3. **Aggressive checkpointing:** Save model weights, optimizer state, replay buffer state, and random seeds at regular intervals.
4. **Start local, scale to cloud:** Develop and debug on your local machine. Only move to cloud when you need scale.
5. **Containerize everything:** Use Docker containers so you can switch between local, AWS, and GCP without code changes.

---

## 5. PERFORMANCE BENCHMARKS AND COMPARISONS

### 5.1 Throughput Comparisons (Frames Per Second)

| System | Peak FPS | Hardware | Notes |
|---|---|---|---|
| **PureJaxRL** (CartPole) | ~10,000,000+ | Single GPU | 2048 parallel agents |
| **Brax** (continuous control) | ~100,000,000+ | TPU v3 8x8 pod (64 chips) | Hundreds of millions of physics steps/sec |
| **Isaac Gym** (robotics) | ~100,000-1,000,000+ | Single A100 GPU | Tens of thousands of parallel envs |
| **JaxMARL** (multi-agent) | 12,500x speedup | GPU | vs CPU-based MARL |
| **SEED RL** (DeepMind Lab) | 2,400,000 | 64 Cloud TPU cores | 80x over IMPALA |
| **IMPALA** (Atari) | 250,000 | Thousands of machines | Original DeepMind setup |
| **Sample Factory** (VizDoom) | 130,000+ | Single GPU + multi-core CPU | 3D pixel-based environments |
| **EnvPool** (Atari) | 1,000,000 | 256 CPU cores | C++ threadpool acceleration |
| **Cleanba IMPALA** | 6.8x MonoBeast | 1 GPU 10 CPU | Sebulba architecture, JAX+EnvPool |

### 5.2 Training Time Comparisons

| Task | System | Training Time | Hardware |
|---|---|---|---|
| Atari Pong (solve) | EnvPool + PPO | 5 minutes | Laptop |
| Atari 200M frames | Cleanba Sebulba | 1 hour | 8-core TPU |
| Ant locomotion | Isaac Gym | 20 seconds | Single A100 |
| Humanoid locomotion | Isaac Gym | 4 minutes | Single A100 |
| Shadow Hand cube rotation | Isaac Gym | 35 minutes | Single A100 |
| Neural MMO (competitive) | PufferLib | 8 hours | Single RTX 4090 |
| CartPole (2048 agents) | PureJaxRL | ~half the time of 1 CleanRL agent | Single GPU |

---

## 6. PRACTICAL RECOMMENDATIONS FOR COMPETITIVE BOT PROGRAMMING

### 6.1 Algorithm Selection

**For board games / turn-based games:**
- If a JAX environment exists (check Pgx): Use PureJaxRL + PPO for maximum speed.
- Otherwise: Use PPO or IMPALA with self-play.
- Consider AlphaZero-style MCTS + neural network if the game tree is tractable.

**For real-time strategy / multi-agent games (Lux AI, Neural MMO):**
- Start with PufferLib for compatibility.
- Use IMPALA (V-trace) if you need distributed training.
- For self-play: Use RLlib's multi-agent support or PufferLib's built-in support.

**For continuous control / robotics competitions:**
- Use Isaac Lab or Brax for GPU-accelerated simulation.
- PPO is the standard algorithm; SAC if you need sample efficiency.

### 6.2 Framework Decision Tree

```
Is your environment available in JAX?
  YES -> Do you need multi-agent?
           YES -> JaxMARL + PureJaxRL
           NO  -> Stoix (Anakin) or PureJaxRL
  NO  -> Is your environment complex/non-standard?
           YES -> PufferLib (best compatibility)
           NO  -> Is single-machine sufficient?
                    YES -> Sample Factory (highest single-machine throughput)
                           or EnvPool + CleanRL
                    NO  -> Do you need production-grade distributed?
                             YES -> Ray RLlib
                             NO  -> Cleanba (Sebulba) for research
```

### 6.3 Key Takeaways

1. **The biggest speed gains come from eliminating CPU-GPU data transfer.** If you can port your environment to JAX, the speedup is 100-1000x.
2. **For most competitions, a single machine is sufficient.** Sample Factory, PufferLib, and PureJaxRL can achieve millions of steps per second on a single workstation.
3. **IMPALA with V-trace is the best algorithm for distributed asynchronous training.** When running distributed, use IMPALA.
4. **PPO remains the best on-policy algorithm for synchronized training.** When running fully synchronous (vectorized envs, single learner), PPO with GAE is hard to beat.
5. **Invest engineering time in the environment, not the infrastructure.** Use existing frameworks rather than building custom distributed systems.
6. **Spot instances for rollouts, reserved instances for training.** The RLBoost approach can reduce cloud costs by 60-70%.
7. **Start simple, scale up.** Begin with CleanRL or PufferLib on your local machine. Profile to find the bottleneck. Only add complexity where the profiling justifies it.

---

## Sources

### Papers
- [GORILA: Massively Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1507.04296)
- [A3C: Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
- [Ape-X: Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933)
- [R2D2: Recurrent Experience Replay in Distributed Reinforcement Learning](https://openreview.net/forum?id=r1lyTjAqYX)
- [SEED RL: Scalable and Efficient Deep-RL with Accelerated Central Inference](https://arxiv.org/abs/1910.06591)
- [Podracer architectures for scalable Reinforcement Learning](https://arxiv.org/abs/2104.06272)
- [Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS](https://arxiv.org/abs/2006.11751)
- [Cleanba: A Reproducible and Efficient Distributed Reinforcement Learning Platform](https://arxiv.org/abs/2310.00036)
- [PufferLib: Making Reinforcement Learning Libraries and Environments Play Nice](https://arxiv.org/abs/2406.12905)
- [Isaac Gym: High Performance GPU-Based Physics Simulation For Robot Learning](https://arxiv.org/abs/2108.10470)
- [EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine](https://arxiv.org/abs/2206.10558)
- [Reverb: A Framework For Experience Replay](https://arxiv.org/abs/2102.04736)
- [Acme: A Research Framework for Distributed Reinforcement Learning](https://arxiv.org/abs/2006.00979)
- [Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning](https://ar5iv.labs.arxiv.org/html/2303.17503)
- [Acceleration for Deep Reinforcement Learning using Parallel and Distributed Computing: A Survey](https://dl.acm.org/doi/10.1145/3703453)
- [RLBoost: Harvesting Preemptible Resources for Cost-Efficient Reinforcement Learning](https://arxiv.org/html/2510.19225v1)
- [PureJaxRL: Achieving 4000x Speedups](https://chrislu.page/blog/meta-disco/)

### Frameworks
- [Ray RLlib Documentation](https://docs.ray.io/en/latest/rllib/index.html)
- [Sample Factory GitHub](https://github.com/alex-petrenko/sample-factory)
- [PufferLib / PufferAI](https://puffer.ai)
- [Cleanba GitHub](https://github.com/vwxyzjn/cleanba)
- [Stoix GitHub](https://github.com/EdanToledo/Stoix)
- [PureJaxRL GitHub](https://github.com/luchris429/purejaxrl)
- [Google Brax GitHub](https://github.com/google/brax)
- [Gymnax GitHub](https://github.com/RobertTLange/gymnax)
- [JaxMARL GitHub](https://github.com/FLAIROx/JaxMARL)
- [EnvPool GitHub](https://github.com/sail-sg/envpool)
- [SEED RL GitHub (archived)](https://github.com/google-research/seed_rl)
- [TorchBeast GitHub](https://github.com/facebookresearch/torchbeast)
- [Moolib GitHub](https://github.com/facebookresearch/moolib)
- [DeepMind Reverb GitHub](https://github.com/google-deepmind/reverb)
- [DeepMind Acme GitHub](https://github.com/google-deepmind/acme)
- [Mava GitHub](https://github.com/instadeepai/Mava)
- [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym)
- [Rejax GitHub](https://github.com/keraJLi/rejax)

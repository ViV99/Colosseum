# Comprehensive Survey of Reinforcement Learning Frameworks and Libraries

**Date:** March 2026
**Purpose:** Reference for competitive bot programming using reinforcement learning

---

## Table of Contents

1. [General RL Frameworks](#1-general-rl-frameworks)
2. [Distributed RL Training Frameworks](#2-distributed-rl-training-frameworks)
3. [Multi-Agent RL Frameworks](#3-multi-agent-rl-frameworks)
4. [Environment Interfaces](#4-environment-interfaces)
5. [JAX-Based RL Ecosystem](#5-jax-based-rl-ecosystem)
6. [Comparative Summary Tables](#6-comparative-summary-tables)
7. [Recommendations for Competitive Bot Programming](#7-recommendations-for-competitive-bot-programming)

---

## 1. General RL Frameworks

### 1.1 Ray RLlib

**Repository:** https://github.com/ray-project/ray (RLlib is a subpackage)
**Stars:** ~41.3k (entire Ray project)
**Framework:** PyTorch (default), TensorFlow (legacy support)
**Last Updated:** Actively maintained (latest: Ray 2.54.0); Ray joined the PyTorch Foundation in Oct 2025
**License:** Apache 2.0

**What it does:** RLlib is the industry-grade, scalable reinforcement learning library built on top of Ray. It is designed for production-level, highly distributed RL workloads while maintaining unified and simple APIs. Used in climate control, industrial control, manufacturing, logistics, finance, gaming, automobile, robotics, and more.

**Key Features:**
- Off-the-shelf distributed and fault-tolerant algorithms
- Multi-GPU and multi-node training (single-node via open-source Ray; multi-node clusters via Anyscale)
- Comprehensive multi-agent RL support with self-play, dynamic policy addition/removal
- Custom environments via Gymnasium or OpenSpiel
- Custom PyTorch models, optimizers, and loss functions
- Hyperparameter tuning via Ray Tune integration
- Checkpointing, logging, and experiment management built-in

**Supported Algorithms:** PPO, APPO (Async PPO based on IMPALA), DQN (with Rainbow improvements), SAC, MARWIL (offline/batch RL), DDPG, TD3, BC, Dreamer, SlateQ, and more. All algorithms support multi-GPU training.

**Distributed Training:** First-class. Distributed sample collection via EnvRunner actors, configurable number of parallel workers (`--num-env-runners`), multiple GPU-based Learners. Example: Atari PPO with 4 learners and 95 env runners.

**Multi-Agent Support:** Native. Supports self-play, multiple policies trained simultaneously, agents sharing or not sharing information. Agents can access other agents' data for centralized training.

**Imitation/Behavioral Cloning:** Supports MARWIL (Monotonic Advantage Re-Weighted Imitation Learning) and BC natively.

**Pros for Competitive Bot Programming:**
- Best-in-class distributed scaling -- can throw hardware at the problem
- Native multi-agent and self-play support is ideal for competitive games
- Mature, production-tested, enormous community
- Flexible enough to implement custom game environments
- Population-based training support

**Cons for Competitive Bot Programming:**
- Complex setup; steep learning curve for the Ray ecosystem
- Heavy dependency tree; overkill for simple single-agent experiments
- Higher resource demands even for small projects
- API has changed significantly between versions, causing migration headaches
- Debugging distributed code is non-trivial

**Customization:** High. Custom environments, models, losses, and exploration strategies are well-supported through documented APIs.

**Performance:** Excellent at scale. Less optimal for small experiments on a single machine due to overhead.

---

### 1.2 Stable-Baselines3 (SB3)

**Repository:** https://github.com/DLR-RM/stable-baselines3
**Stars:** ~12.8k
**Framework:** PyTorch
**Latest Version:** 2.8.0a4 (docs), 2.7.0a0 (release, June 2025)
**License:** MIT

**What it does:** SB3 provides reliable, well-tested implementations of state-of-the-art model-free RL algorithms. It is the most popular entry-point for RL practitioners due to its simplicity, documentation quality, and clean API.

**Key Features:**
- Clean, simple API -- training an agent is a few lines of code
- Thoroughly tested against published results; all functions typed and documented
- TensorBoard/CSV logging, callbacks, vectorized environments
- Environment wrappers for preprocessing (e.g., Atari frame stacking)
- Dictionary observation space support
- RL Baselines3 Zoo: pre-trained agents and hyperparameter tuning via Optuna

**Supported Algorithms (core):** A2C, PPO, DQN, DDPG, TD3, SAC, HER.
**SB3-Contrib (experimental):** Recurrent PPO (PPO LSTM), CrossQ, TQC, QR-DQN, Maskable PPO (invalid action masking -- very useful for game AI).

**Distributed Training:** Not natively supported. SB3 is designed for single-machine training. You can use vectorized environments (SubprocVecEnv) for parallelism on one machine but there is no built-in multi-node distribution.

**Multi-Agent Support:** Not natively supported. SB3 is a single-agent library. You can hack multi-agent setups via self-play wrappers, but it is not a first-class feature.

**Imitation/Behavioral Cloning:** Via the companion `imitation` library (`pip install imitation`), which implements BC, DAgger, GAIL, AIRL, SQIL, and more on top of SB3.

**Pros for Competitive Bot Programming:**
- Fastest way to get a baseline agent running
- Maskable PPO (SB3-Contrib) is ideal for games with invalid actions
- `imitation` library lets you bootstrap from expert replays
- Huge community; extensive examples and tutorials
- Clean code makes debugging straightforward

**Cons for Competitive Bot Programming:**
- No built-in distributed training -- single machine only
- No native multi-agent support
- Limited to model-free algorithms; no MCTS, search, or model-based methods
- Performance ceiling on a single machine for complex environments
- Not ideal for very large-scale training

**Customization:** Moderate-to-high. Custom environments must follow the Gymnasium API. Custom policies and feature extractors are straightforward. Callbacks allow custom training logic.

**Performance:** Good for single-machine workloads. Vectorized environments provide decent throughput.

---

### 1.3 CleanRL

**Repository:** https://github.com/vwxyzjn/cleanrl
**Stars:** ~8.4k
**Framework:** PyTorch (primary), JAX (v1.0+)
**Last Updated:** Actively maintained
**License:** MIT

**What it does:** CleanRL provides high-quality, single-file implementations of deep RL algorithms. Each algorithm variant is self-contained in one file (e.g., `ppo_atari.py` is ~340 lines). It is NOT a modular library meant to be imported -- it is a reference codebase for understanding and prototyping.

**Key Features:**
- Single-file implementations: all details of an algorithm variant in one standalone file
- Benchmarked implementations across 34+ games
- Experiment tracking via Weights & Biases
- Docker support for scalable cloud orchestration (tested on 2000+ machines)
- Gymnasium migration complete
- EnvPool integration for 3-4x speedup on Atari (Linux)
- JAX support added in v1.0
- LeanRL (PyTorch CUDAGraphs) variant for optimized GPU execution

**Supported Algorithms:** PPO, DQN, C51, DDPG, TD3, SAC, PPG.

**Distributed Training:** Not built-in as a feature, but the architecture supports cloud orchestration via Docker. Each run is independent -- you scale by running many experiments in parallel.

**Multi-Agent Support:** Has PettingZoo examples, but not a primary focus.

**Imitation/Behavioral Cloning:** Not built-in.

**Pros for Competitive Bot Programming:**
- Best codebase for understanding exactly what an algorithm does
- Easy to modify and prototype new ideas (no framework overhead)
- Fast iteration cycle for experimentation
- JAX support enables hardware acceleration experiments
- EnvPool integration for high throughput

**Cons for Competitive Bot Programming:**
- Not designed to be imported as a library -- you fork and modify files
- Limited algorithm selection compared to SB3/RLlib
- No multi-agent or self-play infrastructure
- Code duplication by design (each variant is self-contained)
- No built-in distributed training management

**Customization:** Very high -- by design, you directly modify the single-file implementation. Maximum control with minimal abstraction layers.

**Performance:** Good. EnvPool integration and JAX support can achieve high throughput. LeanRL with CUDAGraphs further optimizes GPU utilization.

---

### 1.4 TorchRL

**Repository:** https://github.com/pytorch/rl
**Stars:** ~2.7k
**Framework:** PyTorch (official PyTorch RL library)
**Latest Version:** 0.11 (synced with PyTorch releases)
**Last Updated:** Actively maintained by Meta/PyTorch team
**License:** MIT

**What it does:** TorchRL is the official PyTorch reinforcement learning library. It provides modular, primitive-first components built around the TensorDict data structure, enabling streamlined algorithm development. A PPO training script can be written in under 100 lines.

**Key Features:**
- TensorDict primitive for unified data handling across RL components
- Standardized environment API supporting Gymnasium, Jumanji, RoboHive, dm_env, and more
- Data collectors and replay buffers
- Prebuilt objective functions (DQN, A2C, PPO, SAC, etc.)
- Minimal dependencies (Python stdlib, NumPy, PyTorch)
- NEW: Comprehensive LLM API for RLHF, SFT, and tool-augmented training
- Enhanced vLLM integration with async inference service
- BenchMARL integration for multi-agent benchmarking

**Supported Algorithms:** PPO, DQN, SAC, DDPG, TD3, A2C, REDQ, CQL, IQL, Decision Transformer, and more via objectives/losses. Additional trainers (SAC, TD3, DQN) in development.

**Distributed Training:** Supported via PyTorch's native distributed primitives and async data collectors. The LLM API includes distributed training support.

**Multi-Agent Support:** Via BenchMARL (built on TorchRL), which provides MARL benchmarking with MAPPO, MADDPG, MASAC, QMIX, VDN, etc.

**Imitation/Behavioral Cloning:** Offline RL support (CQL, IQL, Decision Transformer). D4RL and OpenX dataset integration.

**Pros for Competitive Bot Programming:**
- Official PyTorch library -- guaranteed long-term maintenance
- TensorDict provides clean data flow for custom algorithms
- Modular design lets you mix and match components
- Growing ecosystem (BenchMARL, ACEGEN, RL4CO)
- D4RL integration for offline RL/behavioral cloning

**Cons for Competitive Bot Programming:**
- Still maturing; API is evolving
- Smaller community compared to SB3 or RLlib
- Documentation gaps in some areas
- Multi-agent support is indirect (via BenchMARL)
- Not as battle-tested in production as RLlib

**Customization:** Very high. Primitive-first design means you compose your own training loops from building blocks.

**Performance:** Good. Leverages PyTorch optimization, torch.compile, and CUDA. Synced with latest PyTorch versions.

---

### 1.5 Tianshou

**Repository:** https://github.com/thu-ml/tianshou
**Stars:** ~8.0k
**Framework:** PyTorch + Gymnasium
**Latest Version:** 2.x (major overhaul, not backward compatible with v1)
**Requires:** Python >= 3.11
**License:** MIT

**What it does:** Tianshou is an elegant, modular PyTorch deep RL library from Tsinghua University. It uniquely supports online (on/off-policy), offline RL, experimental multi-agent RL, and model-based RL through a unified interface. Version 2 introduced a complete redesign separating Algorithm and Policy abstractions.

**Key Features:**
- Clear separation between Algorithm and Policy (v2)
- 20+ classic algorithms with unified interface
- Vectorized environments (sync/async), EnvPool support
- Recurrent state representation support (RNN-style training for POMDPs)
- N-step returns, PER, GAE -- optimized via Numba JIT
- Rigorous testing: full agent training in test suite ensures reproducibility
- MuJoCo benchmarks reaching or exceeding existing baselines

**Supported Algorithms:** REINFORCE, A2C, TRPO, PPO, DDPG, TD3, SAC, DQN (Double, Dueling, PER, N-step), C51, Rainbow, IQN, BCQ, CQL, discrete BCQ, GAIL, and more.

**Distributed Training:** Not natively distributed across machines. Supports vectorized environments for single-machine parallelism.

**Multi-Agent Support:** Experimental. Has tutorials for training agents in PettingZoo environments (e.g., Tic-Tac-Toe with DQN).

**Imitation/Behavioral Cloning:** Supports GAIL and offline RL algorithms (BCQ, CQL, discrete BCQ).

**Pros for Competitive Bot Programming:**
- One of the most comprehensive algorithm selections in any single library
- Excellent for rapid prototyping with its clean API
- Numba-optimized PER and GAE for performance
- POMDP support (RNN policies) useful for imperfect information games
- Offline RL support lets you learn from recorded game data

**Cons for Competitive Bot Programming:**
- No native distributed training
- Multi-agent support is experimental, not production-ready
- Version 2 breaking changes may affect existing codebases
- Smaller community than SB3 or RLlib
- Requires Python 3.11+

**Customization:** High. Modular low-level API is designed for algorithm researchers to hack on.

**Performance:** Good single-machine performance with EnvPool and Numba optimizations.

---

### 1.6 RLax (DeepMind)

**Repository:** https://github.com/google-deepmind/rlax
**Stars:** ~1.3k
**Framework:** JAX
**Last Updated:** Maintained but infrequent updates
**License:** Apache 2.0

**What it does:** RLax (pronounced "relax") is a library of RL-specific mathematical building blocks in JAX. It does NOT provide complete algorithms -- instead, it provides composable operations (TD-learning, policy gradients, distributional value functions, etc.) that you combine to build agents. Acme is an example of a full framework built on top of RLax.

**Key Features:**
- Building blocks, not complete algorithms
- On-policy and off-policy learning primitives
- JIT compilation for CPU, GPU, TPU via `jax.jit`
- Batch support via `jax.vmap`
- Covers: TD-learning, policy gradients, actor-critics, MAP, PPO, distributional RL, general value functions, exploration methods

**Supported Components:** TD(lambda), Q-learning, SARSA, Retrace, V-trace, policy gradient losses (continuous and discrete), distributional RL (C51, QR-DQN), MPO losses, PPO clipping, and more.

**Distributed Training:** Not directly -- RLax provides math operations. Distribution is handled by the framework built on top (e.g., Acme + Launchpad, or Mava).

**Multi-Agent Support:** Not directly.

**Imitation/Behavioral Cloning:** Not directly, but you can compose BC losses using the primitives.

**Pros for Competitive Bot Programming:**
- Maximum flexibility -- build exactly the algorithm you need
- JAX ecosystem: hardware-accelerated, composable
- Used internally at DeepMind for cutting-edge research
- Clean mathematical abstractions

**Cons for Competitive Bot Programming:**
- Very low-level -- you build everything yourself
- Steep learning curve (need to know JAX well)
- No ready-to-use training loops or environments
- Small community compared to PyTorch-based alternatives
- Must combine with other libraries (Haiku/Flax, Optax, Acme) for a complete system

**Customization:** Maximum. You are composing from primitive operations.

**Performance:** Excellent when JIT-compiled. Benefits from JAX's TPU/GPU optimization.

---

### 1.7 Acme (DeepMind)

**Repository:** https://github.com/google-deepmind/acme
**Stars:** ~3.9k
**Framework:** JAX (primary), TensorFlow (legacy)
**Last Updated:** Last release v0.4.0 (Feb 2022); repo not archived but low activity
**License:** Apache 2.0

**What it does:** Acme is a framework for building readable, efficient, research-oriented RL agents that can run at various scales -- from single-process to highly distributed. It separates "acting" and "learning" into modular components that can be parallelized independently.

**Key Features:**
- Modular acting/learning separation enabling same code to run locally or distributed
- Uses Reverb as a purpose-built data storage system for experience replay
- Asynchronous distributed execution with rate limiting
- Reference implementations of D4PG, MPO, DQN, R2D2, IMPALA, PPO, SAC, and more
- Both TensorFlow (Sonnet) and JAX (Haiku) backends
- Tight integration with Launchpad for distributed execution (though Launchpad is now archived)

**Supported Algorithms:** D4PG, MPO, DQN, R2D2, IMPALA, PPO, SAC, BC, CRR, TD3, DMPO, MCTS (AlphaZero-style), and more.

**Distributed Training:** First-class via Launchpad (now archived as of Oct 2024). Designed for actor-learner architectures with multiple parallel actors.

**Multi-Agent Support:** Not a primary focus.

**Imitation/Behavioral Cloning:** BC and CRR (Critic Regularized Regression) for offline RL are included.

**Pros for Competitive Bot Programming:**
- Clean, readable code by DeepMind researchers
- Strong distributed training design
- AlphaZero-style MCTS implementation
- Good reference for understanding state-of-the-art algorithms

**Cons for Competitive Bot Programming:**
- Effectively in maintenance mode (last release 2022)
- Launchpad (its distribution backbone) is archived
- JAX ecosystem requires learning Haiku/Sonnet, Reverb, etc.
- Fewer community resources than PyTorch alternatives
- Not recommended for new projects given the maintenance status

**Customization:** High for researchers comfortable with the DeepMind JAX ecosystem.

**Performance:** Excellent at scale when properly distributed.

---

### 1.8 PFRL (Preferred Networks)

**Repository:** https://github.com/pfnet/pfrl
**Stars:** ~1.3k
**Framework:** PyTorch
**Last Updated:** v0.4.0 (July 2023); low activity
**License:** MIT

**What it does:** PFRL is the PyTorch successor to ChainerRL, providing comprehensive deep RL algorithm implementations with a focus on reproducibility. It includes reproducibility scripts that closely match original paper settings.

**Supported Algorithms:** DQN, Double DQN, Dueling DQN, PAL, Categorical DQN, IQN, A2C, A3C, ACER, PPO, TRPO, DDPG, TD3, SAC, REINFORCE.

**Key Features:**
- Reproducibility scripts for 9 key algorithms matching original papers
- Parallel environments (BatchAgent, AsyncAgent)
- Modular components (replay buffers, exploration, architectures)
- Optuna integration for hyperparameter search
- Well-benchmarked against Atari and MuJoCo baselines

**Distributed Training:** A3C-style async parallelism. No multi-node distribution.

**Multi-Agent Support:** No.

**Imitation/Behavioral Cloning:** No built-in support.

**Pros for Competitive Bot Programming:**
- Good selection of DQN variants (useful for discrete-action games)
- Reproducibility focus means reliable implementations
- Clean PyTorch code

**Cons for Competitive Bot Programming:**
- Low maintenance activity since 2023
- No multi-agent support
- No distributed training beyond A3C
- Smaller community; successor to a deprecated framework (Chainer)
- Missing recent algorithmic advances

**Customization:** Moderate. Modular design with reusable components.

**Performance:** Decent single-machine performance.

---

### 1.9 Dopamine (Google)

**Repository:** https://github.com/google/dopamine
**Stars:** ~10.5k
**Framework:** JAX (primary), TensorFlow (legacy)
**Last Updated:** Intermittent updates
**License:** Apache 2.0

**What it does:** Dopamine is a research framework for fast prototyping of RL algorithms, designed around easy experimentation, compact codebase, and reproducibility. Intentionally minimalist -- flat class hierarchy, no abstract base class.

**Supported Algorithms:** DQN, Rainbow, C51, IQN, SAC, with Prioritized Experience Replay.

**Key Features:**
- Minimalist design for rapid prototyping
- Gin-based configuration for all hyperparameters
- Colab notebooks for visualization and agent creation
- Checkpointing system for experiment resumption
- Primarily targets ALE (Atari) environments

**Distributed Training:** No.

**Multi-Agent Support:** No.

**Imitation/Behavioral Cloning:** No.

**Pros for Competitive Bot Programming:**
- Very clean codebase for learning and modifying
- Good DQN/Rainbow implementations for discrete-action games
- JAX support for fast experimentation

**Cons for Competitive Bot Programming:**
- Very limited algorithm scope
- Primarily Atari-focused; limited environment support
- No multi-agent or distributed capabilities
- Not suitable for complex competitive game environments
- Low priority for production use

**Customization:** Moderate. Easy to subclass DQN or create agents from scratch.

**Performance:** Good for what it does (Atari benchmarks).

---

### 1.10 Pearl (Meta)

**Repository:** https://github.com/facebookresearch/Pearl
**Stars:** ~2.5k
**Framework:** PyTorch
**Last Updated:** Active development (JMLR 2024 publication)
**License:** MIT

**What it does:** Pearl is Meta's production-ready RL agent library, the official successor to ReAgent/Horizon (now archived). It is designed for real-world production challenges including dynamic action spaces, partial observability, safety, and intelligent exploration.

**Supported Algorithms:** DQN, Bootstrapped DQN, SAC, DDPG, TD3, PPO, REINFORCE, CQL, IQL, TD3BC (offline RL), LinUCB, Neural-LinUCB, LinTS (bandits).

**Key Features:**
- Modular design: mix and match components for custom agents
- Dynamic action spaces (critical for many games)
- Intelligent neural exploration (contextual bandits + full RL)
- Safety-constrained decision making
- History summarization for partial observability
- Data augmentation for improved sample efficiency

**Distributed Training:** Designed for production deployment at Meta scale.

**Multi-Agent Support:** Not a primary focus.

**Imitation/Behavioral Cloning:** Offline RL support (CQL, IQL, TD3BC).

**Pros for Competitive Bot Programming:**
- Dynamic action spaces are very useful for games with varying legal moves
- Partial observability handling (history summarization) for imperfect information games
- Offline RL for learning from recorded games
- Active maintenance by Meta

**Cons for Competitive Bot Programming:**
- Relatively new; smaller community
- Less focus on game-specific features
- No multi-agent support
- Production-oriented design may add complexity for research

---

### 1.11 ElegantRL

**Repository:** https://github.com/AI4Finance-Foundation/ElegantRL
**Stars:** ~3.5k
**Framework:** PyTorch
**License:** Apache 2.0

**What it does:** ElegantRL is a massively parallel DRL library designed to scale from a single GPU to cloud platforms with thousands of GPUs. Core codes are under 1,000 lines (ElegantRL_Helloworld).

**Supported Algorithms:** DQN, Double DQN, D3QN, DDPG, TD3, SAC, PPO, A2C, REDQ (single-agent); QMIX, VDN, MADDPG, MAPPO, MATD3 (multi-agent).

**Key Features:**
- Cloud-native: containerization, microservices, MLOps
- ElegantRL-Podracer for DGX SuperPOD scaling
- Lightweight core (<1000 lines)
- Actor-Critic framework with easy customization
- Ensemble methods for stability

**Distributed Training:** Yes -- designed for massively parallel training across GPU clusters via Podracer architecture.

**Multi-Agent Support:** Yes -- QMIX, VDN, MADDPG, MAPPO, MATD3.

**Imitation/Behavioral Cloning:** No built-in support.

**Pros for Competitive Bot Programming:**
- Multi-agent algorithms built in
- Cloud-scalable for large training runs
- Lightweight and understandable codebase
- Claims better efficiency than RLlib in some benchmarks

**Cons for Competitive Bot Programming:**
- Primarily oriented toward finance applications
- Smaller community and documentation
- Less battle-tested than RLlib
- GPU-centric (less suited for CPU-heavy game simulations)

---

### 1.12 RLtools

**Repository:** https://github.com/rl-tools/rl-tools
**Framework:** Pure C++ (header-only), Python bindings available
**Published:** JMLR 2024
**License:** MIT

**What it does:** The fastest deep RL library, implemented as a dependency-free, header-only C++ library. Solves popular RL problems up to 76x faster than other frameworks. Enables deployment on microcontrollers, smartphones, and browsers (via WASM).

**Supported Algorithms:** TD3, PPO, SAC.

**Key Features:**
- Zero dependencies; compiles for any platform (CPU, GPU, embedded, WASM)
- Training on microcontrollers (ESP32, Teensy, PX4)
- Browser execution via WebAssembly
- Python bindings via PyPI

**Pros for Competitive Bot Programming:**
- Extremely fast training for continuous control
- Deploy trained policies anywhere (even embedded)
- Good for latency-critical inference

**Cons for Competitive Bot Programming:**
- Limited to continuous control (TD3/PPO/SAC)
- No discrete action support
- No multi-agent
- C++ focus requires more development effort
- Limited environment support

---

## 2. Distributed RL Training Frameworks

### 2.1 Sample Factory

**Repository:** https://github.com/alex-petrenko/sample-factory
**Stars:** ~1.0k+
**Framework:** PyTorch
**Published:** ICML 2020
**License:** MIT

**What it does:** Sample Factory is a high-throughput RL training system optimized for single-machine performance, achieving 100k+ environment frames/second on a single GPU machine. It proves that massively parallel training does not require expensive distributed clusters.

**Architecture:**
- Rollout Workers: step environments in parallel
- Inference Workers: generate actions on GPU
- Batcher: aggregates trajectories into training datasets
- Learner: updates model parameters
- All components communicate asynchronously via messages

**Key Features:**
- Single- and multi-agent training
- Self-play with population-based training (PBT)
- Multiple policies on one or many GPUs
- Automatic model architecture from action/observation spaces
- Custom environments as first-class citizens
- LSTM/GRU memory support
- Max-entropy objective option
- Deep integration with VizDoom, DMLab, Mujoco, Atari

**Performance:** 100k+ FPS on non-trivial 3D environments (VizDoom). 4x faster than comparable baselines. Agents trained on billions of transitions in hours. Population of 8 agents beats highest-difficulty VizDoom bots 100% of the time using PBT on a 36-core/4-GPU machine.

**Distributed Training:** Optimized for single-machine, multi-core, multi-GPU. Not designed for multi-node clusters (but can serve as a node in a larger system).

**Multi-Agent Support:** Yes -- treats all environments as multi-agent. Supports multiple policies, self-play, and PBT.

**Imitation/Behavioral Cloning:** Not built-in.

**Pros for Competitive Bot Programming:**
- Best single-machine throughput for PPO-based training
- Self-play and PBT are ideal for competitive games
- Proven on FPS games (VizDoom deathmatch)
- Reasonable to set up on a single powerful machine
- Custom environments are easy to integrate

**Cons for Competitive Bot Programming:**
- PPO-only (no DQN, SAC, etc.)
- Single-machine limitation for very large-scale training
- Less flexibility in algorithm choice compared to RLlib or SB3
- Smaller community

---

### 2.2 EnvPool

**Repository:** https://github.com/sail-sg/envpool
**Stars:** ~1.1k+
**Framework:** C++ backend with Python frontend; supports JAX, PyTorch, NumPy
**License:** Apache 2.0

**What it does:** EnvPool is a C++-based high-performance parallel environment execution engine. It replaces Python's subprocess-based vectorization with a C++ thread pool, achieving dramatic speedups.

**Performance Benchmarks:**
- ~1M FPS on Atari, ~3M FPS on MuJoCo (DGX-A100, 256 cores)
- 14.9x / 19.6x faster than gym.vector_env on high-end hardware
- 3.1x / 2.9x faster on a typical 12-core PC
- Train Atari Pong and MuJoCo Ant in 5 minutes on a laptop

**Key Features:**
- Synchronous and asynchronous execution modes
- Lock-free circular buffers for zero-copy data coordination
- Pre-allocated batch memory (no Python copying overhead)
- Auto-reset (no manual reset on episode end)
- JAX JIT support via XLA custom calls
- Gymnasium and dm_env API compatibility
- Drop-in replacement for vectorized environments in CleanRL, RLlib, Acme, etc.

**Supported Environments:** Atari, MuJoCo, Classic Control, ViZDoom, Box2D (built-in). Extensible in C++ for custom environments.

**Pros for Competitive Bot Programming:**
- Massive speedup for environment stepping (the typical bottleneck)
- Compatible with most RL libraries
- JAX JIT support for end-to-end acceleration

**Cons for Competitive Bot Programming:**
- Adding custom environments requires C++ implementation
- Linux-only for full features (limited macOS/Windows)
- Not all environments are supported
- Not actively maintained (last release 0.8.4)

---

### 2.3 Podracer Architectures (Anakin & Sebulba)

**Paper:** "Podracer architectures for scalable Reinforcement Learning" (DeepMind, 2021) -- https://arxiv.org/abs/2104.06272

These are architectural patterns, not specific libraries, but they have been adopted by multiple JAX-based frameworks.

**Anakin Architecture:**
- Environment, action selection, and learning ALL run on accelerators (TPU/GPU)
- Requires JAX-native environments
- End-to-end JIT compilation of the full training loop
- Computation vectorized via `vmap` and distributed via `pmap`
- Performance: 5M steps/sec on grid-worlds, 3M+ steps/sec on complex environments (16-core TPU)
- Trade-off: restricted to JAX environments, but maximum performance

**Sebulba Architecture:**
- Environments run on CPU hosts, inference and learning on TPU/GPU
- Supports arbitrary (non-JAX) environments
- Splits TPU cores into acting and learning subsets
- Python threads batch environment observations for GPU inference
- Performance: 200M-frame Atari training in 1 hour (8-core TPU)
- Trade-off: more flexible but less end-to-end optimized

**Libraries implementing these architectures:**
- **Mava** (InstaDeep) -- multi-agent RL, both Anakin and Sebulba
- **Stoix** -- single-agent RL, both Anakin and Sebulba
- **Earl** -- both Anakin (Gymnax) and Sebulba (Gymnasium)
- **PureJaxRL** -- Anakin-style end-to-end JAX training

---

### 2.4 Launchpad (DeepMind) -- ARCHIVED

**Repository:** https://github.com/google-deepmind/launchpad
**Stars:** ~330
**Status:** Archived (Oct 25, 2024) -- read-only
**License:** Apache 2.0

**What it did:** Launchpad was a programming model for defining and launching distributed RL systems. It represented distributed systems as directed graphs where nodes are services and edges are gRPC-based communication channels.

**Key Features (historical):**
- Graph-based program representation (Program data structure)
- Handle-based communication via gRPC (CourierNode)
- Platform-agnostic launching (local, cloud, cluster with a flag change)
- Three-phase lifecycle: setup, launch, execution
- Used as the distribution backbone for Acme

**Why it matters:** Launchpad's design influenced how distributed RL systems are architectured. Its archival status means that new projects should use alternatives like Ray for distribution.

---

## 3. Multi-Agent RL Frameworks

### 3.1 PettingZoo (Farama Foundation)

**Repository:** https://github.com/Farama-Foundation/PettingZoo
**Stars:** ~2.7k+
**Framework:** Framework-agnostic (Python environment API)
**Last Updated:** Actively maintained
**License:** MIT

**What it does:** PettingZoo is THE standard API for multi-agent RL environments, analogous to what Gymnasium is for single-agent. It provides two APIs: AEC (Agent Environment Cycle) for turn-based games, and Parallel API for simultaneous-action games.

**Environment Families:**
- **Atari** -- multi-player Atari 2600 (cooperative, competitive, mixed)
- **Butterfly** -- cooperative graphical games
- **Classic** -- card games, board games (chess, go, connect four, etc.)
- **MPE** -- Multi-Particle Environments (communication tasks)
- **SISL** -- 3 cooperative environments

**Key Features:**
- AEC API for sequential/turn-based games
- Parallel API for simultaneous-move games
- Strict versioning for reproducibility
- SuperSuit companion library for wrappers (frame stacking, normalization, etc.)
- Wide ecosystem integration: RLlib, Sample Factory, TorchRL, CleanRL, Tianshou, AgileRL

**Pros for Competitive Bot Programming:**
- De facto standard for multi-agent environments
- Classic game environments (chess, go, connect four) ready to use
- Integrates with every major RL library
- AEC API is perfect for turn-based competitive games
- Active maintenance by Farama Foundation

**Cons for Competitive Bot Programming:**
- Environment library only -- no training algorithms
- Performance limited by Python for >10k agents
- MPE environments being moved to separate package
- Some environment families are simple/toy-level

---

### 3.2 OpenSpiel (DeepMind)

**Repository:** https://github.com/google-deepmind/open_spiel
**Stars:** ~4.2k
**Framework:** C++ core with Python bindings; also supports Julia and Go
**Last Updated:** Active; preparing for v2.0 release with Kaggle Game Arena
**License:** Apache 2.0

**What it does:** OpenSpiel is a comprehensive collection of environments and algorithms for research in game theory, RL, and search/planning. It is the most complete framework for game AI research, supporting the widest variety of game types.

**Game Support:**
- N-player zero-sum, cooperative, and general-sum games
- Turn-taking and simultaneous-move games
- Perfect and imperfect information games
- One-shot (normal-form) and sequential (extensive-form) games
- Board games (Chess, Go, Hex), card games (poker variants), auction games
- Grid worlds and social dilemmas

**Algorithms:** CFR (Counterfactual Regret Minimization) and variants, MCTS (Monte Carlo Tree Search), REINFORCE, Q-learning, policy gradient methods, alpha-beta search, Deep CFR, NFSP (Neural Fictitious Self-Play), alpha-Rank, and more.

**Key Features:**
- C++ core for performance with Python, Julia, Go bindings
- Analysis tools for learning dynamics and evaluation metrics
- Alpha-Rank for ranking agents in multiplayer games
- SpielViz interactive game viewer
- Preparing for OpenSpiel 2.0 with Kaggle integration

**Distributed Training:** Not built-in.

**Imitation/Behavioral Cloning:** Not directly, but supports learning from game data via offline algorithms.

**Pros for Competitive Bot Programming:**
- THE best framework for game AI research
- CFR and MCTS implementations (critical for imperfect information games and tree-search games)
- Widest game type support of any framework
- C++ core for performance
- Alpha-Rank for evaluating agents
- Active development toward v2.0

**Cons for Competitive Bot Programming:**
- Research-focused; not optimized for production training speed
- Adding custom environments requires C++ (though Python wrappers exist)
- Limited deep RL algorithm selection compared to RLlib/SB3
- Linux/macOS primarily (limited Windows)
- Learning curve for the full framework

---

### 3.3 MARLlib

**Repository:** https://github.com/Replicable-MARL/MARLlib
**Stars:** ~1.2k
**Framework:** PyTorch (via Ray/RLlib)
**Published:** JMLR 2023
**License:** MIT

**What it does:** MARLlib is the most comprehensive MARL library, supporting 17+ environments, 18 algorithms, all task modes (cooperative, collaborative, competitive, mixed), and flexible parameter sharing strategies. Built on Ray/RLlib for scalability.

**Supported Algorithms:** Independent learners (IQL, IA2C, IPPO, ITRPO, etc.), value decomposition (VDN, QMIX, FACMAC), centralized critics (MADDPG, MATRPO, MAPPO, MAAC, HAPPO), and more.

**Key Features:**
- All task modes: cooperative, competitive, collaborative, mixed
- Flexible parameter sharing: share, group, separate, customizable
- Multiple architectures: MLP, CNN, GRU, LSTM
- 17+ environments: SMAC, MPE, MAMuJoCo, PettingZoo, Google Football, Hanabi, etc.
- Asynchronous and synchronous sampling
- Built on Ray/RLlib for distributed execution

**Pros for Competitive Bot Programming:**
- Most comprehensive MARL algorithm selection
- Supports competitive game modes natively
- Flexible parameter sharing for heterogeneous agents
- RLlib backend provides distributed scaling

**Cons for Competitive Bot Programming:**
- RLlib dependency adds complexity
- Higher memory usage than alternatives
- RLlib's vectorized environment limitations
- Documentation could be better

---

### 3.4 PyMARL / PyMARL2 / EPyMARL

**Repositories:**
- PyMARL: https://github.com/oxwhirl/pymarl (~800 stars)
- PyMARL2: https://github.com/hijkzzz/pymarl2
- EPyMARL: https://github.com/uoe-agents/epymarl

**Framework:** PyTorch
**Status:** PyMARL is unmaintained; EPyMARL is more active

**What they do:** The PyMARL family are MARL codebases primarily targeting SMAC (StarCraft Multi-Agent Challenge) for cooperative multi-agent learning.

| Feature | PyMARL | PyMARL2 | EPyMARL |
|---|---|---|---|
| Environments | 1 (SMAC) | 2 | 4 (SMAC, LBF, RWARE, MPE) |
| Algorithms | 5 (IQL, COMA, VDN, QMIX, QTRAN) | 11 | 9 (adds IA2C, IPPO, MADDPG, MAPPO, MAA2C) |
| Task modes | Cooperative only | Cooperative only | Cooperative only |
| Action spaces | Discrete only | Discrete only | Discrete only |
| Parameter sharing | Full only | Full only | Full + Separate |

**Pros for Competitive Bot Programming:**
- Clean reference implementations for value decomposition methods (QMIX, VDN)
- EPyMARL adds Gym compatibility and policy gradient methods

**Cons for Competitive Bot Programming:**
- Cooperative-only (no competitive game support)
- Discrete actions only
- Limited environment support
- Outdated (PyMARL) or restricted scope

---

### 3.5 JaxMARL

**Repository:** https://github.com/FLAIROx/JaxMARL
**Stars:** ~519
**Framework:** JAX
**Published:** NeurIPS 2024 (Datasets & Benchmarks)
**License:** Apache 2.0

**What it does:** JaxMARL combines GPU-enabled efficiency with support for many MARL environments and baseline algorithms, all in JAX. Includes SMAX (a vectorized JAX version of SMAC that does not require the StarCraft II engine) and STORM (matrix games on grid worlds).

**Performance:** Up to 12,500x faster than existing approaches in wall-clock time.

**Key Features:**
- End-to-end JAX (environments + algorithms on GPU)
- SMAX: JAX-native SMAC replacement
- Single-file implementations (CleanRL philosophy)
- PettingZoo/Gymnax-inspired interface
- Environments: MPE, SMAX, Overcooked, Hanabi, STORM, Switch Riddle, Coin Game, Multi-Agent Brax

**Algorithms:** IPPO, MAPPO, QMIX, VDN, IQL, SHAQ, TransfQMix.

**Pros for Competitive Bot Programming:**
- Extremely fast MARL training via JAX
- SMAX removes StarCraft II dependency
- Good for rapid prototyping of MARL ideas
- Hanabi support (imperfect information)

**Cons for Competitive Bot Programming:**
- JAX learning curve
- Smaller algorithm selection than MARLlib
- Custom environments must be JAX-native for full speed
- Newer library with smaller community

---

### 3.6 Mava (InstaDeep)

**Repository:** https://github.com/instadeepai/Mava
**Stars:** ~770
**Framework:** JAX (Haiku/Flax + Optax)
**Last Updated:** Feb 2025
**License:** Apache 2.0

**What it does:** Mava is a research-friendly codebase for fast experimentation in multi-agent RL using JAX. It implements both Anakin and Sebulba architectures for scalable training, achieving 10-100x speedups over other MARL frameworks.

**Algorithms:** IPPO, MAPPO, MADDPG, QMIX, VDN, and more (on- and off-policy, IL and CTDE paradigms).

**Key Features:**
- Anakin and Sebulba architecture support
- 10-100x speed advantage over competing MARL frameworks
- Hydra configuration management
- Flashbax accelerated replay buffers
- OG-MARL companion for offline MARL
- MARL-eval for statistically robust evaluation
- Jumanji environment integration

**Pros for Competitive Bot Programming:**
- Extremely fast MARL training
- Both Anakin (JAX envs) and Sebulba (any env) architectures
- Active development by InstaDeep

**Cons for Competitive Bot Programming:**
- JAX-only (smaller ecosystem than PyTorch)
- Not meant to be installed as a library (research tool)
- Limited to Python 3.11+
- Smaller community

---

### 3.7 Unity ML-Agents

**Repository:** https://github.com/Unity-Technologies/ml-agents
**Stars:** ~17.5k
**Framework:** PyTorch (Python training), Unity (C# environments)
**Latest Version:** Package v3.0
**License:** Apache 2.0

**What it does:** The Unity ML-Agents Toolkit enables Unity games and simulations to serve as environments for training RL and imitation learning agents. It bridges the gap between game development and ML research.

**Algorithms:** PPO, SAC, MA-POCA (multi-agent), self-play, BC (behavioral cloning), GAIL.

**Key Features:**
- 2D, 3D, and VR/AR environment support
- Single-agent, multi-agent cooperative, and competitive scenarios
- Curriculum learning
- Environment randomization
- On-demand decision making
- Imitation learning (BC + GAIL)
- Inference via Unity Sentis engine
- Self-play for competitive training

**Pros for Competitive Bot Programming:**
- Rich 3D environments with physics
- Built-in self-play for competitive games
- Imitation learning (BC + GAIL) for bootstrapping from demonstrations
- MA-POCA for multi-agent cooperative/competitive training
- Huge community (17.5k stars)

**Cons for Competitive Bot Programming:**
- Requires Unity engine (heavy dependency)
- Limited algorithm selection (PPO/SAC only)
- Training only on desktop platforms (no mobile/web)
- Python-Unity communication overhead
- Not suitable for pure algorithm research

---

### 3.8 BenchMARL (Meta/PyTorch)

**Repository:** https://github.com/facebookresearch/BenchMARL
**Framework:** PyTorch (TorchRL backend)
**Published:** JMLR 2024, NeurIPS 2024
**License:** MIT

**What it does:** BenchMARL is the first MARL training library for standardized benchmarking across algorithms, models, and environments. Built on TorchRL, it provides reproducible, standardized MARL experiments.

**Algorithms:** Multi-agent extensions of PPO, DDPG, SAC, DQN (via TorchRL), plus QMIX, VDN, MADDPG, MAPPO, IQL.

**Key Features:**
- Vectorized simulation and training (torch.vmap over agents)
- Hydra configuration
- SLURM launcher for HPC clusters
- Compatible with marl-eval for evaluation
- Flexible agent grouping and parameter sharing
- VMAS, MPE, SMAC environments

**Pros for Competitive Bot Programming:**
- Standardized MARL benchmarking
- TorchRL backend (official PyTorch)
- Proven in real-world multi-robot deployment

**Cons for Competitive Bot Programming:**
- Benchmarking-focused (not a full training framework)
- Smaller algorithm selection than MARLlib
- Newer, smaller community

---

## 4. Environment Interfaces

### 4.1 Gymnasium (Farama Foundation)

**Repository:** https://github.com/Farama-Foundation/Gymnasium
**Stars:** ~7.8k
**Latest Version:** 1.2.3 (Dec 2025)
**Published:** NeurIPS 2025 (Datasets & Benchmarks, spotlight)
**License:** MIT

**What it does:** Gymnasium is THE standard API for single-agent RL environments (successor to OpenAI Gym). Every major RL library supports it.

**Key API Features:**
- `Env` base class with `reset()` and `step()` methods
- Clear `termination` vs. `truncation` separation
- `FuncEnv` for functional/JAX-compatible environments
- `VectorEnv` with sync, async, and vector_entry_point modes
- Configurable autoreset modes (next-step, same-step, disabled)
- Comprehensive wrapper system
- Strict versioning for reproducibility

**Environment Families:** Classic Control, Box2D, Toy Text, MuJoCo v5, and hundreds of third-party environments.

**Relevance for Competitive Bot Programming:** Essential. Any custom game environment you build should implement the Gymnasium API to ensure compatibility with all training frameworks.

---

### 4.2 PettingZoo (Farama Foundation)

(Detailed in Section 3.1 above)

The multi-agent equivalent of Gymnasium. Provides AEC API (turn-based) and Parallel API (simultaneous). Essential for multi-agent game environments.

---

### 4.3 dm_env (DeepMind)

**Repository:** https://github.com/google-deepmind/dm_env
**Framework:** Python (framework-agnostic)
**License:** Apache 2.0

**What it does:** DeepMind's environment interface for RL. Uses `TimeStep` namedtuples with step types (FIRST, MID, LAST) rather than Gymnasium's tuple returns.

**Key Components:**
- `dm_env.Environment` abstract base class
- `dm_env.TimeStep` container (step_type, reward, discount, observation)
- `dm_env.specs` for describing action/observation formats
- `dm_env.test_utils` for conformance testing

**Relevance:** Used by DeepMind's libraries (Acme, RLax). EnvPool supports dm_env interface. Less widely adopted than Gymnasium in the broader community.

---

### 4.4 Jumanji (InstaDeep)

**Repository:** https://github.com/instadeepai/jumanji
**Stars:** ~600+
**Framework:** JAX
**Published:** ICLR 2024
**License:** Apache 2.0

**What it does:** A diverse suite of 22+ scalable RL environments written in JAX, featuring combinatorial optimization problems (TSP, BinPack, JobShop), logic puzzles (RubiksCube, Game2048, Sudoku), and games (Snake, Sokoban).

**Key Features:**
- Fully JAX-native (jit, vmap, pmap compatible)
- Scalable difficulty
- Wrappers for Gymnasium, dm_env, Acme, SB3, RLlib
- Combinatorial optimization focus (NP-hard problems)
- A2C baseline agents included

**Relevance for Competitive Bot Programming:** Good for testing algorithms on complex discrete optimization problems. JAX acceleration enables rapid experimentation.

---

## 5. JAX-Based RL Ecosystem

The JAX RL ecosystem has grown significantly and deserves its own section. JAX enables hardware-accelerated (GPU/TPU) end-to-end RL training with automatic differentiation, vectorization, and JIT compilation.

### 5.1 Gymnax

**Repository:** https://github.com/RobertTLange/gymnax
**What it does:** JAX re-implementations of classic RL environments (Classic Control, bsuite, MinAtar, meta-RL tasks). Enables the Anakin architecture pattern.

### 5.2 Brax

**Repository:** https://github.com/google/brax
**What it does:** Differentiable physics engine in JAX. MuJoCo-like continuous control environments (HalfCheetah, Humanoid, Ant, etc.) running entirely on GPU/TPU.

### 5.3 Pgx

**Repository:** https://github.com/sotetsuk/pgx
**What it does:** Vectorized board game simulators in JAX. 20+ games including Chess, Shogi, Go, Backgammon, and imperfect information games like Bridge. 10-100x faster than PettingZoo/OpenSpiel Python implementations. PettingZoo AEC API compatibility.

**Highly relevant for competitive bot programming:** Fast board game simulation is directly applicable.

### 5.4 Stoix

**Repository:** https://github.com/EdanToledo/Stoix
**What it does:** Research-friendly single-agent RL in JAX with both Anakin and Sebulba architectures. Supports Gymnax, Jumanji, Brax, XMinigrid, Craftax, Pgx, and non-JAX environments via EnvPool/Gymnasium.

### 5.5 PureJaxRL

**What it does:** End-to-end JAX RL training (Anakin-style). Demonstrated 4000x speedups and meta-evolving discoveries. Inspired CleanRL's JAX variants and Stoix.

### 5.6 PufferLib

**Repository:** https://github.com/PufferAI/PufferLib
**Framework:** PyTorch (not JAX)
**License:** MIT

**What it does:** Makes RL libraries and environments "play nice." Provides one-line environment wrappers that eliminate compatibility problems and fast vectorization. Includes optimized PPO from CleanRL. Environments run at 1M+ steps/second.

**Highly relevant for competitive bot programming:** Designed to work with complex game simulators like NetHack and Neural MMO. Drop-in vectorization with 30-300% speedups.

---

## 6. Comparative Summary Tables

### 6.1 General RL Frameworks

| Framework | Stars | Backend | Algorithms | Distributed | Multi-Agent | Imitation/BC | Active |
|---|---|---|---|---|---|---|---|
| **RLlib** | ~41k (Ray) | PyTorch/TF | 15+ | Yes (native) | Yes (native) | MARWIL, BC | Yes |
| **SB3** | ~12.8k | PyTorch | 7+contrib | No | No | Via `imitation` lib | Yes |
| **CleanRL** | ~8.4k | PyTorch/JAX | 7 | No (cloud scripts) | Limited | No | Yes |
| **TorchRL** | ~2.7k | PyTorch | 10+ | Yes (PyTorch) | Via BenchMARL | CQL, IQL, DT | Yes |
| **Tianshou** | ~8.0k | PyTorch | 20+ | No | Experimental | GAIL, BCQ, CQL | Yes |
| **Acme** | ~3.9k | JAX/TF | 12+ | Yes (Launchpad) | No | BC, CRR | Low |
| **Dopamine** | ~10.5k | JAX/TF | 5 | No | No | No | Low |
| **Pearl** | ~2.5k | PyTorch | 12+ | Production | No | CQL, IQL, TD3BC | Yes |
| **ElegantRL** | ~3.5k | PyTorch | 13+ | Yes (Podracer) | Yes (5 algos) | No | Moderate |
| **PFRL** | ~1.3k | PyTorch | 18 | A3C only | No | No | Low |

### 6.2 Multi-Agent RL Frameworks

| Framework | Stars | Backend | Algorithms | Environments | Task Modes | Speed |
|---|---|---|---|---|---|---|
| **MARLlib** | ~1.2k | PyTorch/RLlib | 18 | 17+ | All (coop/comp/mixed) | Good |
| **EPyMARL** | ~600 | PyTorch | 9 | 4 | Cooperative only | Moderate |
| **JaxMARL** | ~519 | JAX | 7 | 8+ | Cooperative + some comp | 12,500x faster |
| **Mava** | ~770 | JAX | 6+ | Multiple | IL + CTDE | 10-100x faster |
| **BenchMARL** | ~400 | PyTorch/TorchRL | 8+ | VMAS/MPE/SMAC | All | Good |
| **ElegantRL** | ~3.5k | PyTorch | 5 (MARL) | Limited | Cooperative | GPU-scalable |

### 6.3 Environment Interfaces

| Interface | Stars | Type | Multi-Agent | JAX Native | Coverage |
|---|---|---|---|---|---|
| **Gymnasium** | ~7.8k | Single-agent API | No | Via FuncEnv | De facto standard |
| **PettingZoo** | ~2.7k | Multi-agent API | Yes (AEC + Parallel) | No | De facto standard for MARL |
| **dm_env** | ~400 | Single-agent API | No | No | DeepMind ecosystem |
| **Jumanji** | ~600 | JAX environments | No | Yes | 22 environments (CO, logic, games) |
| **Gymnax** | ~500 | JAX environments | No | Yes | Classic RL benchmarks |
| **Pgx** | ~400 | JAX board games | Yes (via PettingZoo) | Yes | 20+ board games |
| **Brax** | ~2.3k | JAX physics | No | Yes | Continuous control |
| **EnvPool** | ~1.1k | C++ vectorization | Limited | JAX JIT support | Atari, MuJoCo, Classic |

### 6.4 Imitation Learning / Behavioral Cloning Support

| Framework | BC Support | GAIL | DAgger | Offline RL | Notes |
|---|---|---|---|---|---|
| **SB3 + imitation** | Yes | Yes | Yes | No | Most complete IL library |
| **RLlib** | Yes (MARWIL, BC) | No | No | Yes | Production-grade |
| **Tianshou** | No | Yes | No | Yes (BCQ, CQL) | Good offline RL |
| **TorchRL** | No | No | No | Yes (CQL, IQL, DT) | Via D4RL datasets |
| **Pearl** | No | No | No | Yes (CQL, IQL, TD3BC) | Production offline RL |
| **Unity ML-Agents** | Yes | Yes | No | No | Game-focused IL |
| **Acme** | Yes | No | No | Yes (CRR) | Research-grade |
| **OpenSpiel** | No | No | No | No | Game theory focused |

---

## 7. Recommendations for Competitive Bot Programming

### 7.1 Best Choices by Scenario

**Scenario 1: Turn-based competitive games (chess-like, card games, board games)**
- **Primary:** OpenSpiel (CFR, MCTS, game theory algorithms)
- **Environments:** OpenSpiel built-in or Pgx (JAX-accelerated board games)
- **Alternative:** PettingZoo Classic environments + any RL framework
- **Key algorithms:** CFR for imperfect info, MCTS for perfect info, NFSP for learning from self-play

**Scenario 2: Real-time competitive games (first-person, strategy, arcade)**
- **Primary:** Sample Factory (highest throughput, self-play, PBT)
- **Alternative:** RLlib (more flexible, distributed) or PufferLib (compatibility layer)
- **Environments:** Custom Gymnasium environments
- **Key algorithms:** PPO with self-play and population-based training

**Scenario 3: Multi-agent competitive games (team vs team, mixed cooperative/competitive)**
- **Primary:** MARLlib (most comprehensive MARL algorithms, all task modes)
- **High-speed alternative:** JaxMARL or Mava (10-12,500x faster)
- **Environments:** PettingZoo for environment interface, SMAC/SMAX for benchmarks
- **Key algorithms:** MAPPO, QMIX, MADDPG

**Scenario 4: Rapid prototyping and understanding algorithms**
- **Primary:** CleanRL (read and modify single files)
- **Alternative:** SB3 (quick baselines) or Tianshou (broadest algorithm selection)

**Scenario 5: Training with recorded game data (behavioral cloning, offline RL)**
- **Primary:** SB3 + `imitation` library (BC, DAgger, GAIL, AIRL)
- **Alternative:** Tianshou (offline RL: BCQ, CQL) or Pearl (production offline RL)

**Scenario 6: Maximum training speed on limited hardware**
- **Single machine:** Sample Factory (100k+ FPS on one GPU machine)
- **JAX acceleration:** Stoix + Gymnax/Pgx/Jumanji (Anakin architecture)
- **Environment speed:** EnvPool for C++ environment vectorization

**Scenario 7: Large-scale distributed training**
- **Primary:** RLlib on Ray (industry standard for distributed RL)
- **Cloud-native:** ElegantRL-Podracer (designed for GPU clusters)
- **JAX:** Mava or Stoix with Sebulba architecture

### 7.2 Recommended Stack for a Competitive Bot Programming Project

For a project like the one in this repository (Colosseum -- multi-agent RL with custom environments), here is a recommended technology stack:

**Environment Layer:**
- Implement your game environment using the Gymnasium API (single-agent) or PettingZoo API (multi-agent)
- If targeting JAX acceleration, also implement a JAX-native version

**Training Layer (choose based on needs):**
1. **Start simple:** SB3 + self-play wrapper for initial experiments
2. **Scale up:** Sample Factory for high-throughput self-play training
3. **Multi-agent:** MARLlib or RLlib for multi-agent algorithms (MAPPO, QMIX)
4. **Maximum speed:** JaxMARL or Mava for JAX-accelerated MARL
5. **Game theory:** OpenSpiel if your game benefits from CFR/MCTS

**Bootstrapping Layer:**
- Use `imitation` library (on SB3) for behavioral cloning from expert replays
- Use Tianshou or Pearl for offline RL from recorded game data

**Performance Layer:**
- EnvPool for environment vectorization (if supported)
- PufferLib for compatibility and vectorization with complex environments
- Pgx for JAX-accelerated board game simulation

### 7.3 Key Takeaways

1. **There is no single best framework.** The choice depends on your game type, scale requirements, and whether you need multi-agent support.

2. **The JAX ecosystem is catching up fast.** JaxMARL, Mava, Stoix, and Pgx offer 10-12,500x speedups over Python-based alternatives. If your environment can be expressed in JAX, this is the performance frontier.

3. **Self-play and PBT are essential for competitive games.** Sample Factory and RLlib are the most mature options here.

4. **For imperfect information games, OpenSpiel is unmatched.** CFR and its variants are fundamentally better than pure RL for poker-like games.

5. **Behavioral cloning is underrated for competition.** If you can collect expert replays, bootstrapping from BC + fine-tuning with RL is often faster than training from scratch.

6. **The `imitation` library (SB3 ecosystem) provides the most complete imitation learning toolkit.** It supports BC, DAgger, GAIL, AIRL, and SQIL.

7. **PettingZoo is the lingua franca for multi-agent environments.** Build your environment to this API for maximum framework compatibility.

8. **Watch for:** Stoix, PufferLib, and Pearl as rising frameworks that address gaps in the current landscape.

---

## Sources and Links

### General RL Frameworks
- Ray RLlib: https://docs.ray.io/en/latest/rllib/index.html | https://github.com/ray-project/ray
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3 | https://stable-baselines3.readthedocs.io/
- CleanRL: https://github.com/vwxyzjn/cleanrl | https://docs.cleanrl.dev/
- TorchRL: https://github.com/pytorch/rl | https://docs.pytorch.org/rl/
- Tianshou: https://github.com/thu-ml/tianshou
- RLax: https://github.com/google-deepmind/rlax | https://rlax.readthedocs.io/
- Acme: https://github.com/google-deepmind/acme
- PFRL: https://github.com/pfnet/pfrl | https://pfrl.readthedocs.io/
- Dopamine: https://github.com/google/dopamine
- Pearl: https://github.com/facebookresearch/Pearl | https://pearlagent.github.io/
- ElegantRL: https://github.com/AI4Finance-Foundation/ElegantRL
- RLtools: https://github.com/rl-tools/rl-tools | https://rl.tools/

### Distributed RL
- Sample Factory: https://github.com/alex-petrenko/sample-factory | https://www.samplefactory.dev/
- EnvPool: https://github.com/sail-sg/envpool | https://envpool.readthedocs.io/
- Podracer paper: https://arxiv.org/abs/2104.06272
- Launchpad (archived): https://github.com/google-deepmind/launchpad

### Multi-Agent RL
- PettingZoo: https://github.com/Farama-Foundation/PettingZoo | https://pettingzoo.farama.org/
- OpenSpiel: https://github.com/google-deepmind/open_spiel
- MARLlib: https://github.com/Replicable-MARL/MARLlib
- EPyMARL: https://github.com/uoe-agents/epymarl
- JaxMARL: https://github.com/FLAIROx/JaxMARL
- Mava: https://github.com/instadeepai/Mava
- Unity ML-Agents: https://github.com/Unity-Technologies/ml-agents
- BenchMARL: https://github.com/facebookresearch/BenchMARL

### Environment Interfaces
- Gymnasium: https://github.com/Farama-Foundation/Gymnasium | https://gymnasium.farama.org/
- dm_env: https://github.com/google-deepmind/dm_env
- Jumanji: https://github.com/instadeepai/jumanji
- Gymnax: https://github.com/RobertTLange/gymnax
- Pgx: https://github.com/sotetsuk/pgx
- Brax: https://github.com/google/brax
- PufferLib: https://github.com/PufferAI/PufferLib
- Stoix: https://github.com/EdanToledo/Stoix

### Imitation Learning
- imitation library: https://imitation.readthedocs.io/ (built on SB3)

### Papers
- RLlib: Ray documentation
- SB3: JMLR Vol 22 (2021)
- CleanRL: JMLR Vol 23 (2022)
- Tianshou: JMLR (arXiv:2107.14171)
- Acme: arXiv:2006.00979
- OpenSpiel: arXiv:1908.09453
- PettingZoo: NeurIPS 2021
- Gymnasium: NeurIPS 2025
- MARLlib: JMLR 2023
- BenchMARL: JMLR 2024
- Pearl: JMLR 2024
- JaxMARL: NeurIPS 2024
- Sample Factory: ICML 2020
- EnvPool: NeurIPS 2022
- Podracer: arXiv:2104.06272
- RLtools: JMLR 2024
- Jumanji: ICLR 2024

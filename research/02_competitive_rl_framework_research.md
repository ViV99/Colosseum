# Competitive Bot Programming with Reinforcement Learning: Comprehensive Research Survey

**Date:** 2026-03-18
**Purpose:** Deep research document for building a reusable RL training framework targeting competitive programming bot competitions (Lux AI, Neural MMO, CodeCraft, etc.)

---

## Table of Contents

1. [Multi-Agent RL Systems for Competitive Games](#1-multi-agent-rl-systems-for-competitive-games)
   - [AlphaStar (DeepMind, StarCraft II)](#11-alphastar-deepmind-starcraft-ii)
   - [OpenAI Five (Dota 2)](#12-openai-five-dota-2)
   - [AlphaGo / AlphaZero / MuZero](#13-alphago--alphazero--muzero)
   - [Cicero (Meta, Diplomacy)](#14-cicero-meta-diplomacy)
   - [Honor of Kings AI (Tencent)](#15-honor-of-kings-ai-tencent)
   - [Pluribus (Poker, Facebook AI)](#16-pluribus-poker)
2. [Matchmaking Systems for RL Training](#2-matchmaking-systems-for-rl-training)
   - [League Training (AlphaStar Style)](#21-league-training-alphastar-style)
   - [Population-Based Training (PBT)](#22-population-based-training-pbt)
   - [Self-Play Variants: FSP, PFSP, PSRO](#23-self-play-variants-fsp-pfsp-psro)
   - [ELO and TrueSkill Systems](#24-elo-and-trueskill-systems)
   - [Opponent Sampling Strategies](#25-opponent-sampling-strategies)
3. [Behavioral Cloning and Imitation Learning](#3-behavioral-cloning-and-imitation-learning)
   - [Behavioral Cloning (BC)](#31-behavioral-cloning-bc)
   - [DAgger (Dataset Aggregation)](#32-dagger-dataset-aggregation)
   - [GAIL (Generative Adversarial Imitation Learning)](#33-gail-generative-adversarial-imitation-learning)
   - [Inverse Reinforcement Learning (IRL)](#34-inverse-reinforcement-learning-irl)
   - [Kickstarting RL from Demonstrations](#35-kickstarting-rl-from-demonstrations)
4. [N-Player Game Training](#4-n-player-game-training)
   - [Challenges of N-Player Games](#41-challenges-of-n-player-games)
   - [Nash Equilibrium in N-Player Settings](#42-nash-equilibrium-in-n-player-settings)
   - [Neural MMO (100+ Player)](#43-neural-mmo-100-player)
   - [Hanabi (Cooperative Imperfect Information)](#44-hanabi-cooperative-imperfect-information)
5. [Practical Frameworks](#5-practical-frameworks)
   - [OpenSpiel (Google DeepMind)](#51-openspiel-google-deepmind)
   - [PettingZoo (Farama Foundation)](#52-pettingzoo-farama-foundation)
   - [JaxMARL (FLAIROx)](#53-jaxmarl-flairox)
   - [Mava (InstaDeep)](#54-mava-instadeep)
   - [EPyMARL / PyMARL](#55-epymarl--pymarl)
   - [MARLlib (RLlib Extension)](#56-marllib-rllib-extension)
   - [Ray RLlib](#57-ray-rllib)
   - [Weights & Biases for RL Experiments](#58-weights--biases-for-rl-experiments)
   - [Optuna / Ray Tune for Hyperparameter Optimization](#59-optuna--ray-tune-for-hyperparameter-optimization)
   - [Imitation Learning Library](#510-imitation-learning-library)
6. [Key Algorithms Reference](#6-key-algorithms-reference)
   - [MAPPO](#61-mappo)
   - [QMIX](#62-qmix)
   - [MADDPG](#63-maddpg)
   - [HAPPO](#64-happo)
7. [Recent Advances (2023–2025)](#7-recent-advances-20232025)
8. [Practical Recommendations](#8-practical-recommendations)
9. [Framework Selection Matrix](#9-framework-selection-matrix)
10. [References and Links](#10-references-and-links)

---

## 1. Multi-Agent RL Systems for Competitive Games

### 1.1 AlphaStar (DeepMind, StarCraft II)

**Overview:** The first AI to reach Grandmaster level in StarCraft II, ranking in the top 0.2% of human players across all three races (Protoss, Terran, Zerg).

**Architecture:**

Each race agent is a single neural network. The policy is conditioned on a statistic `z` that summarizes a strategy sampled from human replay data (e.g., a build order), enabling diverse strategies. The architecture uses:
- A transformer-based encoder for entities (units, buildings)
- A deep LSTM for temporal reasoning
- A pointer-network-style action head to handle the massive structured action space
- Auto-regressive action decomposition: `P(action) = P(type) * P(delay | type) * P(unit | type, delay) * ...`

The action space has up to 10^26 possible actions per timestep.

**Training Pipeline:**

1. **Supervised Learning (SL) / Behavioral Cloning:** The initial policy is trained via KL divergence minimization against human replay data. This yields an agent that beats ~84% of active players before any RL.
2. **Reinforcement Learning:** Uses a combination of:
   - **V-trace** (importance-weighted actor-critic for off-policy correction)
   - **TD(λ)** for value estimation
   - **UPGO** (Upgoing Policy Gradient for trajectory optimization)
   - **KL loss** toward the supervised baseline to prevent forgetting

**League Training:**

Three agent types maintain a diverse population:

| Type | Role | Opponent Selection |
|------|------|-------------------|
| **Main Agent (MA)** | Primary agent to be improved | 35% self-play + 50% PFSP vs all past players + 15% PFSP vs forgotten mains |
| **Main Exploiter (ME)** | Finds weaknesses in the Main Agent | Trains against Main Agent only; periodic reset |
| **League Exploiter (LE)** | Finds systemic weaknesses across the whole league | PFSP across all agents; added to league if >70% win rate vs all; 25% chance reset to SL weights |

The league is designed to prevent **forgetting cycles**: without exploiters, an MA may learn to beat its current self but forget how to beat older strategies, creating cyclical training.

**Key Innovations:**
- Combining behavioral cloning with league training
- Z-conditioning for strategy diversity
- Three-tier agent population (MA/ME/LE)
- PFSP for adaptive opponent selection

**Pros/Cons:**
- Pro: Achieves extreme strategic diversity; robust to exploitation
- Pro: Behavioral cloning init massively accelerates early training
- Con: Extremely compute-intensive (hundreds of TPUs, weeks of training)
- Con: League setup is complex to engineer

**When to Use:** When you have human replay data and need a highly robust, unexploitable agent.

**2023 Improvement — ROA-Star (NeurIPS 2023):** Introduced opponent-aware training, fixing a key weakness where AlphaStar could be defeated by uncommon strategies like Cannon Rush. ROA-Star achieved >50% win rate against top human players with significantly less compute.

**Code:** [github.com/google-deepmind/alphastar](https://github.com/google-deepmind/alphastar)

---

### 1.2 OpenAI Five (Dota 2)

**Overview:** The first AI to defeat world champions at a professional esports game, accomplished in 2019 after 10 months of training.

**Architecture:**

- **Input:** 20,000 numbers encoding the game state (units, items, cooldowns, vision)
- **Core:** A **4096-unit LSTM** (84% of total parameters) processing the observation vector
- **Output:** 8 independent action heads (action type, target unit, target location X/Y, etc.) computed independently (not auto-regressive)
- **Multi-hero handling:** A hero indicator embedded in the observation distinguishes which of the 5 heroes is being controlled

**Training System:**

Scaled PPO (Proximal Policy Optimization) running on:
- **256 GPUs** for gradient computation
- **128,000 CPU cores** for environment simulation

Four machine types in the distributed system:
1. **Controller:** Distributes updated policy parameters
2. **Rollout Worker CPUs:** Run game simulation, send observations to Forward Pass GPUs
3. **Forward Pass GPUs:** Compute actions from observations
4. **Optimizer GPUs:** Sample from experience buffer, compute gradient updates

By the time of the Finals, the system had consumed **~45,000 years** of simulated Dota experience.

**Reward Shaping:**

Sparse reward (win/loss) is supplemented with dense intermediate rewards:
- Experience gained
- Gold collected
- Last hits
- Staying alive

The final reward signal subtracts the opposing team's average reward to prevent positive-sum situations (e.g., mutual idling being rewarded).

**γ Annealing (Curriculum on Time Horizon):**
- Start: γ = 0.998 (half-life ≈ 46 seconds of game time)
- End: γ = 0.9997 (half-life ≈ 5 minutes of game time)

This curriculum teaches the agent to reason about increasingly long-term consequences.

**Self-Play:** Pure self-play (no league), evaluated against fixed reference agents using **TrueSkill** ratings throughout training.

**Key Innovations:**
- Massive-scale pure self-play (no human data needed)
- γ annealing for curriculum on temporal reasoning
- Reward shaping by team-relative scores

**Pros/Cons:**
- Pro: No human data required; simpler training setup than AlphaStar
- Pro: γ annealing is a simple, powerful curriculum trick
- Con: Pure self-play can be exploited by strategies not in the training distribution
- Con: Requires enormous compute for complex games

**When to Use:** When you do not have expert demonstrations but have massive compute for self-play, and the game has relatively dense intermediate signals.

**References:** [arxiv.org/abs/1912.06680](https://arxiv.org/abs/1912.06680), [openai.com/index/openai-five](https://openai.com/index/openai-five/)

---

### 1.3 AlphaGo / AlphaZero / MuZero

**Overview:** A family of systems that achieve superhuman performance in perfect-information board games through MCTS + neural networks + self-play. No domain-specific heuristics required.

**Core Training Loop:**

```
Self-Play:
  for each game step:
    run N MCTS simulations (e.g., 800)
    select move proportional to visit counts (π)
    record (state, π, outcome z)
  store complete game in replay buffer

Optimization:
  sample (state, π, z) from replay buffer
  minimize: loss = MSE(v, z) + CE(p, π) + L2_reg
  where v = predicted value, z = actual outcome
       p = predicted policy, π = MCTS improved policy
```

**Neural Network (AlphaZero):**
- **Input:** Board state (binary planes: current player's pieces, opponent's pieces, castling rights, etc.)
- **Body:** Deep residual network (20-40 blocks)
- **Policy head:** Softmax over legal moves (prior for MCTS)
- **Value head:** Scalar prediction of win probability

**MCTS (AlphaZero variant):**

Selection uses the PUCT formula:
```
a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)) ]
```
where:
- `Q(s,a)` = average value of actions taken at (s,a)
- `P(s,a)` = policy prior from neural network
- `N(s)` = parent visit count
- `N(s,a)` = child visit count

**AlphaZero vs MuZero:**

| Feature | AlphaZero | MuZero |
|---------|-----------|--------|
| Requires game rules | Yes | No (learns dynamics) |
| Planning model | Known simulator | Learned dynamics network |
| Applicable to | Board games | Board games + Atari + arbitrary |
| Key addition | — | Representation + Dynamics + Prediction networks |

MuZero's three networks:
- **Representation function h:** `h(observation) → hidden_state`
- **Dynamics function g:** `g(hidden_state, action) → next_hidden_state, reward`
- **Prediction function f:** `f(hidden_state) → policy, value`

**Key Innovations:**
- MCTS as a policy improvement operator (not just search)
- Value targets from actual game outcomes (no bootstrapping)
- MuZero's model-free planning applicable to any environment

**Gumbel MuZero (2022):** Achieves equivalent performance with as few as 2 MCTS simulations per step using Gumbel noise for policy improvement guarantees.

**Pros/Cons:**
- Pro: Extremely sample-efficient within simulation budget
- Pro: Proven to converge to superhuman level in 2-player zero-sum games
- Con: MCTS at inference is expensive (800 simulations per move is common)
- Con: Requires a fast, accurate simulator (or MuZero's learned model adds complexity)

**When to Use:** Perfect-information 2-player zero-sum games where a fast simulator is available.

**Code:** [github.com/rlglab/minizero](https://github.com/rlglab/minizero), [github.com/suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)

---

### 1.4 Cicero (Meta, Diplomacy)

**Overview:** First AI to achieve human-level play in Diplomacy, a 7-player board game requiring both strategic play and natural language negotiation. Published in Science (2022).

**Architecture:**

Cicero combines two distinct systems:
1. **Strategic Reasoning Module:** RL-trained value and policy models over the board state
2. **Language Module:** Fine-tuned BART (2.7B parameters) conditioned on game state and strategic intent

**Training Pipeline:**

1. **Behavioral cloning** on human Diplomacy game data to learn action prediction
2. **Joint RL planning** to compute action intentions for all players simultaneously, using:
   - A **dialogue-conditional action model**: `P(actions | messages, board_state)`
   - A **dialogue-free value model**: `V(board_state)`
   - A **KL penalty** to regularize planned actions toward human-like play
3. **Conditional dialogue model**: Trained by fine-tuning BART on `(intent, game_history) → message` pairs

**Multi-Agent Dynamics:**

Diplomacy has a unique **mixed cooperative-competitive structure**:
- Only one player wins, but winning requires building temporary alliances
- Alliances are non-binding, so deception is endemic
- All moves are simultaneous (no turn order)

Cicero's key insight: predict what moves other players will make given the current dialogue, then plan a strategy that is **mutually beneficial** with key allies while positioning against threats.

**Performance:** Ranked in top 10% of participants across 40 anonymous online games; sent ~130 messages per game.

**Key Innovations:**
- Integrating natural language negotiation with strategic RL
- Theory of mind: predicting other players' beliefs and intentions
- Largely honest play despite deception being an option (achieved via strategic regularization)

**When to Use:** N-player games where communication is part of the action space, or where modeling opponent intentions from observations is critical.

**Code:** [github.com/facebookresearch/diplomacy_cicero](https://github.com/facebookresearch/diplomacy_cicero)

---

### 1.5 Honor of Kings AI (Tencent)

**Overview:** Tencent's AI for Honor of Kings (HoK), the world's most popular MOBA game with 100M+ daily active players.

**Notable Contributions:**

- **HoK3v3 (NeurIPS 2023):** A 3v3 MARL environment with heterogeneous heroes, testing generalization across diverse lineups
- **Hokoff (NeurIPS 2023):** A real-game offline RL/MARL dataset benchmark based on HoK, far more complex than previous offline MARL benchmarks
- **Mini HoK (2024):** Lightweight open-source version for research on personal hardware
- **AI Arena (Kaiwu) Competition:** Regular competition using HoK as the environment

**Key Challenges Exposed:**
- Heterogeneous agents (different heroes with different skills)
- Long-horizon cooperative + competitive dynamics (5v5 full game)
- Partial observability via fog of war

**Open Source Resources:**
- [github.com/tencent-ailab/hok_env](https://github.com/tencent-ailab/hok_env)
- [github.com/tencent-ailab/mini-hok](https://github.com/tencent-ailab/mini-hok)
- [github.com/tencent-ailab/hokoff](https://github.com/tencent-ailab/hokoff)

---

### 1.6 Pluribus (Poker)

**Overview:** Meta AI's agent for 6-player no-limit Texas Hold'em poker, the first AI to defeat professional human players at this format (2019, published in Science).

**Method:** Modified Monte-Carlo Counterfactual Regret Minimization (MCCFR) + self-play.

In contrast to two-player settings, Pluribus does **not** attempt to find a Nash equilibrium (which is computationally intractable for 6 players). Instead, it uses:
1. **Blueprint strategy:** Pre-computed via MCCFR self-play offline
2. **Online search (limited depth):** At each decision point, Pluribus runs a limited-depth MCTS-style search using copies of itself as opponents

**Key Insight for N-player games:** Playing a Nash equilibrium in multi-player games is not guaranteed to be optimal; empirical performance against humans is more relevant than theoretical optimality.

**ReBeL (Meta, 2020):** Follow-up work that proves convergence to Nash equilibrium by combining RL with CFR search for 2-player imperfect-information games.

---

## 2. Matchmaking Systems for RL Training

### 2.1 League Training (AlphaStar Style)

**Overview:** Maintain a growing population of diverse agents, adding new agents when old ones stagnate, using matchmaking to select training opponents adaptively.

**Population Structure:**

```
League = {
  main_agents: [MA1, MA2, ...],       # Current primary agents
  main_exploiters: [ME1, ME2, ...],   # Find weaknesses in main agents
  league_exploiters: [LE1, LE2, ...], # Find systemic league weaknesses
  historical_agents: [h1, h2, ...]    # Frozen snapshots (never updated)
}
```

**Adding Agents to the League:**
- Exploiters are added to the historical pool when they achieve >70% win rate vs all current agents
- Both ME and LE have a probability (e.g., 25%) of being reset to supervised weights after being added, to encourage new strategies
- New MAs are added periodically as training progresses

**Key Properties:**
- The league accumulates diverse strategies over time
- PFSP ensures that training focuses on strategically relevant opponents
- Exploiters act as adversarial red-teamers, preventing stagnation

**When to Use:** When you need robust agents that cannot be beaten by specialized counter-strategies. Particularly valuable when the game has a large strategy space (rock-paper-scissors dynamics among strategies).

**2023 Improvement — ROA-Star:**
- More effective exploiters that detect weaknesses more reliably
- More comprehensive evaluation against top humans
- Achieves comparable performance with significantly less compute

---

### 2.2 Population-Based Training (PBT)

**Overview:** A hyperparameter optimization technique that runs a population of agents simultaneously, periodically copying high-performing agents' weights and hyperparameters to replace low-performing ones.

**Algorithm:**
```
Initialize population P = {agent_1, ..., agent_N} with random hyperparameters

Every T steps:
  for each agent_i in P:
    if is_low_performer(agent_i):
      agent_j = sample(high_performers(P))
      agent_i.weights = agent_j.weights
      agent_i.hyperparams = perturb(agent_j.hyperparams)
```

**Advantages for Competitive Games:**
- Automatically discovers optimal hyperparameters (learning rate, γ, entropy coefficient) during training, not before
- Maintains population diversity
- Can discover phase transitions in training (e.g., optimal time to increase γ)

**Cons:**
- Requires running N agents simultaneously (linear compute scaling)
- Early convergence if population diversity collapses

**Integration:** Ray Tune implements PBT natively and integrates with RLlib.

---

### 2.3 Self-Play Variants: FSP, PFSP, PSRO

#### Fictitious Self-Play (FSP)

**What it is:** At each step, train a best-response to the historical average strategy. The historical average strategy is the uniform mixture of all past policies.

**Theorem:** In two-player zero-sum games, FSP converges to a Nash equilibrium.

**Problem:** Slow convergence; the uniform average gives equal weight to early (weak) and recent (strong) policies.

#### Prioritized Fictitious Self-Play (PFSP)

**What it is:** Instead of sampling opponents uniformly, weight opponent selection by a function of the current win rate.

**Priority functions:**
```python
# Focus on hard opponents (AlphaStar style):
f(x) = (1 - x)**p  # x = win rate vs opponent; p > 0
# Prioritizes opponents you currently LOSE to

# Focus on matched opponents:
f(x) = x * (1 - x)
# Prioritizes opponents near 50% win rate
```

**In AlphaStar:** `f(x) = (1 - x)^p` with p=1, so harder opponents are sampled more. This focuses training compute on reducing the worst-case performance.

**Advantages over FSP:**
- Faster convergence
- Less exploitable final policy
- Better final performance

#### PSRO (Policy Space Response Oracles)

**What it is:** A game-theoretic framework that iteratively:
1. Computes a **Nash equilibrium meta-strategy** over the current policy population
2. Computes a **best response** to that meta-strategy
3. Adds the best response to the population
4. Repeats

**Algorithm:**
```
Initialize population Π = {π_0}

While not converged:
  σ* = NashEquilibrium(payoff_matrix(Π))  # Meta-solver
  π_new = BestResponse(σ*)               # Oracle (RL training)
  Π = Π ∪ {π_new}
  Update payoff_matrix with π_new vs all existing policies
```

**Variants:**
- **Alpha-PSRO:** Uses α-Rank instead of Nash equilibrium as meta-solver
- **Pipeline PSRO:** Parallelizes BR computation across iterations
- **Distributed PSRO (2025):** TOP-K truncation of policy pool for efficiency
- **Conflux-PSRO (2024):** Adaptive policy selection at state level
- **SP-PSRO:** Incorporates an approximately optimal stochastic policy in each iteration

**When to Use PSRO vs PFSP:**
- PSRO has stronger game-theoretic guarantees (approaches Nash equilibrium)
- PFSP/League training scales better to complex games like StarCraft
- PSRO is more principled for research; PFSP is more practical for large-scale training

**Survey Paper (2024):** [arxiv.org/abs/2403.02227](https://arxiv.org/abs/2403.02227)

---

### 2.4 ELO and TrueSkill Systems

**ELO Rating:**

Classic chess rating system. After a match between players A and B:
```
Expected score: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
Rating update:  R_A_new = R_A + K * (S_A - E_A)
```
Where K ≈ 32, S_A ∈ {0, 0.5, 1} (loss/draw/win).

**Use in RL Training:**
- Track population of agents during league training
- Plot ELO curves to diagnose training progress
- Use ELO difference as a criterion for league entry (e.g., add to league when >100 ELO above all current members)

**TrueSkill (Microsoft Research):**

Bayesian extension of ELO; models skill as a Gaussian distribution `N(μ, σ²)`.
- `μ` = mean skill estimate
- `σ` = uncertainty

Updates use Bayesian inference after each match. Better for:
- N-player (>2 player) games
- Sparse match histories
- Tracking uncertainty in early training

**OpenAI Five** used TrueSkill ratings against fixed reference agents for training evaluation. A difference of 8.3 TrueSkill corresponds to ~80% win rate.

**Practical Recommendation:** Use ELO for 2-player zero-sum settings. Use TrueSkill for N-player or mixed games with sparse match history.

---

### 2.5 Opponent Sampling Strategies

**Summary of strategies:**

| Strategy | Selection Method | Properties |
|----------|-----------------|------------|
| **Uniform Self-Play** | Random past policy | Simple; suffers from forgetting |
| **PFSP (hard focus)** | `f(x) = (1-x)^p` | Focuses on currently-lost matchups |
| **PFSP (balanced)** | `f(x) = x(1-x)` | Focuses on 50/50 matchups |
| **Win-rate based** | Sample proportional to win rate | Focuses on winnable matchups for policy distillation |
| **Diversity-based** | Behavioral diversity metric | Prevents population collapse |
| **AOS (Automatic Opponent Sampling, 2024)** | Learned sampling policy | Adaptive; outperforms manual PFSP in air combat study |

---

## 3. Behavioral Cloning and Imitation Learning

### 3.1 Behavioral Cloning (BC)

**What it is:** Direct supervised learning on `(observation, action)` pairs from expert demonstrations.

```python
# Training objective
loss = CrossEntropy(policy(observation), expert_action)
# or MSE for continuous actions
```

**Key Problem — Covariate Shift:**
The agent is trained on the expert's state distribution, but at test time it will visit states the expert never encountered (due to compounding errors). This causes exponential degradation.

**Covariate shift formula:** If per-step error probability is `ε`, after T steps, total error is `O(ε T²)`.

**How AlphaStar used BC:**
- Trained on human replay database (millions of games)
- KL divergence loss: `loss = KL(policy || expert)`
- Achieved 84% win rate vs human players before any RL
- Used as initialization for RL and as a regularization target throughout RL training

**When to Use:**
- Initial policy bootstrapping before RL
- When expert data is plentiful and state distribution shift is manageable
- As a regularizer during RL to prevent forgetting

**Library:** `imitation` package, `stable-baselines3` pretrain API

---

### 3.2 DAgger (Dataset Aggregation)

**What it is:** Interactive imitation learning that solves covariate shift by having the expert label the states the *learner* visits (not just states the expert visits).

**Algorithm:**
```
Initialize: D = expert_demonstrations
Train: π_1 = BC(D)

For each round i:
  1. Roll out π_i in environment
  2. Query expert for labels on visited states → D_new
  3. D = D ∪ D_new
  4. Train: π_{i+1} = BC(D)
```

**Theorem:** DAgger achieves O(ε T) error (linear in T), vs O(ε T²) for BC.

**Variants in `imitation` library:**
- `DAggerTrainer`: Low-level API, supports interactive expert feedback
- `SimpleDAggerTrainer`: Automated, uses synthetic expert for online labeling

**When to Use:**
- When you can query an expert (rule-based bot, human player) interactively
- When covariate shift is the primary failure mode of BC

---

### 3.3 GAIL (Generative Adversarial Imitation Learning)

**What it is:** Learns a reward function (via GAN discriminator) that makes the agent's behavior distribution match the expert's distribution, then trains with RL on that learned reward.

**Architecture:**
```
Discriminator D: (state, action) → probability it's from expert
Generator (Policy π): trained to fool D via RL

Training:
  D_loss = -[E_expert[log D(s,a)] + E_agent[log(1-D(s,a))]]
  π_loss = -E_agent[log D(s,a)]  (RL objective)
```

**Advantages over BC:**
- Robust to covariate shift (uses RL for exploration)
- No need for explicit expert action labels (can learn from observation-only demonstrations)
- Converges to expert distribution rather than just individual actions

**Disadvantages:**
- GAN training instability
- More complex to implement and tune
- Discriminator reward signal does not represent ground truth reward

**AIRL (Adversarial IRL):** Extension of GAIL that learns a transferable reward function, not just a policy.

**When to Use:**
- When expert actions are unavailable (observation-only demonstrations)
- When you need a recoverable reward function (AIRL)
- When you have sufficient expert demonstrations for GAN training stability

---

### 3.4 Inverse Reinforcement Learning (IRL)

**What it is:** Learn the reward function `R(s, a)` that best explains observed expert behavior, then use the recovered reward to train an agent via RL.

**Key IRL methods:**

| Method | Key Idea | Library |
|--------|----------|---------|
| **Maximum Entropy IRL** | `R* = argmax_R [entropy(π*) - E_π*[R]]` | `imitation` (MCE IRL) |
| **AIRL** | GAN-based; learns portable reward | `imitation` |
| **GAIL** | Matches state-action distributions | `imitation` |

**When to Use:**
- When you want a reward function for transfer to new environments
- When expert behavior needs to be generalized beyond specific states seen

---

### 3.5 Kickstarting RL from Demonstrations

**The Problem:** After training with BC, switching to RL can cause **catastrophic forgetting**: the agent rapidly departs from the expert policy during RL exploration, losing the pre-trained knowledge.

**Kickstarting Deep RL (Schmitt et al., 2018 — [arxiv:1803.03835](https://arxiv.org/abs/1803.03835)):**

Uses **knowledge distillation from a teacher to a student** during RL training:
```
loss = RL_loss(student) + λ * KL(student || teacher)
```
The λ coefficient decays over training, allowing the student to eventually surpass the teacher. On DMLab-30: kickstarted agent matches scratch-trained performance in ~10× fewer steps and surpasses it by 42%.

**AlphaStar's approach:**
- KL loss toward supervised policy is maintained throughout RL training (never fully dropped)
- Z-conditioning provides a soft inductive bias toward human strategies without forcing exact imitation

**Imitation Bootstrapped RL (IBRL, 2024 — [arxiv:2311.02198](https://arxiv.org/abs/2311.02198)):**
- Train IL policy first
- Use IL policy to propose alternative actions for RL exploration
- Bootstrap Q-value targets using IL policy

**Concurrent IL + RL Training (2024 — [arxiv:2304.09825](https://arxiv.org/abs/2304.09825)):**
- Train both IL and RL objectives simultaneously
- Key finding: **diversity in demonstrations is more important than quality** for generalization in procedurally generated environments

**Key Strategies for Catastrophic Forgetting:**

1. **KL regularization** toward BC policy (AlphaStar approach)
2. **Experience replay** with demonstrations mixed into the RL replay buffer
3. **Teacher-student distillation** with decaying λ (Kickstarting)
4. **Behavioral cloning auxiliary loss** maintained throughout training

**Library:** [imitation.readthedocs.io](https://imitation.readthedocs.io), stable-baselines3

---

## 4. N-Player Game Training

### 4.1 Challenges of N-Player Games

**Non-stationarity magnification:** With N agents, each agent's environment is non-stationary because N-1 others are simultaneously learning. This violates Markov assumptions and destabilizes value function estimates.

**Credit assignment:** In cooperative or mixed settings, attributing outcomes to specific agent actions becomes combinatorially harder.

**Strategy space explosion:** The number of mixed strategy Nash equilibria grows rapidly; computing or approximating equilibria is PPAD-hard for N>2.

**Coalition dynamics:** With N>2 players, temporary coalitions form and dissolve. Optimal play involves theory-of-mind reasoning about which coalitions to join/break.

### 4.2 Nash Equilibrium in N-Player Settings

**Two-player zero-sum:** Unique Nash equilibrium; can be computed efficiently via linear programming or CFR. RL with self-play provably converges.

**N-player zero-sum (N>2):** Multiple Nash equilibria may exist; no general efficient algorithm. Pluribus showed that empirical performance (trained via self-play MCCFR) can beat humans without computing Nash equilibrium.

**General-sum N-player:** No tractable exact algorithm. Approaches:
- **Mean-Field Games:** Approximate N→∞ limit for large populations
- **PSRO:** Iteratively approximate Nash via best-response computation
- **Empirical Game Theory:** Build a payoff matrix from simulated matches and solve the meta-game

**Practical recommendation:** For N-player competitive programming competitions, focus on empirical performance rather than Nash equilibrium. Use PSRO or league training with PFSP to build robust policies.

### 4.3 Neural MMO (100+ Player)

**Overview:** A massively multiagent environment simulating an MMORPG-style world with 8–1024 concurrent agents competing for resources.

**Version 2.0 features:**
- **Multi-task:** 1,297 training tasks, 63 held-out evaluation tasks
- **Procedural generation:** Every map is different; agents must generalize
- **Heterogeneous evaluation:** Evaluated on opponents and tasks never seen during training
- **Competition limits:** 8 A100 GPU-hours equivalent + 12 CPU cores

**Training architecture:**
- PPO (CleanRL single-file implementation)
- PufferLib integration for simplified training
- WandB logging
- Species-based parameter sharing (agents of the same species share weights)

**Key insight for large N:** Species specialization emerges naturally through competitive pressure. Without explicit coordination, species develop behavioral niches.

**Code:** [neuralmmo.github.io](https://neuralmmo.github.io), [github.com/openai/neural-mmo](https://github.com/openai/neural-mmo)

**NeurIPS 2023 Competition Results:** [arxiv.org/html/2508.12524v1](https://arxiv.org/html/2508.12524v1)

### 4.4 Hanabi (Cooperative Imperfect Information)

**Overview:** A 2–5 player cooperative card game where players cannot see their own hand, forcing communication via legal hint actions.

**Why it's important for research:**
- Tests **theory of mind** (modeling other players' beliefs from their actions)
- Tests **zero-shot coordination** (playing with an unknown teammate)
- Benchmark for cooperative MARL without explicit communication channels

**Key algorithms developed for Hanabi:**
- **Other-Play (OP):** Trains agents against randomly permuted copies of themselves; improves zero-shot coordination
- **Simplified Action Decoder (SAD):** Extends policies to act more informatively
- **SPARTA:** Search-based agent reaching near-perfect play

**Environment:** [github.com/deepmind/hanabi-learning-environment](https://github.com/deepmind/hanabi-learning-environment)

---

## 5. Practical Frameworks

### 5.1 OpenSpiel (Google DeepMind)

**What it is:** A collection of environments and algorithms for research in RL and planning/search in games.

**Game Support:**
- Perfect and imperfect information games
- Zero-sum, cooperative, and general-sum games
- N-player games (1 to N)
- Turn-based and simultaneous-move
- Normal-form (one-shot) and extensive-form (sequential)

**Algorithms Included:**
- Deep learning: Deep CFR, Neural Fictitious Self-Play (NFSP)
- Tabular: CFR, CFR+, MCCFR, Extensive-form fictitious play
- Tree search: Alpha-beta pruning, MCTS, PUCT
- Game-theoretic solvers: Linear programming Nash solver, Nash equilibrium computation
- RL: DQN, PPO, REINFORCE, actor-critic
- Imitation: Behavioral cloning

**Architecture:**
- Core in **C++** for performance
- **Python** bindings via pybind11
- Works with JAX, PyTorch, TensorFlow

**N-player support:** Full. Supports up to N-player games natively with utility functions for all N players.

**Notable games included:**
- Chess, Go, Tic-Tac-Toe, Connect Four, Breakthrough
- Texas Hold'em poker (limit and no-limit)
- Hanabi, Kuhn poker, Leduc poker
- Goofspiel, Coin game, and 80+ others

**Upcoming:** OpenSpiel 2.0 (integration with Kaggle Game Arena)

**When to Use:** Research on game-theoretic algorithms, Nash equilibrium computation, comparing RL vs game theory approaches.

**Link:** [github.com/google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel)

---

### 5.2 PettingZoo (Farama Foundation)

**What it is:** A standardized multi-agent environment API, the "multi-agent equivalent of Gymnasium."

**API design:**
- **AEC (Agent Environment Cycle):** Turn-based environments where one agent acts at a time
- **Parallel API:** All agents act simultaneously (e.g., Dota-style)

**Environment families:**
- **Classic:** Chess, Go, Checkers, Poker, Mahjong (using OpenSpiel backends)
- **Atari:** 28 multiplayer Atari games
- **Butterfly:** Cooperative games requiring coordination (Pistonball, Cooperative Pong)
- **MPE (Multi-Agent Particle Environments):** Continuous-space tasks for communication research
- **SISL:** Cooperative safety tasks

**Integration:** EPyMARL 2.0, PyMARLzoo+, BenchMARL, and most major MARL libraries support PettingZoo.

**When to Use:** As your environment standard when building new competitive or cooperative environments. Plug-and-play with most MARL training frameworks.

**Link:** [github.com/Farama-Foundation/PettingZoo](https://github.com/Farama-Foundation/PettingZoo), [pettingzoo.farama.org](https://pettingzoo.farama.org)

---

### 5.3 JaxMARL (FLAIROx)

**What it is:** Pure JAX implementation of MARL environments and algorithms. Presented at NeurIPS 2024.

**Performance:**
- Up to **14x faster** than current popular approaches
- Up to **12,500x faster** when multiple training runs are vectorized (JIT-compiled parallel rollouts)

**Algorithms:**
- IPPO (Independent PPO)
- MAPPO (Multi-Agent PPO with centralized critic)
- QMIX (Value decomposition)
- VDN (Value Decomposition Networks)

**Environments:**
- MPE, Hanabi, SMAX (StarCraft II Multi-Agent Challenge in JAX), Overcooked, Cooperative Navigation

**API:** Actions, observations, rewards, and done values are dictionaries keyed by agent name, supporting heterogeneous agent spaces.

**When to Use:** When training speed is critical and your environment can be implemented in JAX (or you use provided environments). Ideal for large-scale hyperparameter sweeps.

**Link:** [github.com/FLAIROx/JaxMARL](https://github.com/FLAIROx/JaxMARL)

---

### 5.4 Mava (InstaDeep)

**What it is:** A lightweight, performant JAX-based MARL library built on top of CleanRL and PureJaxRL patterns.

**Architectures:**
- **Anakin:** End-to-end JAX, for JAX-native environments. Full JIT compilation including environment steps.
- **Sebulba:** For non-JAX environments. Environment runs on CPU cores in parallel, learner on GPU.

**Algorithms:**
- IPPO, MAPPO
- MADDPG, MASAC
- QMIX, VDN

**Design philosophy:** Modular components (exploration strategies, network architectures) easily swappable. Statistically robust evaluation with JSON logging.

**Wrappers:** PettingZoo, Brax, and other suites.

**When to Use:** When you need production-quality MARL training with JAX performance and want flexibility to bring your own environments.

**Link:** [github.com/instadeepai/Mava](https://github.com/instadeepai/Mava)

---

### 5.5 EPyMARL / PyMARL

**What it is:** Extended PyMARL (EPyMARL) is the standard PyTorch codebase for cooperative MARL, extended from the original PyMARL (which only supported SMAC).

**EPyMARL 2.0 (2024) additions:**
- Gymnasium compatibility
- Support for **different reward functions per agent** (partial competitiveness)
- **WandB logging integration**
- Native PettingZoo support
- SMACv2, SMAClite, VMAS environments

**Algorithms:**
- IQL, COMA, VDN, QMIX, QTRAN (from PyMARL)
- IA2C, IPPO, MAA2C, MAPPO, MADDPG (added in EPyMARL)

**PyMARLzoo+ (2025):** Extended EPyMARL with QPLEX, HAPPO, MAT-DEC, CDS, EOI, EMC, MASER, tested on PettingZoo and beyond.

**When to Use:** When you need a well-benchmarked PyTorch-based cooperative MARL baseline with a large set of algorithms.

**Links:** [github.com/uoe-agents/epymarl](https://github.com/uoe-agents/epymarl)

---

### 5.6 MARLlib (RLlib Extension)

**What it is:** An extension of Ray RLlib that unifies multi-agent environments and algorithms under a single interface.

**Coverage:** Claims the widest coverage of algorithms and environments in the MARL ecosystem.

**Features:**
- Flexible parameter sharing strategies (full sharing, no sharing, partial sharing)
- Supports both cooperative and competitive settings
- Built on Ray for distributed training

**When to Use:** When you already use RLlib and want to extend to MARL with minimal code changes.

---

### 5.7 Ray RLlib

**What it is:** Industry-grade, scalable, fault-tolerant distributed RL library built on Ray.

**Key features:**
- Multi-agent support: independent learning, shared policy, adversarial
- **Self-play support:** League-based self-play, configurable matchmaking
- Fault-tolerant distributed training
- Policy server/client architecture for decoupled inference
- PyTorch default models + multi-GPU training

**Self-play in RLlib:**
```python
config = PPOConfig().multi_agent(
    policies={
        "main": PolicySpec(...),
        "opponent": PolicySpec(...),
    },
    policy_mapping_fn=lambda agent_id, *args, **kw: (
        "main" if agent_id == "player_0" else "opponent"
    ),
    # League-based self-play callbacks available
)
```

**Scale:** Used at 256 GPUs + 128,000 CPUs (OpenAI Five scale is achievable).

**When to Use:** Production-level distributed RL, especially when you need to scale beyond a single machine. Strong choice for competitive game training frameworks.

**Link:** [docs.ray.io/en/latest/rllib/index.html](https://docs.ray.io/en/latest/rllib/index.html)

---

### 5.8 Weights & Biases for RL Experiments

**What it is:** Experiment tracking, visualization, and collaboration platform. Standard in RL research.

**Key features for RL:**
- Real-time reward curve visualization
- Hyperparameter tracking
- Model artifact versioning
- Distributed training support (rank-zero-only logging to avoid duplication)

**Integration patterns:**

```python
import wandb

# Initialize
wandb.init(project="my_rl_training", config=hyperparams)

# Log per step
wandb.log({
    "reward/mean": mean_reward,
    "reward/max": max_reward,
    "loss/policy": policy_loss,
    "loss/value": value_loss,
    "elo": current_elo,
    "win_rate_vs_opponent": win_rate,
})

# For distributed (Ray + wandb):
from ray.air.integrations.wandb import setup_wandb
setup_wandb(config, rank_zero_only=True)
```

**For league training, log:**
- ELO curves per agent type (MA, ME, LE)
- Win rates matrix across league population
- Strategy distribution (Z-statistics for AlphaStar-like agents)
- Population size over time

**EPyMARL 2.0 natively supports WandB logging.**

**Link:** [wandb.ai](https://wandb.ai), [docs.wandb.ai](https://docs.wandb.ai)

---

### 5.9 Optuna / Ray Tune for Hyperparameter Optimization

**Optuna:**
- Bayesian optimization (TPE sampler by default)
- Pruning support (Hyperband) for early stopping of bad trials
- Lightweight; single-machine or distributed

```python
import optuna

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.9999)
    # train agent, return final evaluation score
    return train_and_eval(lr=lr, gamma=gamma)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
```

**Ray Tune:**
- Distributed HPO natively
- Integrates with Optuna (`OptunaSearch`), Ax, BOHB, NeverGrad
- Supports Population Based Training (PBT) as a scheduler
- Integrates with RLlib and WandB

```python
from ray import tune
from ray.tune.search.optuna import OptunaSearch

analysis = tune.run(
    train_fn,
    search_alg=OptunaSearch(metric="mean_reward", mode="max"),
    scheduler=tune.schedulers.ASHAScheduler(metric="mean_reward", mode="max"),
    num_samples=50,
    config={
        "lr": tune.loguniform(1e-5, 1e-2),
        "gamma": tune.uniform(0.95, 0.9999),
    }
)
```

**Recommendation for competitive programming competitions:**
1. Use Ray Tune + OptunaSearch for initial exploration
2. Use PBT scheduler for long training runs (dynamically adapts hyperparameters)
3. Track all experiments with WandB

---

### 5.10 Imitation Learning Library

**What it is:** A Python library implementing imitation learning algorithms on top of Stable-Baselines3 and PyTorch.

**Algorithms:**
- **BC** (Behavioral Cloning)
- **DAgger** (SimpleDAggerTrainer + DAggerTrainer)
- **GAIL** (Generative Adversarial IL)
- **AIRL** (Adversarial IRL)
- **MCE IRL** (Maximum Causal Entropy IRL)
- **Preference Comparisons** (reward learning from human feedback)

**Install:** `pip install imitation`

**Quick BC example:**
```python
from imitation.algorithms import bc
from imitation.data import rollout

rng = np.random.default_rng(0)
expert = load_expert_policy(...)
transitions = rollout.rollout(expert, env, rng=rng, n_timesteps=10000)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)
bc_trainer.train(n_epochs=100)
```

**Link:** [imitation.readthedocs.io](https://imitation.readthedocs.io), [github.com/HumanCompatibleAI/imitation](https://github.com/HumanCompatibleAI/imitation)

---

## 6. Key Algorithms Reference

### 6.1 MAPPO

**Multi-Agent PPO with centralized critic.**

- Decentralized actors: each agent has its own policy network conditioned on local observation
- Centralized critic: value function conditioned on global state (CTDE paradigm)
- Loss: standard PPO clip loss per agent

**When to use:** General cooperative MARL baseline. Consistently strong across diverse tasks.

**Key result (2025 benchmark):** MAPPO and MAA2C are the most consistent algorithms across cooperative tasks; MAPPO significantly outperforms IPPO when global state is available.

### 6.2 QMIX

**Value decomposition for cooperative MARL.**

**Key equation:**
```
Q_total(s, a_1, ..., a_N) = Monotonic_Mixing_Net(Q_1(o_1, a_1), ..., Q_N(o_N, a_N))
```

The mixing network enforces **monotonicity**: `∂Q_total/∂Q_i ≥ 0`. This ensures that individually-greedy actions (argmax over individual Q-functions) also maximize the joint Q-function.

**When to use:** Cooperative tasks where global state is available for training. Stronger than VDN (which uses simple summation). QPLEX generally outperforms QMIX on harder tasks.

### 6.3 MADDPG

**Multi-Agent Deep Deterministic Policy Gradient.**

- Each agent has a centralized critic conditioned on all agents' observations and actions
- Decentralized actor using only local observation
- DDPG-style policy gradient through the critic

**When to use:** Continuous action spaces with multiple agents. MA variants (MASAC, MADDPG, MAPPO) outperform single-agent and Q-learning methods in continuous control.

### 6.4 HAPPO

**Heterogeneous-Agent PPO — sequential update with monotonic improvement guarantees.**

Agents are updated **sequentially** (not simultaneously), using the multi-agent advantage decomposition lemma to guarantee monotonic improvement of the joint policy.

```
# Advantage decomposition:
A^joint(s, a_1, ..., a_N) = Σ_i A^i(s, a_1,...,a_i | a_{<i})
```

**When to use:** Heterogeneous cooperative settings where agents have different observation/action spaces. Stronger theoretical guarantees than MAPPO.

---

## 7. Recent Advances (2023–2025)

### New Systems and Papers

**ROA-Star (NeurIPS 2023):**
Robust opponent-aware improvement over AlphaStar's league training. Uses opponent strategy prediction to make the main agent responsive to counter-strategies. Achieves >50% win rate vs top humans with less compute.
Paper: [proceedings.neurips.cc/paper_files/paper/2023/...](https://proceedings.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf)

**PSRO Survey (2024):**
Comprehensive survey of PSRO variants, meta-solvers, and extensions.
Paper: [arxiv.org/abs/2403.02227](https://arxiv.org/abs/2403.02227)

**JaxMARL at NeurIPS 2024:**
12,500x training speedup via fully vectorized JAX training. Benchmarks all major MARL algorithms.

**Hokoff (NeurIPS 2023):**
Real-game offline MARL dataset from Honor of Kings. 100M+ daily active players ensures dataset quality and practical relevance.

**Conflux-PSRO (2024):**
State-level adaptive policy selection in PSRO. Fully exploits population diversity.

**Code-Space Response Oracles (2025):**
LLM-generated best responses in PSRO framework. Replaces black-box RL oracle with interpretable code generation.

**Distributed PSRO (2025, IEEE):**
TOP-K truncation for efficient large-scale PSRO.

**MARL Textbook (MIT Press, 2024):**
"Multi-Agent Reinforcement Learning: Foundations and Modern Approaches" by Albrecht, Christianos, Schäfer.
Available at [marl-book.com](https://marl-book.com).

**Lux AI Season 3 (NeurIPS 2024):**
Competition focused on meta-learning in a 1v1 format with 5-game series. Tests adaptation to changing game dynamics. GPU-parallelized environment for large-scale RL research.

### JAX Ecosystem Dominance

The clear trend in 2024-2025 is the **shift to JAX-based training** for competitive and cooperative MARL:
- JaxMARL: 12,500x speedup
- Mava: Anakin/Sebulba architectures
- SMAX (StarCraft II in JAX)

For time-constrained competitions, JAX-based training is the highest-leverage architectural choice.

### LLM + RL Integration

Emerging theme: using LLMs as opponent models, strategy generators, or code generators within RL pipelines (Cicero, TiG framework, Code-Space Response Oracles). Practical for competitions where natural language description of strategies is available.

---

## 8. Practical Recommendations

### For Time-Constrained Competitive Programming Competitions

**Phase 1: Environment + BC Baseline (Days 1–2)**
1. Implement your environment as a PettingZoo-compatible AEC or Parallel env
2. Collect rule-based agent trajectories if available, train BC baseline
3. Set up WandB experiment tracking immediately

**Phase 2: RL Training (Days 3–5)**
1. Start with **PPO (single agent or IPPO/MAPPO)** — most robust default
2. Use KL loss toward BC policy as regularization to prevent forgetting
3. Use γ annealing: start with γ=0.99, anneal to γ=0.999 over training
4. Dense reward shaping: identify intermediate signals (resource collection, survival, etc.)

**Phase 3: Self-Play / League (Days 5–10)**
1. Start with **simple self-play** (train against latest checkpoint copy)
2. Maintain a pool of past checkpoints (5–10 recent + 2–3 oldest)
3. Implement PFSP with `f(x) = (1-x)^p` for opponent selection
4. Track ELO across population; add new agent to pool every N training steps

**Phase 4: Optimization (Days 10+)**
1. Ray Tune + OptunaSearch for hyperparameter search
2. Consider PBT for long training runs
3. If JAX environment is possible, migrate to JaxMARL/Mava for 10–100x speedup

### Framework Recommendations by Use Case

| Use Case | Primary Framework | Secondary |
|----------|------------------|-----------|
| Fast iteration, small scale | Stable-Baselines3 + imitation | PettingZoo |
| Research benchmarking | EPyMARL 2.0 or JaxMARL | OpenSpiel |
| Production scale (100+ CPUs) | Ray RLlib + MARLlib | Mava (Sebulba) |
| Game-theoretic analysis | OpenSpiel | PSRO implementations |
| JAX-native (fastest) | JaxMARL or Mava (Anakin) | — |
| Imitation learning | imitation (SB3 ecosystem) | stable-baselines3 |
| Hyperparameter optimization | Ray Tune + Optuna | Optuna standalone |

### Key Hyperparameters for Competitive RL

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Learning rate | 1e-4 to 3e-4 | Anneal if training long |
| Gamma (γ) | 0.99 to 0.9997 | Anneal upward over training |
| PPO clip epsilon | 0.1 to 0.2 | Lower for stability |
| Entropy coefficient | 0.001 to 0.01 | Decay over time |
| KL coefficient (BC loss) | 0.1 to 1.0 | Decay as agent improves |
| Rollout workers | 4–32 | More = faster data collection |
| Minibatch size | 256–4096 | Larger = more stable gradients |

### Architecture Recommendations

**For complex game observations:**
- Use **entity-based architecture** (Transformer over game entities) rather than flat vectors
- AlphaStar's architecture is the gold standard for entity-heavy games

**For spatial observations (maps, grids):**
- Use **CNN + LSTM** or **CNN + Transformer** for spatial encoding + temporal memory

**For simple observations (N numbers):**
- MLP + LSTM is sufficient; 256–1024 hidden units
- OpenAI Five's 4096-unit LSTM is overkill for most competition settings

**Recommended minimal architecture for competitions:**
```python
class CompetitiveAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.policy_head = nn.Linear(hidden, act_dim)
        self.value_head = nn.Linear(hidden, 1)
```

---

## 9. Framework Selection Matrix

| Dimension | Recommendation | Rationale |
|-----------|---------------|-----------|
| **Simplest to start** | Stable-Baselines3 + PettingZoo | Mature, well-documented, GPU support |
| **Fastest training** | JaxMARL (Anakin architecture) | 12,500x vectorized speedup |
| **Most algorithms** | EPyMARL 2.0 or MARLlib | Broadest algorithm coverage |
| **Game theory** | OpenSpiel | CFR, Nash solvers, 80+ games |
| **Imitation learning** | imitation library | BC, DAgger, GAIL, AIRL, all in one |
| **Distributed scale** | Ray RLlib + Mava (Sebulba) | Production-grade distributed execution |
| **Experiment tracking** | Weights & Biases | Standard for RL research in 2024–2025 |
| **HPO** | Optuna (small) / Ray Tune (large) | Both support PBT and Bayesian methods |
| **Competition environments** | PettingZoo + custom | Standard API, integrates everywhere |

---

## 10. References and Links

### Foundational Papers

- **AlphaStar:** [nature.com/articles/s41586-019-1724-z](https://www.nature.com/articles/s41586-019-1724-z) | [Google DeepMind Blog](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
- **ROA-Star (NeurIPS 2023):** [proceedings.neurips.cc/paper_files/paper/2023/...](https://proceedings.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf)
- **OpenAI Five:** [arxiv.org/abs/1912.06680](https://arxiv.org/abs/1912.06680) | [openai.com/index/openai-five](https://openai.com/index/openai-five/)
- **AlphaZero:** [science.sciencemag.org/content/362/6419/1140](https://www.science.org/doi/10.1126/science.aar6404)
- **MuZero:** [nature.com/articles/s41586-020-03051-4](https://www.nature.com/articles/s41586-020-03051-4) | [MuZero Wikipedia](https://en.wikipedia.org/wiki/MuZero)
- **Cicero:** [science.org/doi/10.1126/science.ade9097](https://www.science.org/doi/10.1126/science.ade9097) | [Meta AI Blog](https://ai.meta.com/blog/cicero-ai-negotiates-persuades-and-cooperates-with-people/)
- **Pluribus:** [science.org/doi/10.1126/science.aay2400](https://www.science.org/doi/10.1126/science.aay2400)
- **PSRO Survey (2024):** [arxiv.org/abs/2403.02227](https://arxiv.org/abs/2403.02227)
- **Kickstarting Deep RL:** [arxiv.org/abs/1803.03835](https://arxiv.org/abs/1803.03835)
- **IBRL (2024):** [arxiv.org/abs/2311.02198](https://arxiv.org/abs/2311.02198)

### Framework GitHub Links

| Framework | Link |
|-----------|------|
| AlphaStar (open) | [github.com/google-deepmind/alphastar](https://github.com/google-deepmind/alphastar) |
| OpenSpiel | [github.com/google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel) |
| PettingZoo | [github.com/Farama-Foundation/PettingZoo](https://github.com/Farama-Foundation/PettingZoo) |
| JaxMARL | [github.com/FLAIROx/JaxMARL](https://github.com/FLAIROx/JaxMARL) |
| Mava | [github.com/instadeepai/Mava](https://github.com/instadeepai/Mava) |
| EPyMARL | [github.com/uoe-agents/epymarl](https://github.com/uoe-agents/epymarl) |
| Cicero | [github.com/facebookresearch/diplomacy_cicero](https://github.com/facebookresearch/diplomacy_cicero) |
| imitation | [github.com/HumanCompatibleAI/imitation](https://github.com/HumanCompatibleAI/imitation) |
| Neural MMO | [neuralmmo.github.io](https://neuralmmo.github.io) |
| HoK Env | [github.com/tencent-ailab/hok_env](https://github.com/tencent-ailab/hok_env) |
| MiniZero (AZ/MuZero) | [github.com/rlglab/minizero](https://github.com/rlglab/minizero) |
| alpha-zero-general | [github.com/suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general) |
| Lux AI Season 3 | [github.com/Lux-AI-Challenge/Lux-Design-S3](https://github.com/Lux-AI-Challenge/Lux-Design-S3) |
| Ray RLlib | [docs.ray.io/en/latest/rllib/index.html](https://docs.ray.io/en/latest/rllib/index.html) |
| Weights & Biases | [wandb.ai](https://wandb.ai) |
| Optuna | [optuna.org](https://optuna.org) |

### Textbooks and Surveys

- **MARL Textbook (MIT Press, 2024):** [marl-book.com](https://marl-book.com) — "Multi-Agent Reinforcement Learning: Foundations and Modern Approaches"
- **PSRO Survey (IJCAI 2024):** [ijcai.org/proceedings/2024/0880.pdf](https://www.ijcai.org/proceedings/2024/0880.pdf)
- **Self-Play Methods Survey (2024):** [arxiv.org/pdf/2408.01072](https://arxiv.org/pdf/2408.01072)
- **Population-Based Deep RL Survey:** [mdpi.com/2227-7390/11/10/2234](https://www.mdpi.com/2227-7390/11/10/2234)
- **BenchMARL:** [arxiv.org/abs/2312.01472](https://arxiv.org/abs/2312.01472)
- **JaxMARL (NeurIPS 2024):** [proceedings.neurips.cc/paper_files/paper/2024/file/5aee125f052c90e326dcf6f380df94f6-Paper-Datasets_and_Benchmarks_Track.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/5aee125f052c90e326dcf6f380df94f6-Paper-Datasets_and_Benchmarks_Track.pdf)

---

*Research compiled March 2026. Covers literature and systems through early 2025.*

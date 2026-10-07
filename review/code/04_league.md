# R4 — League / matchmaking / ratings / checkpoints / eval / scenario config: adversarial review

Scope: `coordinator/{coordinator,matchmaker,agent_pool,checkpoint_manager,ratings}.py`, `eval.py`, `core/outcomes.py`, `core/config.py` (training / self_play / checkpoint / agents), the match / checkpoint / result parts of `launcher.py` and `distributed.py`, how `worker/rollout_worker.py` consumes slot maps, `bc/kickstart.py` (warm start), and the related tests.

How I checked: I read all of the code above. I also ran three scripts against the real classes: `scratchpad/r4_league/v1_matchmaking.py`, `v2_ckpt_elo_cfg.py` and `v3_eval.py`. Each one drives the code the same way the launcher does. Finally I ran the tests in scope: `test_ratings`, `test_milestone2`, `test_multi_agent`, `test_eval` and `test_review_fixes`. **All 46 pass** (209 s, run from the scratch cwd). None of them catch the problems below.

## Summary

The league layer works only for one case: a single trainable agent doing 2-player self-play against its own FIFO history. All of the owner's priorities are unsupported or broken:

- arenas with mixed architectures and scripted bots
- sound selection of the best checkpoint
- monitoring ratings
- N-player FFA
- per-agent warm start

The three most serious problems:

1. **The launcher generates matches only for `agents[0]`.** Other agents starve in `self_play` and get second-class treatment in `league`.
2. **Frozen and scripted agents cannot be used anywhere.** The pool API exists but nothing calls it, there is no config for it, and the worker cannot execute such a slot.
3. **Ratings and win rates are wrong for N-player games or when an agent fills more than one seat.** They are also never logged or persisted.

---

## Findings

### R4-01 — CRITICAL — bug — matches are generated only for the first trainable agent
**Location:** `launcher.py:420-422` (launch), `launcher.py:607-610` (`_refresh_worker_matches`: `primary = agent_ids[0]`), `coordinator.py:94-103`

**What is wrong:** Every worker's slot maps come from `coordinator.generate_match_configs(trainable_agents[0], …)`. The matchmakers are all written "for a given agent". Nothing iterates over the other trainable agents.
- In `phase: self_play` with more than one agent, agents 2..N **never appear in any slot**. Their learners receive no data. The run ends with the message "All worker processes exited unexpectedly", leaving those agents untrained.
- In `phase: league`, agents 2..N only ever appear as the PFSP opponent of `agents[0]`:
  - they never play each other;
  - they never play their own (or anyone's) history;
  - they get about 4× less data than `agents[0]`.

**Evidence (VERIFIED-BY-RUNNING, v1 §A/§B).** League with 3 agents, 2 checkpoints each, 16,000 matches generated the way the launcher does it:
```
agent sets per match: {('a0','a2'): 4059, ('a0',): 7992, ('a0','a1'): 3949}     # a1 vs a2: 0 matches
collecting slots per agent: {'a0': 20018, 'a2': 4059, 'a1': 3949}
slot kinds: {('a0','ckpt'): 3974, ...}                                         # a1/a2 checkpoints: never used
self_play phase, agents alpha+beta -> slots per agent: {'alpha': 3200}        # beta: 0
```

**Impact:** Multi-agent training is effectively a single main agent plus sparring partners whose learners are biased. Scenario (a) is impossible, and all league results are skewed.

**Fix:**
- Make the matchmaker produce a *population-level* schedule. For example, `generate_matches(num_envs)` assigns each env an "owner" learner, either round-robin or weighted by a per-agent `data_share`. It then builds that owner's match (solo or arena).
- At minimum, have the launcher cycle `agent_id` over all trainable agents per env (or per worker).
- Add a test that every trainable agent receives collecting slots in both phases.

### R4-02 — HIGH — correctness — fixed seat assignment: the learner always plays seat 0
**Location:** `matchmaker.py:105-111` and `:196-199` (slot 0 is always latest), `:214-221` (`agents[i % 2]` puts `agent_id` at seat 0)

**What is wrong:**
- In every match against a historical checkpoint, the learning policy sits in seat 0 and the checkpoint in seat 1.
- In every arena match, the primary agent takes the even seats and the opponent the odd seats.
- Nothing shuffles seats.

**Evidence (VERIFIED-BY-RUNNING, v1 §A/§C):**
- `seat of the learning policy in matches vs a checkpoint: {'seat0': 3187}` (100%)
- `seat-0 occupant: {'a0': 16000}`
- Tic-tac-toe has a first-mover edge. Even with an identical random network in both seats, seat 0 scores 0.529 vs 0.471 over 2,000 games (v3 §C).

**Impact:**
- In seat-asymmetric games (Lux has asymmetric spawns, turn order and so on), the agent never learns the other seat against past opponents.
- Win rates, PFSP priorities and ELO all mix in the seat advantage. The arena's "agent A beats B" is really "seat 0 beats seat 1".

**Fix:** Shuffle the slot list (`random.shuffle(slots)`) after building each match. Optionally offer `seat_balance: paired`, which emits mirrored match pairs. Then log per-seat win rates.

### R4-03 — HIGH — bug — the worker loses outcomes when an agent fills more than one seat
**Location:** `rollout_worker.py:648-653`, plus the dead aggregation in `coordinator.py:150-156`

**What is wrong:** `player_outcomes[f"{agent}:{net}"] = outcome` overwrites earlier entries. In a 4-player arena match `[a0, a1, a0, a1]` (the only arena pattern the code produces, see R4-09), and in self-play where `latest` fills several seats, only the **last** seat's outcome survives. The coordinator's "average over the agent's seats" logic never sees duplicates.

**Evidence (VERIFIED-BY-RUNNING, v1 §D):**
```
per-slot outcomes: [1.0, 0.0, 0.0, 0.0] -> reported player_outcomes: {'a0:latest': 0.0, 'a1:latest': 0.0}
win rate a0 vs a1 after a0 WON the FFA: 0.0   ELO: {'a0': 1200.0, 'a1': 1200.0}
```

**Impact:** In N-player arenas, ratings and PFSP inputs are effectively random noise.

**Fix:** Report a list of per-seat records `[(seat, player_id, outcome, reward, rank)]`, not a dict keyed by player. Aggregate in the coordinator, or better, feed ranks directly to a multi-player rating system (see R4-08).

### R4-04 — HIGH — bug — the win-rate tracker gets absolute N-player scores instead of pairwise results, and truncates them
**Location:** `coordinator.py:164-167` (`self._win_rates.record(a, b, outcome_a)`), `ratings.py:96-97` (`int(outcome_a * 2)`)

**What is wrong:**
- For a pair (a, b) the coordinator records `outcome_a`, which is a's absolute FFA score, not whether a beat b.
- `record()` then truncates `outcome*2` to an integer, so a's credit and b's credit no longer add up to the number of games.
- The ELO update in the same loop *does* compare `outcome_a` with `outcome_b`. The two statistics therefore disagree.

**Evidence (VERIFIED-BY-RUNNING, v1 §H):**
```
ranks a=2nd,b=4th,c=1st,d=3rd -> outcomes [0.667, 0.0, 1.0, 0.333]
a beat b -> wr(a,b)=0.5  wr(b,a)=0.0                  # should be 1.0 / 0.0
a,b tied (both lost to c) x10 -> wr(a,b)=0.0  wr(b,a)=1.0   # tie recorded as a 10-0 sweep, depends on dict order
ELO a,b: 1104.7 1106.5                                 # same results, different ELO (sequential updates)
```

**Impact:** PFSP (`_select_opponent`) reads these numbers. In any game with more than 2 players, or in 2-player games where outcomes are averaged, opponent prioritisation follows dict insertion order rather than strength.

**Fix:** Pass the pairwise result `s = 1 if oa > ob else 0 if oa < ob else 0.5` to `record()`. Store floats, not `int(x*2)`. Compute all ELO deltas from the pre-match ratings, then apply them together.

### R4-05 — HIGH — missing-feature / bug — frozen and scripted agents cannot be used end-to-end
**Location:**
- `agent_pool.py:63-93`: `register_frozen` and `register_scripted` are never called anywhere in `src/`.
- `config.py:262-267`: `AgentConfig` has no `type`, `class`, `path` or `networks_for_checkpoint` field.
- `matchmaker.py:176`: PFSP reads only `list_trainable()`.
- `rollout_worker.py:291-312, 473-475`: `networks_by_agent` holds only trainable agents.
- `launcher.py:234-244`: checkpoints are loaded only from this run's `CheckpointManager`.

**What is wrong:**
- There is no config path to declare a scripted bot or an external frozen checkpoint.
- Even if a `MatchConfig` contained one, the worker would raise `KeyError` at `networks_by_agent[aid]`.
- There is no `ScriptedPolicy` interface at all.
- An external checkpoint with a different architecture cannot be built, because each worker network factory comes from the owning *trainable* agent's config.
- Pydantic silently ignores unknown keys, so a user who tries `type: scripted` gets an extra trainable agent and no error.

**Evidence (VERIFIED-BY-RUNNING, v2 §C):** `agents: {beta: {...}, bot: {type: scripted, scripted_class: my.Bot}}` produces `trainable ids: ['beta', 'bot']`. Grep confirms no callers of `register_frozen`, `register_scripted` or `list_all`.

**Impact:** Owner priority #2 (an arena with different weights, different architectures and scripted bots) is entirely unsupported. CLAUDE.md and README describe it as implemented ("Agent pool can be TrainableAgent / FrozenCheckpoint / ScriptedAgent").

**Fix:**
- Define a `PlayerSpec` union in config (`trainable | frozen | scripted`, see the Design section).
- Add a `Policy` protocol in the worker: `act(obs_batch, mask, hidden, infos) -> actions`, implemented by `NeuralPolicy` and `ScriptedPolicy`.
- Key the network pool by a `player_id` string, with an architecture spec per player. Store `networks` config in each checkpoint's `meta.json` so frozen snapshots carry their own architecture.
- Set `model_config = ConfigDict(extra="forbid")` on all config models.

### R4-06 — HIGH — bug — restarting in the same checkpoint dir deletes the new run's checkpoints
**Location:** `checkpoint_manager.py:60-76` (`_scan_existing` loads the previous run), `:87` (`ckpt_v{policy_version}`), `:90` (`mkdir(exist_ok=True)`), `:117-122` (FIFO evicts `index[0]` with `rmtree`)

**What is wrong:** `policy_version` restarts at 0 in three situations:
- a second run with the same `checkpoint.dir` (the default is `./checkpoints`);
- `resume_from: some.pt` (`launcher.py:287` sets `policy_version: 0`);
- `run-learner` in distributed mode, which never resumes.

When the new run saves `ckpt_vN` while the old `ckpt_vN` is still indexed:
1. The old directory is overwritten.
2. The index gains a duplicate entry.
3. FIFO eviction pops the old entry and `rmtree`s the shared path, **deleting the checkpoint it just wrote.**

**Evidence (VERIFIED-BY-RUNNING, v2 §A, pool_size=3):**
```
index after new save: ['ckpt_v200', 'ckpt_v300', 'ckpt_v100']
on disk: ['ckpt_v200', 'ckpt_v300']
LOAD FAILED: Checkpoint not found: .../agent_0/ckpt_v100/model.pt
```

**Impact:**
- New checkpoints are silently lost.
- The matchmaker keeps sampling ids that no longer exist; it falls back to `latest` (R4-19).
- Old-run checkpoints, possibly with a *different architecture*, become self-play opponents and crash `load_state_dict` in the worker.
- `tests/test_multi_agent.py::test_multi_agent_pipeline` writes `./checkpoints` into whatever cwd it runs from, so running it twice from the repo root sets up exactly this collision.

**Fix:**
- Put checkpoints in a per-run namespace: `checkpoint.dir/<run_name>/<agent>/`.
- Continue versions from `max(existing)+1`, or refuse to start when `ckpt_vN` already exists.
- Make eviction compare `checkpoint_id`, never a raw path shared with a newer entry.
- Write atomically: write to `ckpt_vN.tmp/`, then `os.replace`.
- Keep a manifest of which run produced each checkpoint.

### R4-07 — HIGH — missing-feature — ratings and win rates are never logged, never persisted, and absent in self-play
**Location:**
- `coordinator.py:180-187`: `get_ratings_summary` has no caller in `src/`.
- `launcher.py:558-564`: results are consumed but nothing is logged.
- `metrics/wandb_logger.py`: no rating logging.
- `coordinator.py:45`: `EloRating()` K-factor is not configurable.
- `agent_pool.py:37-39`: the `elo`, `win_rates` and `match_counts` fields are dead.
- `launcher.py:547-552`: `maybe_save_checkpoint` is called without `metrics`, so `meta.json` never records rating or win rate.

**Evidence (VERIFIED-BY-RUNNING, v1 §E):** In a self-play-only run, 100 games of latest vs `ckpt_v10` produce `ELO table: {}`. Same-agent pairs are skipped, so the most common training mode yields **no skill signal at all**. Code search finds no rating or win-rate matrix output anywhere.

**Impact:**
- Owner priority "monitoring of all results" is unmet.
- When the process exits, all match history is gone, apart from a 10k-entry in-memory deque.
- There is nothing to choose a best checkpoint from.

**Fix:**
- Rate per player snapshot, so `agent_0:latest` vs `agent_0:ckpt_v100` *is* informative.
- Log ELO/OpenSkill, the win-rate matrix and per-seat win rates every `log_interval`.
- Append every match to `league/matches.jsonl` (players, seats, ranks, rewards, length, versions).
- Snapshot `league/ratings.json` with each checkpoint and write the rating into `meta.json`.

### R4-08 — MEDIUM — design / correctness — the rating model is not sound for a non-stationary, N-player league
**Location:** `coordinator.py:128-174`, `ratings.py:16-59`

**What is wrong:**
- **Unit of rating.** Ratings are keyed by base `agent_id`. `latest` of a learning agent and all of its historical snapshots are pooled into one number that keeps drifting. Frozen snapshots never get their own fixed rating.
- **Online ELO is order-dependent.** The same 10–10 record gives a/b 1135/1265 or 1265/1135 depending on order (v2 §B).
- **N-player handling.** The decomposition updates pairs sequentially within one match (order effects), and the effective K grows with N: one 8-player FFA win moves the winner +97.8 vs +16 for a 2-player win (v2 §B, VERIFIED-BY-RUNNING).
- **Ties.** Under reward fallback, `outcomes_from_rewards` (`outcomes.py:23-30`) collapses every non-winner to 0. All losers "draw" each other and rank information is lost.

**Impact:** The numbers cannot rank checkpoints across time. In FFA they are dominated by noise and K inflation.

**Fix:**
- Use **per-snapshot ratings** with a Plackett–Luce / TrueSkill-style multi-player model (e.g. `openskill`) for online tracking.
- Periodically **refit offline**, order-independently, from `matches.jsonl` with Bradley–Terry or Elo-MLE, using bootstrap CIs. That refit is the selection-grade number.
- Fall back on rewards as ranks: `scipy.stats.rankdata(-rewards)` instead of winner-takes-all.
- Make K and the initial rating configurable.

### R4-09 — MEDIUM — design — PFSP is shallow: agent-level, all-time, hard-only, and at most 2 agents per match
**Location:** `matchmaker.py:170-244`

**What is wrong:**
- **Sampling unit.** Arena opponents are always another trainable agent's *latest* weights. Historical snapshots of other agents, frozen agents and scripted bots are never PFSP candidates. Solo matches pick own checkpoints *uniformly* (`random.choice`), not by PFSP.
- **Win-rate window.** Win rates are all-time counts. After 1,000 early losses followed by 200 straight wins, wr(a0, a1) = 0.167, and PFSP still sends a1 about 65% of the time (v1 §F, VERIFIED-BY-RUNNING: `{'a1': 1566, 'a2': 834}`).
- **Weighting.** Only `(1-wr)^p` exists. CLAUDE.md also promises `x(1-x)` (variance) weighting.
- **Unseen opponents.** They get wr = 0.5, so with p = 1 they have the same priority as a 50% opponent. That is acceptable, but it is not configurable.
- **Arena size.** `_arena_match` uses exactly 2 distinct agents even with `num_players=4` and 4 trainable agents. v1 §D: `distinct-agent counts per 4p arena match: {2: 1600}`, all of the form `a0,aX,a0,aX`. CLAUDE.md says "all slots filled with TRAINABLE agents from the pool".
- **Dead code.** `total == 0` is unreachable because of `max(1e-6, …)`.

**Impact:** Strategy cycling is not prevented, since the league has no memory of other agents' past strategies. N-player FFA arenas are degenerate 2-agent mirrors.

**Fix:** See the Design section. In short:
- PFSP over *player snapshots*, with a per-role weighting function;
- an EMA or sliding-window win rate (e.g. last 200 games or half-life);
- for N players, sample N-1 opponents without replacement;
- a configurable unseen prior.

### R4-10 — MEDIUM — design / bug — checkpoint retention and lifecycle
**Location:** `checkpoint_manager.py:117-122`; `rollout_worker.py:179-192` (`_apply_command` only adds networks); `launcher.py:239-244` (reloads from disk on every refresh)

**What is wrong:**
1. Pure FIFO. The best or milestone checkpoint is deleted after `pool_size` saves. There is no pinning, no `keep_every`, no `keep_top_k`. The self-play opponent pool and on-disk retention are the same thing. Both should exist, but separately.
2. Workers never drop evicted checkpoint networks. Memory grows by one network per checkpoint for the whole run (VERIFIED-BY-READING).
3. `_derive_worker_configs` calls `torch.load` for every referenced checkpoint, for every worker, on every refresh (default every 30 s), before it diffs which ones were already sent.
4. Writes are not atomic. `meta.json` has no architecture, config hash, run id, parent lineage or rating.
5. The checkpoint loaded for resume is not restorable as a full state: no ELO, no kickstart step, no LR-schedule step (see R4-15).

**Fix:**
- Separate a `CheckpointStore` (retention policy: `keep_last`, `keep_every`, `pinned`, `keep_top_k_by_rating`) from an `OpponentPool` (what can be sampled).
- Cache `state_dict`s in the main process.
- Send `evict` deltas in `WorkerCommand` so workers can release networks.
- Use atomic directory rename.
- Enrich `meta.json` (`networks` config, `run_id`, `parent`, `rating`).

### R4-11 — HIGH — bug — eval silently bypasses the LSTM/GRU, so recurrent agents are evaluated as a different policy
**Location:** `eval.py:244-247` (`net.act(obs, action_mask=…, deterministic=…)`, no `hidden`), `actor_critic.py:80` (the recurrent trunk is skipped when `hidden is None`)

**Evidence (VERIFIED-BY-RUNNING, v3 §D):** With an LSTM network, `hidden=None returns new_hidden: None | max |logit diff| (no-LSTM vs LSTM path): 0.334`. Because `recurrent_hidden_size == latent_dim`, the shapes match, so nothing raises an error.

**Impact:** Any eval or "best checkpoint" decision for recurrent agents is invalid, with no warning. The same bypass exists in kickstarting (R4-15).

**Fix:** Track hidden state per (env, seat) in eval: initialise at reset, carry it across steps, reset it on done. Better still, reuse the worker's inference grouping (see R4-14), so eval and training share one code path. Also make `forward()` raise if `self.recurrent is not None and hidden is None`.

### R4-12 — MEDIUM — missing-feature — CLI eval supports only one architecture and only neural agents
**Location:** `cli.py:121-140` (one `network_factory` built from global `cfg.networks`; `network_factories` is never passed), `eval.py:83-124`

**Evidence (VERIFIED-BY-RUNNING, v3 §E):** Loading a GRU-agent state_dict into the global factory raises `Error(s) in loading state_dict for ActorCriticNetwork`. There is no way to put a scripted bot into eval.

**Fix:**
- Accept agent specs as `name=agent_id@path` (resolving the architecture from `cfg.agents[agent_id]`) or `name=path` with the architecture read from `meta.json`.
- Accept `name=scripted:module.Class`.

### R4-13 — MEDIUM — bug — eval handles draws inconsistently; the reversed entry's CI is wrong
**Location:** `eval.py:131-141, 313-319`

**What is wrong:**
- `win_rate_a = wins/total` and the Wilson CI is computed on wins only, so draws count as losses.
- The mirrored `(b, a)` entry sets its CI to `[1-hi, 1-lo]`, which is a CI for *(wins_b + draws)*, while its point estimate is `wins_b/total`.

**Evidence (VERIFIED-BY-RUNNING, v3 §A):** With 10 wins, 10 losses and 80 draws, B gets `wr=0.10 CI=[0.826,0.945]`. The point estimate lies outside its own CI. Tic-tac-toe and many competition games are draw-heavy.

**Fix:** Report the score `(W + D/2)/N` with a proper CI (Wilson on the score, or a bootstrap / trinomial). Compute each direction's CI from its own counts.

### R4-14 — HIGH — missing-feature — no statistically sound way to select the best agent or checkpoint
**Location:** `eval.py` overall; absent from coordinator and CLI

**What is missing:**
- No tournament runner over a checkpoint directory.
- No fixed benchmark set (gauntlet against scripted bots and past bests).
- No rating fit from the eval matrix, no ranking output, no "is A better than B" test.
- No multiple-comparison control, no CSV/JSON output.

**Further eval weaknesses:**
- `--num-matches` help says "Matches per agent pair" (`cli.py:102`) but it is the *total* (`eval.py:221`).
- Seats are random (`eval.py:203-209`), not paired. That is unbiased but has higher variance than seat-swapped pairs with shared seeds.
- In FFA, pairwise results drawn from the same match are correlated, but the CIs treat them as independent.
- Eval stops at a global count and discards in-progress episodes. That is a slight bias toward short episodes (suspected, low).
- Eval duplicates the worker's inference loop instead of reusing it. This has already diverged: no hidden state, no active-flag handling.

**Fix:** Add `colosseum tournament` (see the Design section).

### R4-15 — HIGH — missing-feature / bug — warm start is global, all-or-nothing, and partly broken
**Locations and problems:**
- **`config.py:177-194`, `launcher.py:267-307`.** `training.resume_from` and `kickstart_teacher` are *global*. `AgentConfig` cannot override `training`, so every agent loads the same `.pt` file:
  - heterogeneous architectures crash in the learner (`learner.py:67`, strict `load_state_dict`);
  - when that one learner dies, workers block forever on its full queue (`rollout_worker.py:530-535`), and the monitor waits for all learners.
- **No per-agent init.** There is no "init agent B from agent A's checkpoint", no `strict: false` / shape-matched partial loading for changed architectures, and no "reset exploiter to init".
- **`bc/kickstart.py:84-93`.** Kickstarting calls `encoder → policy` directly and **bypasses the recurrent trunk**, the same defect as R4-11.
- **Kickstart teacher limits.** The teacher is built from the *student's* config (`launcher.py:166`), so it must share the student's architecture. It cannot be a scripted expert. CLAUDE.md "Mode B: scripted expert provides actions" is not implemented.
- **Kickstart loss details.** Masks are not applied. The loss is KL(student‖teacher), which is mode-seeking. The kickstarting paper uses cross-entropy H(teacher, student), i.e. the KL(teacher‖student) direction; worth making configurable.
- **`learner.py:66-77`.** On resume, the LR scheduler and kickstart lambda start over from step 0 while `train_step` continues from the resumed `policy_version`. The LR schedule is truncated (it never decays to its end value) and kickstarting restarts at full strength.
- **`distributed.py:222-233`.** `run-learner` ignores `resume_from` entirely, which also triggers R4-06.
- **`resume_from: ckpt_vN`** resolves within *each agent's own* directory. It works only if every agent happens to have a checkpoint with the same id; otherwise the agent silently starts from scratch (warning only).
- **Phases.** `phase: bc` is accepted but behaves like `self_play` (`coordinator.py:81`). There is no in-run phase transition.

**Fix:** Per-agent `init:` and `kickstart:` blocks (Design section). Have the learner restore `{lr_scheduler, kickstart_step}` from the checkpoint. Make teacher type and loss direction configurable.

### R4-16 — MEDIUM — ux / doc-mismatch — per-agent overrides replace whole sections, and typos are silently ignored
**Location:** `config.py:306-314`; README "agents" section ("override specific fields")

**Evidence (VERIFIED-BY-RUNNING, v2 §C):** Global settings are `lr_schedule=constant`, `batch_chunks=2`, `queue_size=16`. `agents.beta: {algorithm: {learning_rate: 1e-4}, learner: {device: cpu}}` gives beta `lr_schedule: linear, batch_chunks: 16, queue_size: 64`: every other field silently resets to the model defaults. Unknown keys (`pfsp_weighting`, the typo `latset_prob`) are accepted without error.

**Fix:** Deep-merge `overrides.model_dump(exclude_unset=True)` onto the global dict, and use `extra="forbid"`. `networks` overrides currently also require all three class paths, so allow partial overrides there too.

### R4-17 — MEDIUM — doc-mismatch — code vs CLAUDE.md / README
| Claim | Reality |
|---|---|
| Agent pool supports Trainable / Frozen / Scripted; PFSP over pool | Only trainable is reachable (R4-05) |
| Arena: "all slots filled with trainable agents"; "all trainable agents collect" | Exactly 2 agents per arena (R4-09); others never meet (R4-01) |
| PFSP `f(x)=x(1-x)` balanced option | Not implemented |
| "ELO and pairwise win rates tracked automatically"; "WandB: ELO over time, win-rate matrix" | Tracked in memory only; never logged (R4-07) |
| Online BC Mode B with scripted expert | Only a same-architecture neural teacher, global (R4-15) |
| README Phase 1 "Use kickstarting…" with a YAML that shows `resume_from` | That is init, not kickstarting; `kickstart_*` keys are undocumented in README |
| README `resume_from`: "Checkpoint ID to resume from" | Also accepts a `.pt` path; applies to all agents |
| Eval: "agents assigned to slots round-robin"; `--num-matches` "per pair" | Random sampling; total match count |
| `MatchResult.match_id` "matches the MatchConfig's match_id" | Worker invents `w{w}_e{e}_{step}`; `MatchConfig.match_id` and `env_config` are dropped (`launcher.py:226-257`) |
| `phase: bc` | Runs RL self-play |
| Distributed docs (CLAUDE.md "dynamic add/remove", PFSP) | `distributed.py` uses a static round-robin `slot_agent_map`, no coordinator, no results, no ratings, no historical opponents (documented only in the module docstring) |

### R4-18 — MEDIUM — missing-feature — distributed mode has no league
**Location:** `distributed.py:331-337, 288-304`

**What is wrong:** The slot map is fixed when workers launch: all seats collect, there are no checkpoint or frozen opponents, and no `results_queue`. This is honest in the module docstring, but it means multi-machine runs get none of this subsystem.

**Fix:** Run the coordinator as a gRPC service (`RequestMatches`, `ReportResults`, `GetCheckpoint`). Workers poll it the same way they already drain `command_queue`.

### R4-19 — LOW — bug — a missing checkpoint falls back to `latest` but keeps `collect=False`
**Location:** `launcher.py:245-251`

**Evidence (VERIFIED-BY-RUNNING, v1 §G):** `([['latest','latest']], [[True, False]], …)`. The current policy plays without its data being collected, and the result is then recorded as a same-agent game and discarded.

**Fix:** Re-sample another checkpoint, or set `collect=True`.

### R4-20 — LOW — design — matchmaking state is static per phase
**What is wrong:**
- `setup_matchmaker` is re-created on every checkpoint save. In `self_play` it is chosen from the *saving* agent's checkpoint count.
- There is no curriculum (e.g. `latest_prob` annealing, switching to league after X steps), so a BC → self-play → league pipeline needs three runs and manual handover. That handover hits R4-06 and R4-15.
- `latest_prob` is per-seat-independent for seats ≥ 1. In N-player games, the expected number of historical opponents is `(N-1)(1-latest_prob)`, which is fine but undocumented.

### R4-21 — MEDIUM — test-gap
- `test_ratings.py::test_coordinator_match_reporting` uses keys `"a0"`, not the real `"a0:latest"` format, and only 2 players.
- No test covers any of the following:
  - multi-agent coverage (R4-01);
  - seat balance (R4-02);
  - duplicate seats or N-player outcomes (R4-03/04);
  - restart in an existing checkpoint dir (R4-06);
  - recurrent eval (R4-11);
  - heterogeneous-architecture league or eval;
  - per-agent override merge (R4-16);
  - draw-heavy eval CIs (R4-13).
- `test_multi_agent_pipeline` and the milestone2 pipeline test write `./checkpoints` into the cwd (the repo root when run normally). That leaks state between runs and interacts with R4-06.
- **Fix:** Add property-style tests: sample 10k matches and assert per-agent / per-seat shares; round-trip N-player results through the real worker reporter.

---

## Scenario support (today)

| Scenario | YAML a user would write | Works | Missing / broken |
|---|---|---|---|
| **(a) BC → self-play → league, 3 agents with different architectures** | Per arch: `colosseum bc -c cfg_X.yaml` (the BC CLI only uses global `networks`, so 3 configs). Then 3 separate single-agent self-play runs with `resume_from: bc_X.pt`. Then a league config with `agents: {A: {networks: …}, B: {networks: …}, C: {networks: …}}`, `phase: league`, `resume_from: ckpt_v5000` | Offline BC per architecture; single-agent self-play from BC weights; heterogeneous `networks` overrides build and train | A multi-agent self-play phase starves B and C (R4-01). League init cannot be per agent: a `.pt` path crashes the other architectures, and a ckpt id must coincide across agents (R4-15). In league, B and C never meet and never see history (R4-01/09). Kickstarting is global and same-architecture (R4-15). Restarting phases in the same dir loses checkpoints (R4-06). No ratings output (R4-07). |
| **(b) 2 trainable + 2 scripted + 5 frozen from a previous run, 4-player FFA** | Would need something like `agents: {t1: {}, t2: {}, bot1: {type: scripted, class: …}, old1: {type: frozen, path: …}}` | Nothing beyond 2 trainable agents | No config for scripted/frozen; silently turned into trainable agents (R4-05). Worker cannot run them. PFSP excludes them. 4-player arenas are `a,b,a,b` (R4-09) with corrupted outcomes (R4-03/04). |
| **(c) Main agent + exploiter (AlphaStar-style)** | `agents: {main: {}, exploiter: {}}`, `phase: league`, `self_play_ratio: 0.5` | By accident: `agents[0]` = main gets solo + arena, and the exploiter only plays main | Main also collects and trains on every exploiter game (no "frozen main for the exploiter"). No exploiter reset to init. No variance weighting. Exploiter snapshots never join main's opponent pool. No roles in config. |
| **(d) Final tournament to choose the submission** | `colosseum eval -c cfg.yaml -a v1:…/model.pt -a v2:… --num-matches 4000 [--deterministic]` | Same-architecture, feed-forward checkpoints; random seats; Wilson CI per pair | No LSTM (R4-11), no mixed architectures (R4-12), no scripted gauntlet, wrong CIs with draws (R4-13), no ranking / BT fit / significance, `--num-matches` is total, FIFO may already have deleted the best checkpoint (R4-10), no stored ratings to shortlist from (R4-07). |

---

## Subsystem assessment

The building blocks exist: `AgentPool` with three types, CheckpointManager, PFSP formula, ELO, win-rate matrix, Wilson CI, env-authoritative outcomes, runtime match refresh, and per-agent network overrides. The wiring between them is the weak part:

- Matchmaking is driven from a single agent.
- The pool's non-trainable types are dead code.
- Results lose information on the way from worker to coordinator (dict-key collisions, absolute vs pairwise scores).
- Ratings live and die in RAM.
- Eval is a separate re-implementation with its own bugs.

For a "competition in 2 weeks" framework, this subsystem is where the framework adds the most value over a single-agent PPO script. It is currently the least trustworthy part, and the existing tests pass without exercising any of the failure modes.

## Proposed design (keeps the current IMPALA structure; it is mostly data model + matchmaker rewrite)

### 1. Data model
```text
PlayerId     = "<agent>@<version>" | "<agent>@latest" | "bot:<name>"
PlayerSpec   = Trainable(agent_id, networks, algorithm, learner, init, kickstart, role)
             | Frozen(agent_id, path | ckpt_ref, networks | from meta.json)
             | Scripted(name, class_path, kwargs)
Snapshot     = immutable (agent_id, version, path, networks_cfg_hash, parent, created_step)
LeagueState  = {players: {PlayerId: PlayerRecord}, ratings: {PlayerId: (mu, sigma)},
                pair_stats: {(PlayerId, PlayerId): EMA wins/games}, matches.jsonl}
MatchRecord  = {match_id, seats: [{player_id, seat, rank, outcome, reward}], length, ts}
```

- **Worker.** It holds `policies: dict[PlayerId, Policy]` behind an LRU cap. Its inputs are `WorkerCommand(assignments=[[PlayerId…]…], collect=[[bool…]…], add={PlayerId: PolicyBlob}, evict=[PlayerId])`. A `Policy` is either neural (state_dict + `networks` cfg) or scripted (class path). Inference stays grouped by `PlayerId`.
- **Results.** Seat-level records with ranks. No dict keyed by player.

### 2. Config schema sketch
```yaml
league:
  players:
    main:
      kind: trainable
      role: main
      networks: {...}                       # partial override, deep-merged
      init: {from: runs/bc/main.pt, strict: false}          # or {from_player: other@ckpt_v300}
      kickstart: {teacher: runs/bc/main.pt, lambda: 1.0, decay_steps: 20000, direction: teacher_student}
    main_exploiter:
      kind: trainable
      role: main_exploiter
      target: main
      init: {from_player: main@init}
      reset_every_steps: 4000               # reset to init when converged/timeout (AlphaStar)
    league_exploiter: {kind: trainable, role: league_exploiter}
    greedy: {kind: scripted, class: my_game.bots.Greedy}
    prev_best: {kind: frozen, path: runs/r1/main/ckpt_v9000, networks: from_meta}
  snapshot_every_steps: 500                 # adds a frozen Snapshot to the pool
  data_share: {main: 0.5, main_exploiter: 0.25, league_exploiter: 0.25}   # owner sampling per env
  roles:
    main:             {self_play: 0.35, pfsp: 0.50, vs_exploiters: 0.15, weighting: hard, p: 2}
    main_exploiter:   {vs_target_latest: 0.5, vs_target_snapshots: 0.5, weighting: variance}
    league_exploiter: {pfsp_all_snapshots: 1.0, weighting: hard}
  opponents_per_match: n_players_minus_1    # FFA: sample without replacement
  seat_assignment: shuffle                  # or: paired (mirror matches)
  collect: owner_only                       # or: all_trainable (current arena behaviour)
  win_rate: {kind: ema, halflife_games: 200, unseen_prior: [1, 2]}   # (wins, games)
ratings: {online: openskill_pl, refit: bradley_terry, persist: league/ratings.json}
checkpoint:
  dir: runs/${run_name}/ckpt
  retention: {keep_last: 20, keep_every: 5000, keep_top_k_by_rating: 5, pinned: []}
  atomic: true
selection:
  gauntlet: [greedy, prev_best, main@best]
  tournament: {games_per_pair: 400, seat_swap: true, paired_seeds: true, deterministic: [false, true]}
  gate: {metric: score_lcb, threshold: 0.5, alpha: 0.05}
```

### 3. Matchmaker algorithm (simple)
For each env:
1. Sample an owner from `data_share`.
2. Sample a match type from the owner's role mix.
3. Sample N-1 opponents from that type's candidate set. Weights are `f(ema_wr(owner@latest, cand))`, with `hard=(1-x)^p` or `variance=x(1-x)`, and an unseen prior.
4. Shuffle seats.
5. `collect = (player == owner@latest)`, optionally also other trainables' latest.

For **N-player FFA**, compute the pairwise EMA from rank comparisons. A practical candidate weight is the mean of `f` over the opponents' pairwise win rates against the owner.

Policy defaults:
- When there is a single agent, this reduces to OpenAI-Five-style 80% latest / 20% PFSP-weighted past.
- Scripted bots should keep a floor probability (e.g. 5%) as anti-forgetting regression tests.

### 4. Ratings and selection
- **Online.** OpenSkill (Plackett–Luce) per `PlayerId`, updated from ranks. Frozen snapshots and bots keep stable ratings, and `@latest` is a moving entry. Log μ−3σ and the per-seat win rate.
- **Offline.** `colosseum league refit` runs Bradley–Terry / Elo-MLE over `matches.jsonl` with bootstrap CIs. It is order-independent and used for selection and plots.
- **`colosseum tournament`:**
  - takes candidates: top-k by online rating + pinned + gauntlet;
  - runs round-robin (2-player) or balanced incomplete blocks (FFA);
  - uses seat-swapped paired games with shared seeds;
  - supports both stochastic and greedy action modes;
  - reuses the worker's `Policy` / inference path (recurrent-correct, mixed architectures, scripted);
  - outputs a BT ranking with CIs, a score matrix (W + D/2) and the P(best) per candidate from the bootstrap;
  - writes `selection.json` and pins the winner.

### 5. Warm start
- **Per-player `init`.** `.pt` path, `player@version`, or `latest_of: <agent>`. Use `strict: false`, which loads tensors whose name and shape match and reports what was skipped and why.
- **Resume.** Restore `{model, optimizer, lr_scheduler.last_epoch, kickstart.step, policy_version}`. The learner's LR schedule continues from there. The league restores `ratings.json`, `pair_stats` and the snapshot registry from the run dir.
- **Kickstart teacher.** Any `Policy`, including scripted ones (cross-entropy on teacher actions) or neural ones of a different architecture (KL on action distributions). It passes hidden state for recurrent students and honours action masks.

### Priorities
- **P0 (correctness, about 1–2 days).**
  - R4-01: matches for every agent;
  - R4-02: seat shuffle;
  - R4-03/04: seat-level results and pairwise win-rate/ELO;
  - R4-06: run namespace + version continuation;
  - R4-11: eval hidden state;
  - R4-13: draw-aware CIs;
  - R4-16: deep-merge overrides + `extra="forbid"`;
  - log ratings and win-rate matrix (part of R4-07).
- **P1 (owner priorities).**
  - `PlayerSpec` config with frozen/scripted players, worker `Policy` abstraction and snapshot-level PFSP (R4-05/09);
  - per-agent `init` / `kickstart` (R4-15);
  - retention with pinning + `meta.json` architecture (R4-10);
  - `matches.jsonl` + `ratings.json` persistence;
  - `colosseum tournament` (R4-14).
- **P2.**
  - roles (main / main-exploiter / league-exploiter with resets);
  - OpenSkill + BT refit;
  - coordinator as a gRPC service for distributed leagues (R4-18);
  - in-run phase schedule (R4-20).

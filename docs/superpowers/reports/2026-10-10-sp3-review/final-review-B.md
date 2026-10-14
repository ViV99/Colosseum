> Origin: `.superpowers/sdd/2026-10-10-sp3-league/final-review-B.md` (SP3 SDD workspace, git-ignored, deleted after the merge). Copied 2026-10-10.
> Frozen record; do not edit. Copied verbatim (no absolute scratch paths to replace).

# SP3 final review — Area B: league, ratings, metrics, distributed guards

Reviewer: area B (Opus/high). Range `c16fe5e..c31eda9` (branch `sp3-league`).
Scope read end to end: `src/colosseum/league/{__init__,schedule,pfsp,base,lineups,mixture}.py`,
`coordinator/coordinator.py`, `coordinator/ratings.py`, `metrics/aggregator.py`, `metrics/hub.py`, the matchmaking
parts of `core/config.py` and `core/validation.py`, `distributed.py` (scope check, reduction, retention),
`configs/examples/*.yaml` vs `tests/fixtures/sp2/configs/*`, `scripts/team_tag_anchors.py`, the area's tests, the
launcher's coordinator wiring (`_build_coordinator`, `_refresh_worker_matches`, `_resolve_lineups`), the SP2
`LineupMatchmaker` at `c16fe5e` for the compatibility comparison, and `docs/LEAGUE_GUIDE.md` / README for my area.
Experiments: `.superpowers/sdd/2026-10-10-sp3-league/final-review/B/pfsp_resurrect.py`; `colosseum validate` on
`team_tag`, `unit_harvest_league`, `predator_prey`, `tic_tac_toe_multi` (all valid, printouts as documented); the
area's unit tests (210 passed, 5.9 s).

## Strengths

- **Matchmaker matches spec block 5 step by step.** `MixtureMatchmaker` (`league/mixture.py`) implements layout
  draw, owner team, independent opposing cores with the four categories, `rivals` folded into `latest` when the owner
  cannot play the team, proportional spreading of empty categories, the runtime fallback (`latest`, then `anchors`,
  source `fallback`, one warning per (agent, layout)), `teammates: self | mixed`, role fillers (trainable latest first,
  then O's anchors with weight > 0 now), collect only on trainable latest, and `permute_seats` moving assignments as
  objects so `source` survives. I walked the asymmetric (`predator_prey`), FFA, team and cooperative cases; all
  produce the spec's lineups.
- **The structural check is sound across schedules.** `validate_matchmaking` checks role coverage and fillability at
  0 and every breakpoint; because shares and anchor weights are piecewise linear between the same breakpoints, a
  category positive and available at one endpoint stays so on the open interval, so point checks are sufficient.
  `effective_mix` (validate) and `_opponent_core` (runtime) use the same availability rules (A26).
- **PFSP statistics are attributed correctly.** `member_pairs` now carries both network ids; `PfspStats._record`
  keeps only pairs with O@latest on one side and anything but O@latest on the other, orients the score per side,
  splits team weights as SP2 ratings do, and uses the specified EMA step `1 - 2^(-w/h)` with per-agent half-lives.
  Snapshot candidates of every E_T agent (own, opponent's, others') are weighted by the owner's own score.
- **Ratings:** scripted/frozen agents are separate ELO / win-rate / role-win-rate entities, snapshots still fold into
  the base agent, `wr_vs_past` unchanged; the PFSP table rides in `ratings.json` and is kept out of WandB rows.
- **Every lineup is checked** (`check_lineup`): layout, seat count, roles, existence of stored snapshots,
  `fixed` only for scripted/frozen, collect only on trainable latest, known `source`; the same check runs in
  `validate` on a custom matchmaker's lineups, and evicted snapshots cannot be drawn because candidates come from the
  store's in-memory index.
- **SP2 translation is exact and contained:** knobs only in the raw global section, rounded (P3), never stored,
  clash with `opponents`/`pfsp` = ConfigError, per-agent knobs rejected; `merge_matchmaking` (A18) replaces schedules
  and `anchors`/`layouts` whole. `test_sp3_example_configs_v3` proves each live example's matchmaking and checkpoint
  sections equal the translation of its SP2 copy.
- **Distributed guards decide by values** (`check_distributed_scope`): fixed agents, custom matchmaker, per-agent
  matchmaking/kickstart, `init.from`, critic warm-up, scripted or non-`.pt` teachers are refused with an SP5 pointer
  before any file is read; any non-latest-only mix is reduced with one warning; distributed retention uses the same
  `CheckpointManager` rules and passes `final`.
- Tests are strong: statistical share tests with stated tolerances (drawn shares, spreading, schedules, override,
  asymmetric snapshots, PFSP weightings, FFA independence, mixed teammates in opposing teams, zero-weight anchors),
  coordinator wiring to the global env-step counter, PFSP orientation and team weights, evictions, console/WandB
  filters.

## Issues

### Critical

None.

### Important

**I1. Evicted snapshots come back into the PFSP table through late results — `src/colosseum/league/pfsp.py:175-183`
(`_record`), `:198-203` (`forget`); `src/colosseum/coordinator/coordinator.py:132-140, 178-184`.**
`forget()` drops the cells at eviction, but `_record()` re-creates any cell with `setdefault`. Results that contain
the evicted snapshot keep arriving after the eviction by design (spec block 3: the worker keeps the model while its
current or pending lineup uses it; lineups persist until the next refresh, episodes in progress finish), so almost
every evicted snapshot that was being played is resurrected and then never forgotten again.
- Failure scenario (verified, `final-review/B/pfsp_resurrect.py`): `keep_last: 1, keep_every: 0`; save v10..v50 and
  after each save report one result of `agent_0@latest` vs the just-evicted snapshot. Store: `['ckpt_v50']`;
  `ratings_snapshot()["2p"]["pfsp"]["agent_0"]` lists `ckpt_v10, v20, v30, v40` (all evicted).
- Why it matters: spec block 4/5 «Статистика удалённых снимков сбрасывается» holds only momentarily. Selection is not
  affected (candidates come from the store), but the table grows with every snapshot ever played: memory in the
  coordinator, `ratings.json`, and — since the hub writes the full ratings (with `pfsp`) into `metrics.jsonl` every
  `console_interval_sec` (10 s) — per-record size linear in run length, so `metrics.jsonl` grows quadratically on
  long league runs (≈ 60 B per entry × owners × layouts × snapshots-ever per record). It also shows evicted players
  in the observability table the spec added.
- Fix: in `PfspStats`, remember forgotten keys (`self._forgotten: set[PlayerKey]`; snapshot ids are never reused
  within a run) and skip them in `_record`; or have the coordinator drop pairs whose non-latest, non-`fixed` player is
  not in the store before `update`. Add the test above (late result after eviction stays out of the table).

**I2. LEAGUE_GUIDE understates the SP2 behaviour changes — `docs/LEAGUE_GUIDE.md:34` and `:288-292` (§16).**
Line 34 says the example configs are an exact translation "поведение не изменилось", and §16 says SP2 configs work
"с двумя отличиями" (single-agent `league` + `self_play_ratio: 0`; `.pth` teacher). The code (and spec §8,
«Уточнения» 2–3, which the owner approved as the list of behaviour changes) changes more for unedited SP2 configs:
- a config without matchmaking knobs now gets `{0.7, 0.2, 0, 0.1}` with PFSP hard exponent 2 (SP2: 0.5 / 0.5 with a
  uniform checkpoint);
- translated configs choose snapshots by PFSP (hard, `pfsp_exponent`, EMA half-life 200), not uniformly; SP2's
  arena PFSP used cumulative ratings win rates, now an EMA;
- asymmetric games now meet the other agent's snapshots: `predator_prey` (SP2 copy and v3 example alike) plays the
  prey's snapshots in 20 % of hunter-owned envs, where SP2 always seated prey@latest (verified against
  `LineupMatchmaker` at `c16fe5e`: "owner plays none of the team's roles → PFSP among other agents' latest");
- `mode: self_play` with several agents now also meets other agents' snapshots;
- `mode: league` is drawn per opposing team, not once per match (differs from SP2 in layouts with ≥ 3 teams);
- `pool_size` FIFO becomes `keep_last` + `keep_every: 10` + final (disk grows by total/(interval·10) snapshots).
- Failure scenario: an owner re-running `predator_prey.yaml` or an SP2 league config after the upgrade reads §16,
  expects SP2 dynamics, and gets a different opponent distribution (and more disk use) with no documented reason.
  Spec principle «Поведение может меняться там, где SP3 улучшает дефолты; такие места перечислены в разделе 8»
  makes the list user-facing; the guide is the user doc of the real state.
- Fix: rewrite line 34's last sentence ("the v3 examples behave exactly like their SP2 copies loaded by SP3") and
  extend §16 with the bullet list above (short, one line each).

### Minor

- **M1. Played-share metrics are silently absent for teams with roles of different agents** —
  `src/colosseum/metrics/aggregator.py:64`. `opponent_draws` returns None when the owner's team has the latest
  weights of more than one agent. Besides `teammates: mixed` (documented), this is always the case when the owner's
  team contains a role the owner does not play and another trainable agent does (`_role_player`), e.g. a
  pilot/gunner team trained by two agents: no `opponents` metrics ever, for any owner of that layout. Fix: carry the
  owner explicitly (e.g. `source="owner"` only on the owner's own seat plus `"owner-team"` on its teammates, or an
  owner field on `Lineup`/`MatchResult`), or at least name the case in the docstring and LEAGUE_GUIDE §14.
- **M2. Anchor attribution ties with mixed teammates** — `aggregator.py:76`. With `teammates: mixed` a team drawn from
  `anchors` can hold two anchors 1:1; `most_common` then credits the first seat after permutation (≈50 % wrong).
  Only with `mixed` and several anchors; same fix as M1 (tag the core seat).
- **M3. Shares are relative weights, silently normalized** — `league/mixture.py:75-79`, `core/config.py:368-383`.
  Nothing says shares must sum to 1; a per-agent override of one share on the default global section
  (`agents.x.matchmaking.opponents: {anchors: 0.5}`) yields an effective anchors share of 0.5 / 1.4 ≈ 0.36. `validate`
  prints the effective mix, which mitigates it. Fix: state "shares are normalized over the fillable categories" in
  the `OpponentShares` docstring and LEAGUE_GUIDE §2, or log one INFO line when a schedule point's shares do not sum
  to 1.
- **M4. LEAGUE_GUIDE:171 says per-agent `matchmaking` is a "deep-merge"**; per A18 only `opponents`/`pfsp` merge per
  key, while `anchors`, `layouts` and any schedule replace the global value whole (a per-agent `layouts: {4p: 1}` does
  not add to a global `{2p: 1}`). Same stale wording in code: `core/config.py:749` (TrainableAgent docstring),
  `:1000` (`get_agent_config` docstring), `:1005` (error lists only networks/algorithm/learner), `:894` (agents field
  description lacks matchmaking/init/kickstart) — ledger T3.1 deferred.
- **M5. No fast test validates the live v3 example configs.** `test_sp3_example_configs_v3` only loads them;
  `team_tag.yaml` (changed on purpose) and the new `unit_harvest_league.yaml` are exercised by `validate_config` only
  through slow tests. Both validate today (checked). Fix: one parametrized `validate_config` test over those two (a few
  seconds each).
- **M6. Dead defensive branch** — `league/mixture.py:207-209`: the fallback's zero-weight-anchor list and the
  `anchors` fallback are unreachable for any config `validate_matchmaking` accepts (with no trainable agent for a team,
  role coverage requires an anchor with weight > 0 at every point, so `anchors` share must be > 0 and `mix` is never
  empty). Ledger T3.2/T3.3 deferred. Either delete the zero-weight widening (keep the RuntimeError) or comment it as
  belt-and-braces.
- **M7. `lineup_for` raises `KeyError` for a non-trainable owner or no playable layout** (`mixture.py:164, 173`);
  `KeyError` repr-quotes the message. RuntimeError/ValueError reads better (ledger T3.2).
- **M8. Matchmaker-class construction is duplicated with different wrapping** — `coordinator.py:75-83` re-raises a
  user `ConfigError` as is, `validation.py:_check_matchmaker_class` wraps everything (ledger T3.3). One helper in
  `league.base` (`build_matchmaker(path, context)`).
- **M9. `mixture.CATEGORIES` duplicates `OPPONENT_CATEGORIES[:4]`** (`mixture.py:52`, `core/types.py:30`).
- **M10. `pfsp.exponent` accepts `inf`** (`core/config.py:392`; ledger T3.1): `(1-x)^inf` collapses every weight to the
  1e-6 floor (uniform among losers, surprising). `allow_inf_nan=False` on the field.
- **M11. Distributed mode still drops global `teammates: mixed` silently** (ledger T6.1; SP2 parked item). It is a
  value that cannot be honoured; adding it to the existing "reduced to latest only" warning is one line.
- **M12. `scripts/team_tag_anchors.py:74` tie test without tolerance** (ledger T6.3): with win rates in steps of 1/200,
  a minimum exactly 0.02 below the best (`0.91` vs `0.93`) fails `0.91 >= 0.93 - 0.02` in binary floats. It did not
  affect the T6.3 decision (0.2 had the best minimum). Compare in integer counts or with `+ 1e-9`.
- **M13. `PfspStats` does not validate `prior`** (ledger T2.3) — trivial.

## Deferred minors to fix now (single fix wave)

Recommended, all cheap and in files the wave will touch anyway:
- **T3.1** stale wording (M4: `get_agent_config` docstring and error, `agents` field description, TrainableAgent
  docstring) and `pfsp.exponent` `allow_inf_nan=False` (M10) — schema descriptions are user-visible.
- **T3.2 / T3.3** remove or annotate the dead zero-weight anchor fallback (M6); `KeyError` → `RuntimeError` (M7);
  one matchmaker-construction helper (M8); `CATEGORIES = OPPONENT_CATEGORIES[:4]` (M9).
- **T6.1** mention `teammates: mixed` in the distributed reduction warning (M11).
- With I1's fix, also add the late-result test.
Leave parked: T2.3 `_ratings_row` placement, frozen/scripted-teammate test, prior validation (M13); T3.1 `--set` into
anchor maps (hint only); T3.3 validate exercising a custom matchmaker only at step 0 without snapshots; T6.1 warning
before file logging, spurious warning for empty categories, multi-problem test; T6.3 tie tolerance (M12), W/D/L
helper triplication, slow-test printout.

## Recommendations

- I1 is the only code defect in this area worth a fix-wave item of its own; the rest is wording and tidying.
- Consider writing only a compact PFSP summary (or none) into the per-tick `ratings` record of `metrics.jsonl` and
  keeping the full table in `ratings.json`: even with I1 fixed, `keep_every` makes the stored pool grow with run
  length, and every 10 s record repeats it.
- For the acceptance report: criterion 3.4 now evaluates team_tag against the same uniformly random team the agent
  trains against as an anchor (draw share 0.2, played ≈ 0.36). The result shows the passive mode is gone; it does not
  show generalization beyond the anchor. Worth one sentence so the owner reads the 6-seed result correctly.
- When SP5 replaces the reduction warning with real distributed leagues, `check_distributed_scope` can become the
  hub's capability check; it is already value-based and centralised, which keeps that move cheap.

## Assessment

Ready to merge: **With fixes** — I1 (PFSP table resurrection, small code fix + test) and I2 (LEAGUE_GUIDE
compatibility section) in the final fix wave; the listed deferred minors are optional but cheap. No Critical issue:
matchmaking, PFSP attribution, ratings, schedules, evictions and the distributed guards behave as the spec and the
amendments require.

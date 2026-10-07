# Colosseum: поддержка разных типов игр — ревью

Область: сквозная проверка, как пять типов игр (solo, 1v1, команда юнитов, команда на команду, 1 vs N/FFA/асимметрия) проходят через env API → пространства → воркер → APPO → матчмейкинг → исходы/рейтинги → eval.
Источник истины — код (`src/colosseum`). Все эксперименты — в `scratchpad/env_types/` (`toy_envs.py`, `harness.py`, `exp_solo_ffa_turn.py`, `exp_spaces.py`, `exp_teams.py`, `solo.yaml`, `ffa_league.yaml`; выводы — `out_*.txt`, `train_*.log`). Воркер запускался «по-настоящему» (`rollout_worker_process` в процессе, с `queue.Queue`), плюс `colosseum train` на solo- и 3-player FFA league-конфигах.

**Кратко:**
- **Хорошо поддержаны** только 1v1 с одновременными ходами и фиксированным составом (chase, space_miners).
- **Solo** обучается, но на выходе нечего смотреть: eval ничего не выводит, счёт эпизодов нигде не логируется.
- **Turn-based и выбывание.** Задокументированная конвенция `info["active"]` теряет награды и `done` (ET-01, critical), а per-player `terminated` сбрасывает весь матч (ET-02).
- **Команды и FFA.** Eval и PFSP смешивают агентов внутри одной команды. Win-rate для N>2 считается неверно. Если один агент занимает несколько мест, его результаты затираются.
- **Командные игры с юнитами** работают только через хаки: паддинг, маска и no-op. При K>10 MultiDiscrete отдаёт действия не тем юнитам. Joint ratio при многих юнитах вырождает PPO-clip.
- **Асимметричные роли** без изменения кода не поддержаны (❌).

---

## 1. Матрица покрытия

Легенда: ✅ работает; ⚠️ работает с оговорками или хаками; ❌ невозможно без изменения кода.

| Тип игры \ стадия | Env API | Obs / Action spaces | Worker (chunks, reward, done) | Алгоритм (APPO/V-trace) | Матчмейкинг | Исходы и рейтинг | Eval |
|---|---|---|---|---|---|---|---|
| **Solo (1 игрок)** | ✅ `num_players=1` допустим (`config.py:98`, `ge=1`). VERIFIED: `colosseum train --config solo.yaml` дошёл до `ckpt_v50` | ✅ Box/Discrete | ✅ VERIFIED: 40 chunks, награды верные | ✅ VERIFIED: метрики конечные | ⚠️ работает, но чекпоинты и «arena» не имеют смысла (`matchmaker.py:214`: arena-матч = `['a']`) | ❌ outcome всегда 0.5 (`outcomes.py:28-29`), ELO не меняется, счёт/return нигде не логируется (ET-11, ET-12) | ❌ `evaluate_agents` для одного агента возвращает `{}`, пустую таблицу (ET-11) |
| **1v1 simultaneous** | ✅ | ✅ включая Dict-действия (space_miners, chase) | ✅ | ✅ | ⚠️ latest всегда сидит в seat 0 против чекпоинтов, в arena тоже (ET-15) | ✅ WDL через `outcome`/`rank`/reward | ⚠️ места рандомизируются, но без балансировки; recurrent-сети не работают (ET-10) |
| **1v1 turn-based** | ⚠️ «неактивный игрок шлёт любое действие» (`base_env.py:19-20`) | ⚠️ маска неактивного игрока не может быть пустой → NaN-краш (ET-08) | ❌ с `info["active"]` теряются награда и `done` за ход соперника (ET-01, VERIFIED); без `active` 50% переходов — шум (ET-18) | ⚠️ с LSTM hidden продвигается на чужих ходах (ET-17) | ⚠️ первый ход почти всегда за latest/primary (ET-15) | ✅ | ⚠️ как 1v1 |
| **Команда юнитов (Lux/Halite)** | ⚠️ переменное число юнитов только через паддинг до K_max | ⚠️ `MultiDiscrete([A]*K)` + плоская маска + no-op для мёртвых юнитов. ❌ при K>10 действия уходят не тем юнитам (ET-07, VERIFIED). ❌ 2D-nvec/GridNet (ET-14). ❌ Dict-obs (ET-13). Policy видит только `latent` (ET-14). Пустая строка маски → NaN (ET-08) | ✅ фиксированная форма `[T, K]` | ⚠️ joint log-prob = сумма по K: clip_fraction 0.00→0.91 при K=1→128 (ET-09, VERIFIED); нет per-unit reward/credit | ✅ как 1v1 | ✅ как 1v1 | ⚠️ как 1v1 |
| **Team vs team (2v2)** | ⚠️ команда — только конвенция env (outcome на каждое место); понятия team в API нет | ✅ | ✅ общий reward задаёт env | ✅ (каждое место — отдельный policy instance с теми же весами) | ❌ arena `[a,b,a,b]`: при командах {0,1}/{2,3} обе команды смешанные (ET-05). Self-play сэмплирует чекпоинт независимо на каждое место, поэтому сокомандником может оказаться старый чекпоинт (ET-16) | ❌ коллизия ключей `agent:net` (ET-04); win-rate по абсолютному outcome (ET-03) | ❌ VERIFIED: строго лучший A получает WR 0.35, 392/600 ничьих (ET-05) |
| **1 vs N: FFA с рангами** | ⚠️ выбывание выражается только через `active` (ET-01) или ломает эпизод (ET-02) | ✅ | ❌ выбывший игрок теряет финальную награду и `done` (VERIFIED: seat0 −99 вместо −200, dones=0) | ⚠️ эпизоды «склеиваются» без done → bootstrapping через границу эпизода | ⚠️ arena заполняется только 2 агентами `[a,b,a]`, N−1 соперников не сэмплируются (ET-16); данные смещены в пользу agents[0] (ET-15, VERIFIED) | ⚠️ rank→[0,1] и попарный ELO разумны (VERIFIED x>y>z → 1611/1201/788), но win-rate неверен (ET-03) и результаты затираются (ET-04) | ⚠️ попарные WR есть, но нет среднего ранга и top-1 |
| **1 vs N асимметричный (hunter vs prey)** | ❌ одно `observation_space`/`action_space` на всех (`base_env.py:29-39`) | ❌ один буфер и один `ActionSpec` на воркер (`rollout_worker.py:392-414`) | ❌ (хак: union-пространства + маски + one-hot роли в obs, одна общая сеть) | ⚠️ только хак с общей сетью | ❌ нет ролей; матчмейкер не подключается из конфига (`coordinator.py:64-92`) | ❌ нет per-role рейтинга | ❌ |
| *Сквозные аспекты:* выбывание / разная длина эпизода / ход раз в N шагов / partial obs | ❌ / ⚠️ / ⚠️ / ⚠️ | — | ET-01, ET-02, ET-17, ET-20; truncation = terminal (ET-22) | LSTM-обучение есть (`evaluate_actions_recurrent` с ресетами) | — | — | ❌ LSTM в eval (ET-10) |

---

## 2. Находки

### ET-01 — critical — награды и `done` неактивных мест теряются (turn-based, выбывание)
- **Где:** `worker/rollout_worker.py:496-518`, флаги из `:699-723`.
- **Проблема:** при `info["active"]=False` (pre-step) переход не записывается, и вместе с ним пропадают `reward` и `done` этого шага. В turn-based игре, которая заканчивается на ходу соперника, проигравший или победитель не получает финальную награду. Ещё хуже, что его эпизоды склеиваются без `done`: V-trace бутстрапит через границу эпизода. То же происходит с выбывшим игроком в FFA: финальная награда за место и `done` не доходят.
- **Доказательство (VERIFIED-BY-RUNNING, `out_solo_ffa_turn.txt`):**
  - `AlternatingEnv` (env даёт seat0 +1 за эпизод, 100 эпизодов): `seat 0: recorded steps=200, sum reward=0.0, dones=0`, `seat 1: sum reward=-100.0, dones=100`. Chunk seat0: `rewards [0,0,0,0] dones [0,0,0,0] t:[0,0.5,0,0.5]`, то есть эпизоды идут подряд без done.
  - `FFAElimEnv(active)`: `seat 0: sum reward=-99.0 (ожидалось -200), dones=0`, `seat 1: dones=0`.
  - Конвенция `active` нигде не протестирована (`grep active tests/` пусто) и не описана в README. Пример tic_tac_toe её не использует, поэтому баг не проявлялся.
- **Исправление:**
  - Для каждого (env, seat) держать «открытый» последний переход и прибавлять к нему награды неактивных шагов: `buf._rewards[cursor-1] += r`.
  - При конце эпизода ставить `done=1` на последний собственный переход места.
  - Запечатывать полный буфер лениво: в момент следующего собственного действия места (тогда bootstrap = V этого состояния) или в конце эпизода.
  - Целевой вариант — `SeatTrajectory` (D3).

### ET-02 — high — per-player `terminated` сбрасывает весь матч; выбывание не поддержано API
- **Где:** `envs/vec_env.py:217-234` (`any(term_k[p])` → reset), контракт `envs/base_env.py:74-75`; `obs_dict[p]` нужен для всех мест (`vec_env.py:138-140`).
- **Доказательство (VERIFIED):** `FFAElimEnv(per_player_term)` выдаёт `episode lengths reported: [2] (true game length is 6)`. Матч обрывается на первом выбывании, оставшиеся игроки финала не видят.
- **Исправление:** per-seat termination. Эпизод заканчивается, когда все места terminated, либо по явному `episode_over`. Выбывшее место исключается из acting-set; obs и маска для него не требуются (D1/D3).

### ET-03 — high — WinRateTracker в N>2 и командных играх считает победы по абсолютному outcome
- **Где:** `coordinator/coordinator.py:167` → `ratings.py:317-341` (`_wins[a][b] += int(outcome_a*2)`).
- **Проблема:** в паре (a, b) учитывается `outcome_a ∈ [0,1]` относительно всего матча, а не результат a против b. ELO (`coordinator.py:169-174`) сравнивает outcome корректно, а win-rate — нет. PFSP берёт приоритеты именно из win-rate.
- **Доказательство (VERIFIED):**
  - FFA x>y>z: `WR(y,z)=0.5`, хотя y всегда выше z.
  - w выигрывает, y и z делят последнее место: `WR(y,z)=0.0, WR(z,y)=1.0` — результат зависит от порядка ключей в dict.
  - Смешанная 2v2 (оба агента outcome 0.0): ELO записал ничью, win-rate — поражение a.
- **Исправление:** передавать в `record` `1.0 if outcome_a>outcome_b else 0.0 if < else 0.5`; пары внутри одной команды не сравнивать (D6/D7).

### ET-04 — high — результаты затираются, когда один агент занимает несколько мест
- **Где:** `worker/rollout_worker.py:648-653`: `player_outcomes[f"{agent}:{net}"] = …` перезаписывается последним местом. Тот же ключ в `total_rewards`.
- **Доказательство (VERIFIED):** FFA с местами `a,b,a`, ранги 1/2/3. Получено `{'a:latest': 0.0, 'b:latest': 0.5}`: победа a в seat0 потеряна, и ELO стал `{'b':1216,'a':1184}`. В self-play все места `a:latest` схлопываются в один ключ. Агрегация «среднее по местам» в `coordinator.py:150-156` никогда не получает несколько значений.
- **Исправление:** `MatchResultV2.seats: list[SeatRecord]` (seat, team, role, agent, net, reward, rank) (D6). Быстрый фикс — ключ `agent:net#seat` плюс агрегация в координаторе.

### ET-05 — high — eval и PFSP-arena смешивают агентов внутри команд
- **Где:** `eval.py:203-209` (`base=[k % num_agents]` + shuffle) и `:274-280` (усреднение по местам); `coordinator/matchmaker.py:214-225` (`agents[i % 2]`).
- **Доказательство (VERIFIED, `out_teams.txt`):** `TeamEnv` 2v2, A строго лучше B. Eval выдал `A vs B | 600 | 208 | 0 | 392 | 0.347`, при истинных 100% побед. Две трети матчей прошли со смешанными командами {A,B} vs {A,B} и засчитаны как ничьи. PFSP arena: `['a','b','a','b']` — при командах {0,1}/{2,3} обе команды смешанные, и сигнала a-против-b нет вовсе.
- **Исправление:** назначать агента на команду целиком (lineup), а внутри команды и между командами переставлять места (D8, D10).

### ET-06 — high — асимметричные роли невозможны
- **Где:** `envs/base_env.py:29-39` (одно пространство на всех); `rollout_worker.py:392-414` (один `obs_shape`/`ActionSpec`/буфер на воркер); `network_factories[aid]` — одна сеть на агента; `coordinator.py:64-92` — набор матчмейкеров зашит, понятия «роль» нет.
- **Доказательство:** BY-READING.
- **Обходной путь:** union-пространства, маски, one-hot роли в obs и одна общая сеть. Отдельные policy на hunter и prey посадить на «свои» места нельзя: arena чередует агентов по местам без учёта ролей.
- **Исправление:** `GameSpec/RoleSpec/SeatSpec` (D1), сети и буферы per (agent, role) (D9), role-aware матчмейкер (D8).

### ET-07 — high — MultiDiscrete с K>10 отдаёт действия не тем юнитам
- **Где:** `core/action_spec.py:110-112` (`named={str(i):…}`) + `:128-129` (`sorted(named)`) — лексикографический порядок `'0','1','10','11','2',…`; `networks/distributions.py:133` сортирует ключи так же.
- **Доказательство (VERIFIED):** политика с головами `str(i)`, где голова i выбирает действие i, при K=12 приводит к `env receives [0,1,10,11,2,3,…,9]`: юниты 2–11 получают действия чужих голов. Маска согласована с позицией (голова '10' получает строку маски env-юнита 2), поэтому падения нет и ошибка тихая. Политика, считающая logits юнита i из его эмбеддинга, будет управлять другим юнитом.
- **Исправление:** для `multi_discrete`/`tuple` не сортировать ключи (числовой порядок); в `CompositeDist` принимать `list`/`OrderedDict` без сортировки. Лучше — отдельный `FactorizedCategorical` (D4).

### ET-08 — medium — строка маски без валидных действий даёт NaN и краш
- **Где:** `networks/distributions.py:53-56` (`masked_fill(-inf)` → `Categorical` с NaN-нормализацией).
- **Доказательство (VERIFIED):** `ValueError: Expected parameter logits … found invalid values: tensor([[nan, nan, nan, nan], …])` в двух сценариях: (а) у неактивного игрока turn-based все действия запрещены; (б) у мёртвого или несуществующего юнита пустая строка маски. Оба способа выражения — самые естественные.
- **Исправление:** в `CategoricalDist` для строк без валидных действий подставлять фиктивный logit (например, разрешать действие 0) и обнулять `log_prob`/`entropy` этой строки; отдавать `unit_mask` наружу (D4).

### ET-09 — high (для игр с юнитами) — joint log-prob по многим юнитам вырождает PPO-clip и V-trace
- **Где:** `distributions.py:179-198` (сумма log-prob/энтропии по компонентам) → `algorithms/appo.py:208-218`, `vtrace.py:56-59`.
- **Доказательство (VERIFIED, синтетика):** одинаковое малое изменение per-unit logits (σ=0.15).

  | K | std(log ρ) | clip_fraction (ε=0.2) | mean c_t |
  |---|---|---|---|
  | 1 | 0.037 | 0.00 | 0.98 |
  | 8 | 0.29 | 0.51 | 0.88 |
  | 32 | 0.69 | 0.79 | 0.72 |
  | 128 | 1.45 | 0.91 | 0.47 |

  То есть при сотнях юнитов почти все сэмплы клипаются, а V-trace обрезает трассы почти до 1-step TD. Энтропия суммируется по юнитам, поэтому её вклад растёт с K.
- **Исправление:** `ratio_mode="per_unit"` — clip на ratio каждого юнита, общий advantage, усреднение по живым юнитам; для V-trace ρ = exp(mean log ρ_u); энтропия — среднее по живым юнитам (D5). Per-unit reward/value — опционально позже.

### ET-10 — high (partial observability) — eval игнорирует рекуррентное состояние
- **Где:** `eval.py:244-247` вызывает `net.act(...)` без `hidden`; `actor_critic.py:319`: при `hidden is None` RNN пропускается.
- **Доказательство (VERIFIED):**
  - При `latent_dim≠hidden_size`: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x32 and 16x4)`.
  - При равных размерах ошибки нет, но поведение другое: `max |logit diff| = 0.229` относительно правильного пути.
  - Тот же вызов без hidden есть в `validate_config` (`registry.py:129` передаёт hidden — там корректно).
- **Исправление:** хранить hidden для каждого (env, seat) в eval, сбрасывать на done, передавать в `act`. Если сеть рекуррентная, а `hidden=None`, использовать `initial_hidden` (а не пропускать RNN).

### ET-11 — medium — solo: нет осмысленного outcome и eval
- **Где:** `core/outcomes.py:28-29` (один игрок → `mx==mn` → 0.5); `eval.py:195-201, 282-295` (только пары).
- **Доказательство (VERIFIED):** `player_outcomes([7.0]) → [0.5]`; eval одного агента возвращает `results: {}` и пустую таблицу. `colosseum eval` для solo-игры бесполезен.
- **Исправление:** `OutcomeKind.SCORE`; eval-режим «score»: среднее, std и CI очков, длина эпизода; сравнение двух чекпоинтов — разница средних с CI (D10).

### ET-12 — medium (для solo — high) — эпизодные метрики не логируются
- **Где:** `launcher.py:558-564` — `MatchResult` (с `total_rewards`, `episode_length`) уходит только в координатор; `metrics/wandb_logger.py` логирует только loss-метрики learner'а; `get_ratings_summary` нигде не вызывается.
- **Доказательство:** BY-READING; в `train_solo.log` нет ни одной строки с return или score.
- **Исправление:** агрегировать результаты в мониторе и логировать `episode_return`, `episode_length` и score по агентам и местам, а также ELO и средний ранг (D11).

### ET-13 — medium — Dict-наблюдения (и entity-list как Dict) невозможны
- **Где:** `vec_env.py:126-128, 134-140` (`np.array` по dict → object-массив); `rollout_worker.py:72, 134` (`float32`-буфер, `torch.from_numpy`); `registry.py:128`.
- **Доказательство (VERIFIED):** `VectorEnv obs dtype/shape: object (2, 2)`, затем `TypeError: can't convert np.ndarray of type numpy.object_` в `rollout_worker.py:134` и в `validate_config` (`registry.py:128`).
- **Обходной путь:** упаковать всё в один плоский Box, а маску сущностей передавать отдельным признаком.
- **Исправление:** `ObsSpec` + `TensorTree` во всём пайплайне (D2).

### ET-14 — medium — нет GridNet и per-unit голов с доступом к per-unit фичам
- **Где:**
  - `action_spec.py:110-112`: `MultiDiscrete` с 2D-nvec.
  - `networks/base.py`, `actor_critic.py:316-333`: policy получает только `latent [B, D]`.
- **Доказательство:** VERIFIED: `TypeError: only 0-dimensional arrays can be converted to Python scalars` для `MultiDiscrete(np.full((2,4),4))`. BY-READING: чтобы головы видели эмбеддинги юнитов или клеток, энкодер вынужден отдавать их «расплющенными» в latent. С LSTM-trunk это не работает: RNN получит вход в тысячи признаков, а per-unit структура потеряется.
- **Дополнительно (suspected, не замерено):** `CompositeDist` создаёт K Python-объектов `CategoricalDist` и циклы по K на каждый forward, что медленно при K в сотни или тысячи клеток.
- **Исправление:** `UnitActions`/nd-`ActionSpec`, `FactorizedCategorical` на тензоре `[B, U, A]`, `EncoderOutput(latent, aux)` (D4).

### ET-15 — high — асимметрия мест и данных в матчмейкинге
- **Где:**
  - `matchmaker.py:105-111`, `196-199`: latest всегда в seat 0, чекпоинты только в seat≥1.
  - `matchmaker.py:216-221`: в arena seat 0 всегда у focus-агента.
  - `launcher.py:420-422, 607-610`: матчи всегда генерируются для `agent_ids[0]`.
  - `distributed.py:334-337`: фиксированный round-robin.
- **Доказательство (VERIFIED):**
  - В 2-player arena seat 0: `{'a': 200}` из 200.
  - FFA league (`ffa_league.yaml`): к моменту `a: ckpt_v45` у b было только `ckpt_v10`. Primary-агент получает примерно 4× данных, второй агент никогда не играет solo/self-play.
  - В turn-based игре latest против истории всегда ходит первым; в асимметричных картах всегда играет одной стороной.
- **Исправление:** случайная или сбалансированная перестановка мест; выбор focus-агента по всем trainable (D8).

### ET-16 — medium — нет набора из N−1 соперников; сокомандники из истории
- **Где:**
  - `matchmaker.py:214-225`: arena = ровно 2 агента, чередование.
  - `matchmaker.py:104-126`: в self-play чекпоинт сэмплируется независимо для каждого места.
- **Доказательство:** VERIFIED: `3-player FFA arena seats: ['a','b','a']`. BY-READING: в 2v2 self-play в команде latest может оказаться старый чекпоинт как сокомандник (и не собирать данные), а в соперниках — latest.
- **Исправление:** `LeagueMatchmaker` — на каждую команду-соперника выбирается одна сущность; в FFA — N−1 независимых PFSP-выборов (D8).

### ET-17 — medium — рекуррентное состояние продвигается на неактивных шагах
- **Где:** `rollout_worker.py:473-479` (инференс и обновление hidden для всех мест) против `:496-497` (запись только активных).
- **Доказательство:** BY-READING. Learner разворачивает LSTM только по записанным шагам с `h_init` начала chunk'а. На rollout hidden «видел» чужие ходы, на обучении — нет: policy lag растёт и не исправляется V-trace.
- **Исправление:** для неактивных мест не вызывать `act` и не трогать hidden (заодно экономится инференс) (D3).

### ET-18 — low — без `active` turn-based обучается на игнорируемых ходах
- **Доказательство (VERIFIED):** `fraction of recorded transitions where the slot's action is ignored by env: 0.50`.
- **Проблема:** половина переходов — шум для policy gradient и энтропии; инференс тратится зря.
- **Исправление:** ET-01 плюс acting-set из D1.

### ET-19 — low — конфиг `env.num_players` не сверяется с env
- **Где:** `registry.py:106-162` (нет проверки), `coordinator.py:100`.
- **Проблема (BY-READING):** если в конфиге меньше мест, чем у env, будет `IndexError` в воркере; если больше — молча лишние места в MatchConfig. Для solo-env со значением по умолчанию `num_players: 2` arena будет `[a,b]` для одного места.
- **Исправление:** добавить проверку в `validate_config` или брать значение из env.

### ET-20 — low — маска и obs обязательны для каждого места
- **Где:** `rollout_worker.py:691`: `infos[e][p]["action_mask"]` даёт `KeyError`, если у выбывшего места маски нет, а у seat0 есть; `vec_env.py:138-140` требует obs для всех мест.
- **Доказательство:** BY-READING.
- **Исправление:** obs и маска нужны только для acting-мест (D1).

### ET-21 — low — инвертированный CI в отражённой строке eval
- **Где:** `eval.py:134-141`.
- **Доказательство (VERIFIED):** `B vs A | … | 0.000 [0.614, 0.690]` — интервал не содержит точечную оценку, потому что ничьи учитываются асимметрично.
- **Исправление:** считать Wilson CI отдельно для `wins_b`, либо выводить score = (W + D/2)/N с CI.

### ET-22 — low — truncation считается terminal (bootstrap 0)
- **Где:** `rollout_worker.py:490, 531`; `dones = term|trunc` в V-trace.
- **Проблема (BY-READING):** для игр с лимитом шагов, где конец по лимиту — настоящий конец игры, это допустимо. Для solo-задач с обрезкой по времени это смещение. Пересекается с ревью chunk-логики, поэтому здесь только упоминание.

### ET-23 — low — ELO в FFA масштабируется с N
- **Где:** `coordinator.py:159-174`.
- **Проблема:** каждая пара получает полный K=32, поэтому рейтинг в 8-player FFA в 7 раз волатильнее, чем в 1v1. Ранжирование при этом правильное (VERIFIED: 1611/1201/788).
- **Исправление:** K/(N−1) или Plackett-Luce/OpenSkill (D7).

**Что работает (VERIFIED):**
- solo-обучение end-to-end, включая CLI и чекпоинты;
- 1v1 simultaneous;
- Dict-действия с фиксированным числом юнитов (space_miners: 3 корабля);
- плоские маски;
- Box-действия формы (K, D);
- rank→outcome с ничьими (`[1, .67, .67, 0]` для рангов 1, 2, 2, 4);
- попарный ELO для FFA;
- LSTM-обучение с ресетами внутри chunk'а.

---
## 3. Предложение по редизайну

Цель — минимальный набор изменений, после которого все 5 типов игр становятся first-class, при сохранении обратной совместимости (старый `BaseEnv` работает через адаптер).

### 3.1 Env API v2: роли, команды, места, acting-set (D1, **M**, prerequisite для D3–D9)

```python
# colosseum/envs/spec.py
@dataclass(frozen=True)
class RoleSpec:
    name: str                              # "player", "hunter", "prey", ...
    observation_space: gym.Space           # Box | Dict (см. D2)
    action_space: gym.Space                # Discrete | Box | Dict | MultiDiscrete(nd) | UnitActions (см. D4)

@dataclass(frozen=True)
class SeatSpec:
    seat: int                              # 0..num_seats-1
    role: str
    team: int                              # команда; FFA: team == seat; solo: 0

class OutcomeKind(str, Enum):
    SCORE = "score"        # solo / score-attack: только число
    WDL = "wdl"            # win/draw/loss между командами
    RANK = "rank"          # FFA / ranked teams: ранги 1..N, ничьи разрешены

@dataclass(frozen=True)
class GameSpec:
    roles: dict[str, RoleSpec]
    seats: tuple[SeatSpec, ...]
    outcome_kind: OutcomeKind
    symmetric: bool = True                 # все места взаимозаменяемы → можно ротировать места
    max_episode_steps: int | None = None

    @property
    def num_seats(self) -> int: ...
    def teams(self) -> dict[int, list[int]]: ...
    def seats_of_role(self, role: str) -> list[int]: ...

@dataclass
class Outcome:
    kind: OutcomeKind
    seat_score: dict[int, float] | None = None   # «сырой» счёт игры (solo, Halite)
    team_rank: dict[int, int] | None = None      # 1 = лучший; WDL = ранги {1,2} или {1,1}

@dataclass
class StepResult:
    obs: dict[int, Obs]                 # наблюдения для мест, которые действуют на СЛЕДУЮЩЕМ шаге
    rewards: dict[int, float]           # награды ВСЕМ местам (в т.ч. неактивным/выбывшим на этом шаге)
    terminated: dict[int, bool]         # per-seat: место закончило (выбыло или игра окончена)
    truncated: dict[int, bool]
    acting: frozenset[int]              # кто действует на следующем шаге (turn-based, «ход раз в N тиков»)
    action_masks: dict[int, Mask]       # только для acting-мест
    infos: dict[int, dict]
    outcome: Outcome | None = None      # заполнено ⇔ эпизод окончен (все места terminated/truncated)

class MultiAgentEnv(ABC):
    spec: GameSpec
    def reset(self, seed: int | None = None) -> StepResult: ...
    def step(self, actions: dict[int, Action]) -> StepResult:    # actions только для acting-мест
        ...

class LegacyEnvAdapter(MultiAgentEnv):
    """Оборачивает текущий BaseEnv: одна роль, team=seat, acting=info['active'] или все,
    outcome из info['outcome'|'rank'] или из суммарной награды."""
```

Семантика (из PettingZoo AEC/Parallel и OpenSpiel): эпизод окончен, когда все места terminated/truncated; выбывшее место перестаёт появляться в `acting`; награды места накапливаются, пока оно не действует.

### 3.2 Seat-трекер в воркере: корректные reward/done для неактивных и выбывших мест (D3, **M**, зависит от D1; quick-fix-версия поверх текущего API — **S**)

```python
class SeatTrajectory:
    """Буфер одного (env, seat). Переход (s_t, a_t) «открыт», пока место снова не начнёт действовать
    или не завершится. Все награды, пришедшие в промежутке, суммируются в r_t."""
    def on_act(self, obs, action, logp, value, mask, hidden) -> Optional[TrajectoryChunk]:
        # закрывает предыдущий открытый переход; если буфер полон — запечатывает chunk
        # с bootstrap = value (V текущего состояния того же места) и возвращает его
    def on_reward(self, r: float) -> None          # += к открытому переходу
    def on_terminal(self, truncated: bool, bootstrap: float | None) -> Optional[TrajectoryChunk]:
        # done=1 на открытом переходе, сброс hidden; chunk не обязан быть полным (padding+valid-mask)
```

Ключевые свойства: (1) ни одна награда не теряется; (2) `done` ставится на последнем собственном переходе места, даже если игра закончилась на чужом ходу; (3) инференс и обновление LSTM-состояния — только для acting-мест (тот же порядок шагов, что видит learner); (4) выбывание = `terminated[seat]`, без сброса всего env.

Для «быстрого фикса» в текущем `rollout_worker.py` (без API v2): `pending_reward[e,p]` и запечатывание буфера только при следующем активном шаге/конце эпизода; при `done` и неактивном слоте — `buf._rewards[cursor-1] += r; buf._dones[cursor-1] = 1`.

### 3.3 Структурированные наблюдения (D2, **M**, независимо)

```python
TensorTree = torch.Tensor | dict[str, "TensorTree"]
class ObsSpec:                                   # из gym.Space: Box, Dict, MultiBinary (вложенные Dict)
    @classmethod
    def from_space(cls, space) -> "ObsSpec": ...
    def allocate(self, leading: tuple[int, ...]) -> dict[str, np.ndarray]: ...
    def stack(self, obs_list) -> dict[str, np.ndarray]: ...
def tree_map(fn, tree: TensorTree) -> TensorTree: ...
```
`TrajectoryChunk.observations: TensorTree`; `BaseEncoder.forward(obs: TensorTree)`; `VectorEnv`, `RolloutBuffer`, `_prepare_batch`, сериализация gRPC, нормализация — через `tree_map`. Entity-list = `Dict(entities=Box(N_max, F), entity_mask=MultiBinary(N_max))` — паддинг + маска становятся конвенцией, а не хаком.

### 3.4 Per-unit действия (D4, **M**, независимо; D5 зависит от D4)

```python
class UnitActions(gym.spaces.Space):
    """max_units юнитов × под-действие (Discrete(A) | MultiDiscrete([A1..Ac])). Также GridNet: max_units = H*W."""
    def __init__(self, max_units: int, per_unit: gym.spaces.Discrete | gym.spaces.MultiDiscrete): ...

class FactorizedCategorical(Distribution):
    """logits [B, U, A] (или список по компонентам), mask [B, U, A], unit_mask [B, U].
    Юнит без валидных действий/несуществующий → logp=0, entropy=0, без NaN."""
    def log_prob(self, a: Tensor) -> Tensor              # [B]   сумма по живым юнитам (joint)
    def log_prob_per_unit(self, a: Tensor) -> Tensor     # [B, U]
    def entropy(self, reduce: Literal["sum", "mean"] = "mean") -> Tensor
    def unit_mask(self) -> Tensor                        # [B, U]
```
- `ActionSpec`: нативный тип `"units"` / nd-`MultiDiscrete` с `action_shape=(U,)|(U,C)|(H,W,C)`, dtype int64, маска формы `(U, A)`; никаких Python-циклов по компонентам; исправить порядок компонент (числовой, а не строковый).
- Политика должна видеть per-unit фичи, а не только `latent`:
```python
@dataclass
class EncoderOutput:
    latent: Tensor                    # [B, D] → RNN → value / глобальные головы
    aux: dict[str, Tensor]            # напр. unit_emb [B, U, E], spatial [B, C, H, W]
class BasePolicy(nn.Module):
    def forward(self, latent: Tensor, aux: dict[str, Tensor] | None = None) -> Distribution: ...
```
`ActorCriticNetwork.forward`: если encoder вернул `EncoderOutput`, RNN применяется к `latent`, `aux` передаётся в policy как есть (совместимо со старыми энкодерами, возвращающими Tensor).

### 3.5 APPO для многих юнитов (D5, **S**, зависит от D4)

`AlgorithmConfig.ratio_mode: Literal["joint", "per_unit"] = "joint"`:
- `joint` — как сейчас (корректно для малого числа компонент).
- `per_unit` — PPO-clip применяется к ratio каждого юнита с общим (командным) advantage, loss усредняется по живым юнитам; для V-trace ρ/c берётся `exp(mean_u log_ratio_u)` (или `clamp(joint)` — опция). Энтропия — среднее по живым юнитам.
- Опционально (позже, **M**): per-unit reward/value (`rewards: [T, U]`) для credit assignment.

### 3.6 Исходы, MatchResult v2 и рейтинги (D6 **S**, D7 **S-M**; зависят от D1)

```python
@dataclass
class SeatRecord:
    seat: int; role: str; team: int
    agent_id: str; network_id: str
    total_reward: float; score: float | None; rank: int | None   # team rank
@dataclass
class MatchResultV2:
    match_id: str; game_kind: OutcomeKind; seats: list[SeatRecord]; episode_length: int
```
(решает коллизию ключей `agent:network`).

```python
class BaseRating(ABC):
    def update(self, teams: list[list[str]], ranks: list[int]) -> None: ...   # команды × ранги (ничьи = равные ранги)
    def get(self, entity: str) -> tuple[float, float]: ...                    # (mu, sigma)
class TeamElo(BaseRating): ...          # 1v1 и командные; рейтинг команды = среднее участников
class PlackettLuceRating(BaseRating):   # FFA/N-команд (OpenSkill / TrueSkill-подобный), без внешних зависимостей
    ...
class ScoreTracker:                     # solo: EMA/среднее/CI очков на agent/checkpoint
    def update(self, entity: str, score: float) -> None: ...
```
Правило: пары сравниваются только между РАЗНЫМИ командами; одно и то же базовое значение-сущность (agent:network) внутри команды не сравнивается само с собой. Entity для рейтинга — `agent_id:network_id` (чекпоинты тоже получают рейтинг → нормальная PFSP по истории).

### 3.7 Матчмейкинг с ролями, командами и ротацией мест (D8, **M-L**; зависит от D1, D6)

```python
@dataclass
class Lineup:                   # кто играет за одну команду
    team: int
    members: dict[int, PlayerSlot]          # seat -> slot

class OpponentSampler(ABC):
    def sample(self, focus: str, role: str, k: int, game: GameSpec) -> list[str]:  # entity ids
        ...                                   # PFSP / uniform / latest / scripted

class BaseMatchmaker(ABC):
    def generate(self, game: GameSpec, focus_agents: list[str], n: int) -> list[MatchConfig]: ...

class LeagueMatchmaker(BaseMatchmaker):
    """1) выбирает focus-агента (round-robin/пропорционально по всем trainable, а не только agents[0]);
       2) выбирает команду/роль для focus с учётом allowed_roles агента;
       3) всю команду focus заполняет им самим (latest, collect=True) или (опция) teammates из пула;
       4) каждой из остальных команд выбирает ОДНУ сущность (PFSP по рейтингу/WR) и заполняет ею все места команды
          (FFA: N-1 независимых выборов);
       5) применяет перестановку мест: random или balanced (пары матчей с переставленными сторонами)."""

class SoloMatchmaker(BaseMatchmaker): ...   # num_seats == 1: нет соперников, чекпоинты не нужны
```
Конфиг: `matchmaking.class`, `matchmaking.seat_permutation: random|balanced|fixed`, `agents.<id>.roles: [hunter]`, `matchmaking.teammates: self|pool`. Coordinator получает матчмейкер по dotted path (сейчас жёстко зашит).

### 3.8 Асимметричные роли (D9, **L**; зависит от D1, D2, D8)

- Буферы и сети строятся per (agent, role): `network_factories[(agent_id, role)]`, `ActionSpec/ObsSpec` per role.
- Два варианта: (а) отдельный trainable-агент на роль (свой learner) — просто и изолированно; (б) один агент с несколькими головами/энкодерами по роли (`MultiRoleActorCritic(role -> ActorCritic)`), один learner, chunk помечается `role`.
- Рейтинг: отдельный по ролям (hunter-Elo против prey-Elo) + матрица WR role×role.

### 3.9 Eval v2 (D10, **M**; зависит от D1, D6, частично D7)

| Тип игры | Метрики |
|---|---|
| Solo | mean/std/95% CI score, длина эпизода, success-rate, распределение очков |
| 1v1 | WR + Wilson CI, раздельно по месту (первый/второй ход), ничьи; balanced seat schedule |
| FFA | средний ранг ± CI, top-1 rate, pairwise «beats»-матрица, PL/TrueSkill μ±σ |
| Teams | команда = фиксированный lineup (agent на всю команду), WR команд, перестановки сторон |
| Asym | матрица WR role×role (hunter_ckpt × prey_ckpt), по каждой роли отдельно |

Плюс: поддержка recurrent-сетей (hidden per (env, seat)), scripted-агентов, per-agent архитектур из CLI (`--agent name:path.pt:config.yaml`).

### 3.10 Метрики обучения (D11, **S**, независимо)

Монитор уже получает `MatchResult` — логировать в WandB: `episode_return`, `episode_length`, `score` (solo), WR/рейтинг по агентам и по месту, средний ранг (FFA), долю выбываний.

### 3.11 Сводка: порядок работ и оценки

| ID | Работа | Effort | Зависит от |
|---|---|---|---|
| Q1 | Quick-fixes: reward/done для неактивных мест (ET-01), коллизия ключей результата (ET-04), win-rate N>2 (ET-03), порядок компонент MultiDiscrete (ET-07), all-False маска (ET-08), recurrent в eval (ET-10), `env.num_players` vs config (ET-19), логирование эпизодов (ET-12), solo-eval (ET-11) | S (каждый), ~M суммарно | — |
| D1 | GameSpec/RoleSpec/SeatSpec/StepResult + LegacyEnvAdapter | M | — |
| D2 | Dict/entity наблюдения (ObsSpec, TensorTree) | M | — |
| D3 | SeatTrajectory в воркере (per-seat termination, acting-set) | M | D1 |
| D4 | UnitActions + FactorizedCategorical + nd ActionSpec + EncoderOutput.aux | M | (D2 желательно) |
| D5 | APPO ratio_mode=per_unit | S | D4 |
| D6 | Outcome + MatchResultV2 | S | D1 |
| D7 | Рейтинги: TeamElo, Plackett-Luce, ScoreTracker | S-M | D6 |
| D8 | LeagueMatchmaker: команды/роли/ротация мест/все focus-агенты; pluggable | M-L | D1, D6, (D7) |
| D9 | Асимметричные роли: per-role spaces/сети/буферы | L | D1, D2, D8 |
| D10 | Eval v2 по типам игр | M | D1, D6, (D7) |
| D11 | Логирование эпизодных метрик | S | — |

Рекомендуемый порядок: Q1 → D1+D6 → D3 → D11 → D4+D5 → D2 → D7 → D8 → D10 → D9.

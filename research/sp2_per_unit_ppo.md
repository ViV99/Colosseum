# Per-unit PPO/V-trace для Colosseum SP2: что делают практики

Метод: читал первоисточники (код, README, Kaggle-writeup'ы через публичный JSON API Kaggle, PDF статей), а не только пересказы. Пометка [не подтверждено] означает, что утверждение я проверить не смог. Пометка [вывод] означает мою собственную оценку или интерпретацию, а не факт из источника.

Краткий итог:
- В играх с 16-24 юнитами (Gym-μRTS, Lux S1, Lux S3, Lux S2) побеждали или работали с **joint** лог-вероятностью, то есть суммой log-prob по юнитам (ссылки в п. 1).
- Единственный найденный публичный код с **per-entity ratio и per-entity clipping** при общем advantage на таймстеп это `enn-trainer`. Теория (Sun et al.) и MAPPO тоже допускают независимые ratio.
- Для V-trace в больших факторизованных пространствах ближайший прецедент это AlphaStar: независимые группы аргументов, отдельные трассы и value через TD(λ) без off-policy коррекции.
- Per-unit value в победивших решениях я не нашёл. Автор Frog Parade (2-е место Lux S3) прямо пишет, что у него не получилось.

---

## 1. Как считают PPO ratio и clipping для факторизованных multi-unit действий

### 1.1 Joint (сумма log-prob по юнитам и компонентам)

Формула: `ρ = exp(Σ_i log π_new(a_i) − Σ_i log π_old(a_i))`, loss = `−min(ρA, clip(ρ,1±ε)A)`.

**Gym-μRTS / GridNet (Huang et al.)**
- Код: `logprob.sum(1).sum(1)` и `entropy.sum(1).sum(1)`, то есть сумма по всем клеткам и компонентам. `ratio = (newlogproba − b_logprobs).exp()`. Затем PPO-clip, `clip_coef=0.1`, 4 эпохи, 4 минибатча, `ent_coef=0.01`, GAE λ=0.95. https://github.com/vwxyzjn/gym-microrts/blob/master/experiments/ppo_gridnet.py
- Статья: градиент записан как `Σ_t ∇ log Π_{a∈D} π(a|s)`, то есть joint-вероятность. Таблица гиперпараметров в приложении. https://arxiv.org/abs/2105.13807
- Клетки без юнитов получают все логиты равными `mask_value`. Их log-prob константа и градиента нет, но в сумму и в логируемую энтропию они попадают константой. Это видно из кода `CategoricalMasked`.

**Toad Brigade (Lux S1, 1-е место, IMPALA + UPGO + teacher KL)**
- README автора: «Policy losses were computed by summing over the log probabilities of the selected actions for all units that acted in a given timestep, effectively computing the log of the joint probability». https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
- В `monobeast.py` `combined_*_action_log_probs` суммируются по юнитам, клеткам и action-space'ам. Они идут **и** в V-trace (`vtrace.from_action_log_probs`, `clip_rho=clip_pg_rho=1.0`), **и** в UPGO (`min(exp(log_rhos),1)`). Policy loss это обычный `−logπ·adv`, без PPO-clip. https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/blob/main/lux_ai/torchbeast/monobeast.py
- Финальный конфиг: `unroll_length=16`, `discounting=0.999`, `lmb=0.9`, `reduction: sum`, `entropy_cost=0.0002`, `teacher_kl_cost=0.005`. https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/blob/main/conf/conv_phase5%2B_final_model.yaml
- Joint-ratio не схлопывается у них, потому что (а) PPO-clip нет, (б) unroll короткий и лаг мал. [вывод]

**Lux S3**
- Frog Parade (2-е место): «relatively vanilla PPO with clipping … I summed the log-probabilities from all units across the main and sap action distributions to get the joint log-probabilities». Юнитов 16 на команду, GAE с γ=0.9999–1.0. https://www.kaggle.com/competitions/lux-ai-season-3/discussion/568621
- Flat Neurons (1-е место): IMPALA + V-trace + UPGO + TD + teacher KL. Как именно редуцируются log-prob по 16 юнитам, в тексте не сказано [не подтверждено]. https://www.kaggle.com/competitions/lux-ai-season-3/discussion/569562
- EcoBangBang (6-е место): V-trace + teacher KL, действие `(16, 230)`. Автор убрал UPGO pg loss «из-за большой величины в общем loss», после чего рейтинг вырос с ~1400 до 1900+. https://www.kaggle.com/competitions/lux-ai-season-3/discussion/568721
- Сводка RL-решений S3: https://zenn.dev/kurupical/articles/61dbeedf89a29d

**Lux S2 (sgoodfriend, `rl-algo-impls`)**
- Код: `GridnetDistribution.log_prob` суммирует по `(H·W, A)`, затем стандартный PPO-clip на joint ratio. Энтропия тоже суммируется. https://github.com/sgoodfriend/rl-algo-impls (`shared/actor/gridnet.py`, `ppo/ppo.py`)
- Конфиги Lux: `clip_range 0.1`, `n_epochs 4`, `ent_coef 0.01→0.001`. Есть LR по KL (`target_kl 0.01`). Writeup: https://www.kaggle.com/competitions/lux-ai-season-2/discussion/406791
- Победители S2 (writeup'ы не открылись) [не подтверждено]. По сводке kurupical в S2 было много rule-based решений (zenn, ссылка выше).

**MA-Trace (V-trace для MARL, SMAC)**
- `π_ω(a|s) = Π_k π_ω(a_k|s_k)`, ρ_t и c_t считаются по joint, клип на 1.0, центральный critic. Авторы пишут, что при 30 распределённых акторах importance sampling обязателен, без него обучение нестабильно. https://arxiv.org/abs/2111.11229

### 1.2 Per-unit ratio, общий advantage, loss усредняется по юнитам

Формула: `ρ_i = exp(logπ_new(a_i) − logπ_old(a_i))`, `L = −mean_i min(ρ_i A, clip(ρ_i,1±ε) A)`, A общий для команды.

- **enn-trainer** (Entity Neural Network): для каждого action-head и каждого актора свой `ratio`. Advantage таймстепа «broadcast» на всех акторов. `pg_loss = max(−A·ρ, −A·clamp(ρ)).mean()` по конкатенации всех (актор, действие). Клиппинг и `clipfrac`/`approx_kl` тоже per-entity. Энтропия: `torch.cat(entropy.values()).mean()`. В коде оставлены TODO «what's the correct way of combining loss from multiple actions/actors on the same timestep? should we split the advantages across actors?». https://github.com/entity-neural-network/enn-trainer/blob/main/enn_trainer/ppo.py и `train.py`
- **MAPPO** (shared params): `L = (1/(Bn)) Σ_i Σ_k min(r_{θ,i}^{(k)} A_i^{(k)}, clip(...))`, энтропия усредняется по тем же `Bn`. Рекомендация: ε ≤ 0.2 (в экспериментах 0.05–0.2). https://arxiv.org/abs/2103.01955
- **DI-engine** (multi-agent ветка `ppo_policy_error`): `ratio.mean(dim=1)`, то есть среднее ratio по агентам до clip. Энтропия также `.mean(dim=1)`. https://github.com/opendilab/DI-engine/blob/main/ding/rl_utils/ppo.py
- **Song et al., «Joint action loss for PPO»**: «sub-action ratio» `r_i = π_new^i/π_old^i` для каждой под-компоненты. Joint-ratio обнуляет градиент всего сэмпла, если одна компонента вышла из диапазона. Также предложено mixed: `r_mix = w·r_joint + (1−w)·r_sub`, w=0.5. Результат: в Gym-μRTS sub-action loss лучше стандартного PPO при большом ε. Доля неклипнутых сэмплов: compound < mix-ratio < sub-action < mix-loss. https://arxiv.org/abs/2301.10919
- **DISC** (Han & Sung, ICML 2019): `J = (1/M) Σ_m [Π_d min{κρ_{m,d}, κ·clip(ρ_{m,d})}] κÂ_m − α_IS·J_IS`, где `J_IS = (1/2M) Σ (log ρ_m)²`, `κ=sgn(Â)`, α_IS адаптивный (×2 / ÷2 относительно целевого значения). Измерено: при D=3–6 доля обнулённых градиентов 15–20%, при D=17 (Humanoid) >60%. Рост вариации π при независимых размерностях экспоненциальный по D. https://arxiv.org/abs/1905.02363
- **HoK (Honor of Kings)**: «assume independence between action heads», Dual-clip PPO (`max(min(rA, clip(r)A), cA)` при A<0, c>1). https://arxiv.org/abs/2011.12692
- **Негативный результат**: sgoodfriend добавлял `scale_loss_by_num_actions` (вес таймстепа ∝ числу действующих юнитов) и убрал из PPO: «MicroRTS and Lux did horribly with this enabled». https://github.com/sgoodfriend/rl-algo-impls/commit/d45190d Причина в коммите не объяснена.

### 1.3 Теория: независимые ratio против joint

- Sun et al. (AAMAS 2023): монотонное улучшение гарантируется для joint ratio. Независимые ratio работают как trust region над совместной политикой, если ограничить их с учётом числа агентов. Эмпирически `Σ_i D_TV(π_i, π̃_i)` растёт почти пропорционально числу агентов при фиксированном ε. Рекомендованы малые ε (0.03–0.1) на картах с 10–27 агентами. https://arxiv.org/abs/2202.00082
- Там же (разд. 4.5): при ε=0.1 joint-clipping стабильно лучше independent-clipping на картах с 10 и 27 агентами, и разрыв растёт с ε. Поэтому «per-unit с тем же ε» не гарантированно лучше. Это эксперимент на on-policy MAPPO/IPPO в SMAC (≤27 агентов). Для сотен юнитов данных нет [не подтверждено].
- HAPPO: per-agent ratio, умноженное на compound ratio уже обновлённых агентов и на joint advantage `M^{i1:m}`. Это последовательное обновление агентов с отдельными параметрами, для shared-weights сотен юнитов не применимо [вывод]. https://arxiv.org/abs/2109.11251
- DPO (Decentralized Policy Optimization): отдельный surrogate с адаптивными коэффициентами, читал только abstract. https://arxiv.org/abs/2211.03032

### 1.4 Mean log-ratio (геометрическое среднее) — аналог из LLM

- GSPO (Qwen): `s_i(θ) = exp((1/|y|) Σ_t log(π_θ/π_old))`, то есть **длина-нормированный** sequence-level ratio. Клиппинг на уровне всей последовательности с ε = 3e-4 / 4e-4. Авторы объясняют нормировку тем, что иначе изменение нескольких токенов даёт скачки ratio, а разной длине нужны разные ε. https://arxiv.org/abs/2507.18071
- Это LLM, не игры [вывод: прямой аналог для «mean log-ratio по юнитам», но в RL-играх я его в готовых решениях не нашёл].
- Оговорка: GSPO клипует на два порядка больше токенов, чем GRPO, и всё равно обучается эффективнее. Поэтому высокая clip-fraction на joint-уровне сама по себе не обязательно плоха, плоха потеря градиента без компенсации.

### 1.5 Другие крупные проекты

- **OpenAI Five**: 5 героев, у каждого свой сэмпл, PPO clip 0.2, энтропия 0.01→0.001. Как именно склеены ratio разных голов (action/delay/unit/offset), в статье не описано [не подтверждено]. https://arxiv.org/abs/1912.06680
- **Hide-and-seek**: PPO + GAE, shared policy, у каждого агента свой omniscient critic. Это 1–3 агента на команду, масштаб неинформативен. https://arxiv.org/abs/1909.07528
- **Massive-agent Lux** (Chen et al.): pixel-to-pixel PPO + GAE, централизованная политика. https://arxiv.org/abs/2301.01609 Детали ratio в основном тексте не описаны [не подтверждено].
- **Google Research Football**: победитель (WeKick) управляет одним игроком, это не multi-unit. Детали из вторичных источников [не подтверждено]. https://arxiv.org/abs/2305.09458

### 1.6 Оценка масштаба проблемы (мой расчёт, [вывод])

- Пусть per-unit лог-ratio `δ_i ~ N(−s²/2, s²)`, независимы (E ρ_i = 1). Тогда joint `Δ = Σδ_i ~ N(−N s²/2, N s²)`.
- Для N=128, s=0.05: `σ_Δ ≈ 0.57`, медиана joint-ρ ≈ exp(−0.16) ≈ 0.85, и вне [0.8, 1.2] оказывается порядка 80% сэмплов. Это согласуется с вашим замером 0.91 (K=128).
- Для s=0.1 медиана ρ ≈ 0.53. Произведение таких ρ по нескольким шагам трасс V-trace быстро стремится к 0.
- Сходная картина у DISC (>60% обнулённых при D=17) и у Sun et al. (сумма TV растёт ∝ N).

---

## 2. V-trace importance weights в больших факторизованных пространствах

**AlphaStar (Nature 2019, Methods)**
- «off-policy correction methods like V-trace can be inefficient in large, structured action spaces … because distinct actions can result in similar (or even identical) behaviour. We address this by using a hybrid approach. The policy is updated using V-trace and the value estimates are updated using TD(λ), which does not apply off-policy corrections».
- «To mitigate early trace cutting due to the large action space, we assume independence between the action type, delay, and all other arguments, and so update them separately».
- Ссылки: https://www.nature.com/articles/s41586-019-1724-z, PDF с Methods: https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf
- Также UPGO с `ρ = min(π/π_actor, 1)` и отдельные value-функции на каждый pseudo-reward.
- [Не подтверждено]: из Methods не ясно, считается ли ρ внутри группы «все остальные аргументы» как произведение по аргументам.

**AlphaStar Unplugged (DeepMind, 2023)**
- Offline actor-critic: IS-коррекция применена «only for some of the arguments, namely function and delay, and we use the behavior policy for the other ones». ρ и c клипуются в 1. Рисунок 7b показывает, как клипнутое ρ падает со временем при критике V^π, и авторы трактуют это как расхождение π и μ. https://arxiv.org/abs/2308.03526

**Что ломается (прямые свидетельства)**
- «Early trace cutting» (AlphaStar, цитата выше).
- Экспоненциальный рост вариации ρ с размерностью (DISC, https://arxiv.org/abs/1905.02363).
- Рост суммарного TV пропорционально числу агентов (Sun et al., https://arxiv.org/abs/2202.00082).
- В Toad Brigade V-trace/UPGO работают на joint ρ без видимых проблем, но при unroll=16 и малом лаге. EcoBangBang убрал UPGO pg loss (ссылка выше). Причинная связь с масштабом ρ не доказана [не подтверждено].
- Для V-trace с per-unit ρ в играх готовых реализаций я не нашёл [не подтверждено]. Ближайший прецедент это AlphaStar (независимые группы).
- V-trace с геометрическим средним ρ в литературе по RL я не нашёл [не подтверждено]. Идея взята из GSPO (LLM).

---

## 3. Нормировка энтропии при многих юнитах

| Решение | Редукция энтропии | Коэффициент |
|---|---|---|
| Gym-μRTS / GridNet | **сумма** по клеткам и компонентам, затем mean по батчу | 0.01 (https://github.com/vwxyzjn/gym-microrts/blob/master/experiments/ppo_gridnet.py) |
| Toad Brigade | **сумма** по юнитам (только действовавшим), reduction=sum по батчу | 2e-4, при этом pg loss тоже sum (конфиг по ссылке в п. 1) |
| sgoodfriend (Lux S2) | сумма по клеткам, mean по батчу | 0.01→0.001 (https://github.com/sgoodfriend/rl-algo-impls) |
| enn-trainer | **mean** по всем (актор, действие) | `ent_coef` (https://github.com/entity-neural-network/enn-trainer/blob/main/enn_trainer/train.py) |
| MAPPO, DI-engine | **mean** по агентам | 0.01 в MAPPO |
| Flat Neurons (Lux S3) | адаптивный коэффициент к **целевой энтропии** на голову: 0.9 (движение) и 3.9 (sap-цель), линейно до 0 за 100M шагов; после 100M сброс на 0.45 и 2.0 | https://www.kaggle.com/competitions/lux-ai-season-3/discussion/569562 |
| OpenAI Five | не описано | 0.01→0.001 (https://arxiv.org/abs/1912.06680) |

- Что у Flat Neurons редуцируется по юнитам, не сказано [не подтверждено]. Цели 0.9 и 3.9 лежат ниже `ln 6 = 1.79` и `ln 225 = 5.4`, что похоже на энтропию на один юнит [вывод].
- Практически: те, кто суммирует энтропию, суммируют и pg loss, так что относительный вес сохраняется, но коэффициент зависит от числа юнитов. Те, кто усредняет, делают это с тем же знаменателем, что и pg loss.
- AlphaStar про редукцию энтропии пишет только «standard entropy regularisation». [не подтверждено]

---

## 4. Per-unit награда и value против командной

- **Toad Brigade** (S1): одна награда ±1 в конце, один critic ∈[−1,1] (README). https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
- **Flat Neurons** (S3): награда это победа в матче (zero-sum, масштабируется скользящим средним в диапазон [−5,+5]). Один baseline от Transformer по обеим командам. Пробовали награды за очки, урон, видимость, остались только match win. https://www.kaggle.com/competitions/lux-ai-season-3/discussion/569562
- **Frog Parade** (S3): value = mean по карте 24×24, одно число на команду. Цитата: «I made some attempts early on to factorize the value function on a per-unit level, but was unable to figure out how to make it work successfully … I'd be very curious to know if anyone got a per-unit value factorization … working!». https://www.kaggle.com/competitions/lux-ai-season-3/discussion/568621
- **EcoBangBang** (S3): match win/loss и game win/loss, zero-sum baseline. https://www.kaggle.com/competitions/lux-ai-season-3/discussion/568721
- **sgoodfriend** (S2): один critic, но **несколько голов value по потокам reward** (shaped / win-loss / score, `multi_reward_weights`). Плотная награда отжигается к разреженной. https://www.kaggle.com/competitions/lux-ai-season-2-neurips-stage-2/discussion/459891
- **Massive-agent Lux**: pixel-to-pixel «avoids the credit assignment problem», один critic, flatten+FC. https://arxiv.org/abs/2301.01609
- **OpenAI Five**: per-hero reward, `r_i = (1−τ)ρ_i + τρ̄`, «team spirit» τ растёт 0.3→0.8→1.0, у каждого героя свой value. Масштаб 5 агентов. https://arxiv.org/abs/1912.06680
- **HoK**: multi-head value по декомпозированной награде. https://arxiv.org/abs/2011.12692
- **AlphaStar**: отдельные value-функции на каждый pseudo-reward (не на юнит). https://www.nature.com/articles/s41586-019-1724-z
- **MAPPO**: critic на агента, но «death masking» (вход critic для мёртвого агента заменяется на нулевой вектор с agent-ID) заметно лучше остальных вариантов. https://arxiv.org/abs/2103.01955
- GRF: единственный управляемый игрок, командный reward [не подтверждено].

Вывод по доказательной базе: для сотен юнитов с рождением и смертью найдено только «одна команда, один value» (+ multi-head по потокам reward). Per-unit value работал лишь при малом числе агентов с фиксированными ролями (Five, MAPPO).

---

## 5. Мёртвые, несуществующие юниты и юниты без легальных действий

- **Toad Brigade**: `actions_taken_mask` обнуляет вероятности неиспользованных действий до расчёта лог-вероятности. Нулевые условные вероятности заменяются на 1, чтобы не получить −inf. Энтропия: `log_policy.isneginf() → 0`, затем `* actions_taken_mask`. KL и энтропия считаются только в клетках, где хоть один юнит действовал (`any_actions_taken`). Код: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021/blob/main/lux_ai/torchbeast/monobeast.py
- **Условные подголовы**: Flat Neurons учат sap-цель «only on timesteps where the first head selected sap», первую голову на всех шагах (https://www.kaggle.com/competitions/lux-ai-season-3/discussion/569562). У sgoodfriend: `torch.where(action[ref]==value, c.log_prob(a), 0)` (`gridnet.py`, ссылка выше).
- **Gym-μRTS**: маска из нулей даёт равномерное распределение с константным log-prob, градиента нет. Но такая клетка остаётся в сумме и в логе энтропии как константа. Теоретическая часть: маскирование большим отрицательным числом это корректный policy gradient, градиент по замаскированному логиту нулевой. https://arxiv.org/abs/2006.14171
- **Lux S3**: `units_mask` даёт живые юниты. Frog Parade индексирует только живых и считает sap-маски по каждому юниту отдельно. Правила: ID юнитов `0..max_units−1`, **переиспользуются** при респауне. https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/docs/specs.md
- **MAPPO**: death masking для value (п. 4).
- Инженерные правила [вывод, не из источников]:
  - вычислять ratio как `exp((new−old)·mask)` через `torch.where`, а не умножением (0·NaN = NaN);
  - строка, где все действия замаскированы, не равномерное распределение, а `valid=False` с нулевым весом в ratio, энтропии и KL;
  - нормировать на `max(1, n_valid)`;
  - один и тот же `valid`-маска-тензор применять в ratio, clip-метриках, энтропии, KL и BC.

---

## 6. Представление: GridNet `[H,W,A]` против entity-list `[U_max, A]` + `unit_mask`

**GridNet / per-cell**
- Плюсы: обычная CNN, размер выхода не зависит от числа юнитов, естественная трансляция пространственных отношений. Gym-μRTS, Toad Brigade, sgoodfriend S2, massive-agent Lux. https://arxiv.org/abs/2105.13807
- Минусы:
  - Несколько юнитов на одной клетке. У Toad Brigade сэмплирование без возвращения и условные вероятности; Flat Neurons и Frog Parade различают юнитов по энергии, подавая её отдельно.
  - Большая часть выхода «пустая» (нужна маска).
  - TStarBot-X: при scatter/gather + conv «the trained agent prefers to control a group of units that are spatially close to each other and barely controls one or a few units for specific micro-management». https://arxiv.org/abs/2011.13729

**Entity-list `[U_max, A]`**
- AlphaStar: pointer network по списку юнитов, до 64 выборов авторегрессивно. https://arxiv.org/abs/2308.03526
- OpenAI Five: unit selection это softmax по 189 видимым юнитам. https://arxiv.org/abs/1912.06680
- Lux S3: Frog Parade индексирует живых юнитов по позиции из общего поля → `n_units × d_model` (https://www.kaggle.com/competitions/lux-ai-season-3/discussion/568621). Flat Neurons вырезают патч 15×15 вокруг каждого из 16 юнитов (https://www.kaggle.com/competitions/lux-ai-season-3/discussion/569562). EcoBangBang берёт вектор в точке `(x,y)` плюс энергию (https://www.kaggle.com/competitions/lux-ai-season-3/discussion/568721).
- Минусы: слоты и их стабильность. В Lux S3 действие имеет форму `(max_units, 3)`, ID переиспользуются. Нельзя считать слот постоянной идентичностью юнита (рекуррентное состояние на слот, статистика на слот).
- TStarBot-X: независимый multi-binary выбор юнитов «well works and is much more efficient», но «unavoidably suffers from the unit independency» (юниты часто пропускаются), поэтому они взяли последовательный. https://arxiv.org/abs/2011.13729 Для нас это довод против авторегрессии только там, где нужна координация внутри выбора.

**Что это значит для фреймворка** [вывод]: единым примитивом для loss сделать entity-список (`logp[B,T,U]`, `valid[B,T,U]`, `entropy[B,T,U]`). GridNet реализовать как «вид»: gather логитов в позициях юнитов в список (и опционально scatter обратно). Тогда одна и та же реализация ratio/V-trace/энтропии/диагностик работает для обоих. Исключение: «per-cell» действия, где субъект это клетка (например, здание без юнита), здесь `entity = клетка с владельцем`.

---

## 7. Размер ε и диагностика

- ε для per-agent ratio: MAPPO советует <0.2 и подбирать между 0.05 и 0.2 (https://arxiv.org/abs/2103.01955). Sun et al. используют 0.1/0.05 для 10 и 27 агентов и пишут про 0.03–0.08 как предпочтительные при большом числе агентов (https://arxiv.org/abs/2202.00082). Формулы «ε как функция N» в источниках нет [не подтверждено]. Единственный принцип: сумма TV растёт ∝ N, значит ε надо уменьшать.
- Gym-μRTS: ε=0.1 для joint при ~16×16 картах, есть опции `target-kl` (0.03) со стоп/rollback (https://github.com/vwxyzjn/gym-microrts/blob/master/experiments/ppo_gridnet.py). sgoodfriend: ε=0.1 и LR-контроллер по KL (target 0.01), отдельно отслеживает L2-норму градиента и пишет, что KL выше ~0.02 было слишком высоко (https://www.kaggle.com/competitions/lux-ai-season-2-neurips-stage-2/discussion/459891).
- DISC добавляет штраф `(log ρ)²/2` с адаптивным α (https://arxiv.org/abs/1905.02363). Это единственная найденная явная «пружина» на |log-ratio|.
- Что логировать (в основном [вывод], подкреплено практикой):
  - clip_fraction отдельно per-unit и joint (enn-trainer усредняет по головам; sgoodfriend логирует `clipped_frac`, `approx_kl`, `val_clip_frac`);
  - approx_kl как mean по валидным юнитам (k3: `(ρ−1)−log ρ`). На joint k3 неустойчив при больших |Δ|;
  - mean и p95 `|log ρ_i|` и `|Σ log ρ_i|`;
  - ESS = `(Σρ)²/Σρ²` по батчу для joint-весов;
  - доля шагов, где клипнуты ρ и c в V-trace, и средний произведённый trace за окно (в AlphaStar Unplugged клипнутое ρ как раз используется как диагностика расхождения);
  - L2-норма градиента;
  - число валидных юнитов на шаг (его распределение определяет масштаб joint).

---

# Рекомендации для Colosseum

Рекомендации это мои выводы на основе перечисленных источников, а не готовый рецепт из литературы.

**Общий принцип**: делить поведение по числу единиц, действующих в одном «seat»-шаге. Для малого числа компонентов (≤ ~8, как сегодня) оставить текущий joint. Для entity/grid-голов включить per-unit режимы.

1. **`ratio_mode: joint | per_unit | geo_mean | mixed`**
   - Default для entity/grid-голов: **`per_unit`** (ratio и clip на юнит, общий A, loss = masked-mean по валидным юнитам; ближайший прецедент enn-trainer, MAPPO, Song et al. https://arxiv.org/abs/2301.10919).
   - `joint` оставить для малых K и для воспроизведения результатов Gym-μRTS/Lux (эти решения работали при K≈16–24).
   - `geo_mean` (GSPO-подобный, клип по `exp(mean log ρ_i)`) как экспериментальный. `mixed` с весом `w` (Song et al.).
   - Обоснование: ваш замер 0.00→0.91 и п. 1.6 показывают, что joint клип при K≥~100 обнуляет градиент.

2. **`clip_eps` + `clip_eps_scale: none | inv_sqrt_units`** (экспериментально)
   - Per-unit ε по умолчанию 0.1, ниже при большом N (Sun et al.: сумма TV ∝ N).
   - Масштаб по `1/√N_valid` это моя эвристика [не подтверждено в литературе].

3. **`ratio_safeguard`: `none | log_ratio_penalty | kl_early_stop`**
   - Joint-страховка поверх per-unit: штраф `(Σ_i log ρ_i / N)²` или `(log ρ)²/2` как в DISC, либо early-stop эпох по mean-KL (как `target-kl` в Gym-μRTS). Per-unit клиппинг сам по себе не ограничивает совместный сдвиг (Sun et al.).

4. **`vtrace.rho_mode: joint | per_unit | geo_mean`**, **`vtrace.value_target_traces: joint | geo_mean | none`**
   - Default для entity-голов: скалярные ρ/c для value-таргета и δ по **geo_mean** (или `none` как «TD(λ) без коррекции» в AlphaStar), а для policy-gradient веса per-unit клипнутое `min(ρ_i, ρ̄_pg)`. `joint` как legacy и для малого K.
   - Обоснование: AlphaStar разводит независимые группы и value через TD(λ) «to mitigate early trace cutting» (Nature, п. 2). Формулировка «per-unit pg-вес и скалярная трасса» моё предложение. Для `geo_mean` в V-trace прецедента нет (только GSPO в LLM), поэтому обязательно A/B с joint на небольшом K (там joint известно рабочий).

5. **`entropy_reduction: mean_valid | sum | target_entropy`**
   - Default **`mean_valid`** (masked-mean по валидным юнитам с тем же знаменателем, что у pg loss): коэффициент переносим между N (MAPPO, enn-trainer, DI-engine).
   - `sum` для воспроизведения Toad Brigade / Gym-μRTS, где pg тоже sum.
   - `target_entropy` по голове (Flat Neurons: https://www.kaggle.com/competitions/lux-ai-season-3/discussion/569562) как опция для длинных запусков.
   - Правило: pg loss, entropy и KL/kickstart должны делить **один** способ редукции, иначе эффективный коэффициент зависит от N.

6. **`value_mode: team` (default) | `team_multihead`**
   - Один командный value плюс опционально multi-head по потокам reward (AlphaStar, HoK, sgoodfriend). Per-unit value не включать: у победителей я его не нашёл, а Frog Parade прямо сообщает о неудаче.
   - Для мёртвых/отсутствующих seat'ов при per-seat value: death-masking как в MAPPO.

7. **Маскирование**
   - Единый `unit_valid[B,T,U]` (существует и действовал) для ratio, clip-метрик, энтропии, KL, BC. Условные подголовы (sap-цель) учить только там, где родительская голова выбрала действие. `torch.where`, а не умножение, без деления на 0.

8. **Представление**
   - Базовый примитив: entity-список `[U_max, …]` + `unit_mask` + стабильный `unit_id` (но слот не считать постоянной идентичностью, как в Lux S3). GridNet как вид через gather/scatter, одна реализация loss на оба.

9. **Диагностика** (всегда включена в entity-режимах): per-unit и joint `clip_frac`, `approx_kl` (mean per-unit), mean/p95 `|log ρ_i|`, ESS, доля клипа ρ/c в V-trace, grad-norm, распределение `N_valid`.

Приоритет проверки: A/B на небольшом K=8–16 (где joint известно работает) и на K=128, сравнивая `ratio_mode` и `vtrace.rho_mode` по win-rate и clip_frac на одном и том же бюджете.

---

Ограничения исследования:
- Содержимое Kaggle-writeup'ов получено через публичный JSON API Kaggle. Статьи читались из PDF/HTML. Страницы W&B-отчётов sgoodfriend не открылись.
- Не найдены (или не открылись) writeup'ы победителей Lux S2 и 10-го места S3 (Boey, JAX RL). Для них [не подтверждено].
- Ни в одном найденном источнике нет эксперимента «joint против per-unit ratio при сотнях юнитов». Все прямые данные при ≤27 агентах (Sun et al., SMAC) или ≤24 юнитах (Lux, μRTS).

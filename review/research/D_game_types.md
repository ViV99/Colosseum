# R4. Обучение ботов для разных типов игр: обзор для универсального фреймворка Colosseum

Дата: 2026-10-07. Автор: исследовательский агент (RL для игровых соревнований).
Методика: WebSearch/WebFetch по первоисточникам (arXiv/ar5iv, Nature PDF, GitHub README/код решений, docs библиотек, Context7 для PettingZoo/RLlib). Ничего не устанавливалось, бенчмарки не запускались. PDF-файлы разбирались локально через `pdftotext`.

Соглашения о пометках:
- **[не подтверждено]** — не удалось проверить в первоисточнике в этой сессии (источник недоступен, вторичная сводка или память).
- **[вывод]** / **[предложение]** — мой вывод или предложение для Colosseum, не утверждение источника.
- Страницы Kaggle (discussion/writeups) WebFetch не отдаёт (только заголовок). Поэтому writeup Toad Brigade прочитан через README их GitHub (он дословно воспроизводит writeup), writeup Frog Parade — через `write-up.md` в их репозитории. 1st-place writeup Hungry Geese (Kaggle 263279) и Lux S3 Flat Neurons прочитать **не удалось**.

---

## 1. Резюме

1. **Что реально побеждало в Kaggle-симуляциях** (обзор nagiss, DeNA, 2022: https://speakerdeck.com/nagiss/kagglesimiyuresiyonkonpenodong-xiang): RL доминировал в Lux S1 и Google Research Football (GRF); Halite IV и Kore 2022 — чистые rule-based (все топ-5 Kore); ConnectX — поиск; Hungry Geese — HandyRL + поиск/ансамбли; Pommerman 2018 — tree search на 1–3 местах, RL 4–5. Вывод [вывод]: фреймворк обязан одинаково хорошо работать с **RL-агентами, скриптовыми ботами и поиском (MCTS)** как равноправными «контроллерами слота», иначе он не покроет реальные соревнования.
2. **Один бот — много юнитов (тип 3)**: де-факто стандарт — карта-в-карту (GridNet/ResNet/U-Net) с **суммой log-prob по действующим юнитам** как совместным log-prob (Gym-μRTS/CleanRL, Toad Brigade, Frog Parade, GridNet). Альтернативы: entity-attention с общими весами (hide-and-seek, OpenAI Five) и авторегрессивный pointer-«командир» (AlphaStar; очень дорого). Размеры: Toad Brigade ~20M параметров (24 блока, 128 каналов, 5x5), Frog Parade (Lux S3) ~10M (8 блоков, d_model=256), AlphaStar 139M весов.
3. **Главный скрытый риск для APPO/V-trace Colosseum**: совместное отношение вероятностей = произведение по юнитам; при сотнях юнитов оно уходит от 1 экспоненциально, PPO-клиппинг зануляет градиенты, V-trace-веса схлопываются (DISC; Trust-Region-Bounds; AlphaStar прямо пишет, что V-trace неэффективен в больших структурных пространствах действий и использует V-trace только для политики, а TD(λ) для value). Нужны настраиваемые: редукция log-prob (sum/mean/per-unit ratio), ε, зависящий от числа юнитов (0.1/0.05/0.03 для 5/10/27 агентов), логирование clipfrac/KL.
4. **Интерфейс среды** надо разделить на три сущности: **slot** (участник матча, кому принадлежит политика/контроллер; роль, команда), **entity** (юниты внутри слота, переменное число, маски) и **acting set** (кто ходит на этом шаге). Результат матча — вектор score + rank + outcome в [-1,1] (подход HandyRL: попарное сравнение рангов, ±1/(N-1)) + производный WDL.
5. **FFA/N игроков**: награда — ранговая/попарная (HandyRL Hungry Geese), рейтинг — OpenSkill Plackett-Luce / TrueSkill (мульти-команда с рангами), оппоненты — PFSP по попарным win-rate + смесь прошлых чекпоинтов и скриптов; seat-ротация обязательна (HandyRL: first/second для 2p, случайная перестановка для N>2).
6. **Асимметричные роли**: либо одна сеть с role-embedding (SearchBot Diplomacy: 60-мерный embedding державы; OpenAI Five: идентификатор героя во входе), либо отдельные агенты на роль (AlphaStar: свой main/exploiter на каждую расу; Neural MMO v1: популяции с несвязанными весами). Теория: асимметричную игру раскладывают на симметричные «популяции по ролям» (Tuyls et al. 2018); PSRO для многопопуляционных игр требует отдельных популяций на роль.
7. **Соло**: заменитель self-play — Ranked Reward (R2: перцентильная бинаризация награды относительно буфера собственных недавних результатов, 250 эпизодов, 75-й перцентиль) + PLR для сидов/уровней; оценка — распределения очков, IQM/bootstrap CI (rliable), парные сиды.
8. **Команда независимых ботов (тип 4)**: Skynet (Pommerman, 2 идентичные сети, без коммуникации), TiKick (все 10 игроков GRF — общие параметры), hide-and-seek (общие параметры + омнисциентный critic, который критичен), FTW (независимые агенты из популяции PBT, 30 агентов). Централизованный критик с глобальным состоянием (доступно только при обучении) — must-have.

---

## 2. Разделы A–F

### A. Управление множеством юнитов одним ботом

#### A.1. Семейства архитектур

| Подход | Примеры | Выход | Плюсы | Минусы |
|---|---|---|---|---|
| **Per-cell action map (GridNet, fully-conv)** | GridNet (Han 2019), Gym-μRTS, Toad Brigade (Lux S1), Frog Parade (Lux S3), U-Net в Lux S2 | тензор [A,H,W]; действие юнита берётся из его клетки | вычислительная стоимость не зависит от числа юнитов; receptive field даёт «коммуникацию»; общие свёрточные параметры (опыт одного юнита сразу полезен всем); пустые клетки маскируются | нужна сеточная структура; при малом числе юнитов дорого (Gym-μRTS: UAS быстрее при малом числе юнитов, GridNet — при большом) |
| **Per-unit shared policy + entity embeddings / attention** | hide-and-seek (OpenAI), OpenAI Five, IPPO/MAPPO | действие на сущность из общей сети | работает без сетки, перестановочная инвариантность, переменное число сущностей через маски | ресурсоёмко при сотнях юнитов (attention O(E²)); нужен явный механизм коммуникации |
| **Автогрессивный «командир» + pointer** | AlphaStar | последовательно: тип действия, задержка, выбранные юниты (pointer), цель | выразительно, моделирует зависимость между действиями | очень дорого; 139M весов; не масштабируется на сотни юнитов без упрощений |
| **UAS (Unit Action Simulation)** | Gym-μRTS | итеративный вызов политики: выбрать юнит, затем его действие | логитов ~301 вместо 50 млн для 16x16 | много симуляционных шагов при многих юнитах |
| **Rule/поиск/ассигнование** | Halite IV (4-е место 0Zeta: 100% rules + linear sum assignment), Kore 2022 (rules) | детерминированно | выигрывали там, где RL не заработал | нет обучения |

Факты и числа:

- **GridNet (Han et al., ICML 2019)**: «Политика» = **произведение вероятностей по клеткам**: π(a|s)=∏_g π_g(a_g|s); пустые клетки маскируются из совместной политики; критик централизованный на общем представлении энкодера; энкодер-декодер из conv/pool, декодер 1x1 conv + опциональный upsampling; для гетерогенных агентов канал действий = дизъюнктное объединение пространств действий. Интегрируется с PPO (advantage A = r + v(s') - v(s)) и (через DPG) с Q-learning. URL: https://proceedings.mlr.press/v97/han19a/han19a.pdf (проверено локальным разбором PDF).
- **Gym-μRTS (Huang et al. 2021, arXiv 2105.13807)**: GridNet выдаёт (h,w,79) логитов (79 = 1 маска source unit + 6 типов действий + 4x4 параметров направлений + 7 типов produce + 49 позиций атаки); UAS даёт hw+36+a_r² ≈ 301 логитов против 9216·(hw a_r²) ≈ 50 млн для сырого пространства. Гиперпараметры PPO (Table III): 300M шагов, 24 env, 256 (UAS) / 512 (GridNet) шагов на env, 4 минибатча, γ=0.99, λ(GAE)=0.95, **ε=0.1**, K=4 эпохи, lr 2.5e-4 с линейным затуханием, ent-coef 0.01, vf-coef 0.5, grad-norm 0.5. Лучший агент: 91% win rate против соревновательных ботов, **63 часа обучения на 1 GPU/3 vCPU/16GB**; пул разнообразных оппонентов (18 CoacAI, по 2 RandomBiased/WorkerRush/LightRush). Различие: UAS присваивает награду индивидуальным действиям юнитов, GridNet — «коллективному» действию игрока; GridNet получал выше shaped-return, но sparse-return сопоставим с UAS. URL: https://ar5iv.labs.arxiv.org/html/2105.13807 , PDF https://arxiv.org/pdf/2105.13807 .
- **Код CleanRL/gym-microrts `ppo_gridnet.py`** (проверено): энкодер Conv(32,3x3)-MaxPool-Conv(64)-MaxPool; декодер 2 ConvTranspose (→78 каналов логитов); critic: Flatten→Linear(64·4·4,128)→Linear(128,1) (для 16x16); **`logprob.sum(1).sum(1)` и `entropy.sum(1).sum(1)`** — суммирование по клеткам и компонентам; маска: `torch.where(masks.bool(), logits, -1e8)`. URL: https://raw.githubusercontent.com/vwxyzjn/gym-microrts/master/experiments/ppo_gridnet.py
- **Toad Brigade (Lux S1, 1-е место)**: полностью свёрточный ResNet с squeeze-excitation, **24 блока, 128 каналов, 5x5, без нормализации, ~20M параметров**; входы паддятся до 32x32 с маской; discrete-каналы — 32-мерные embedding-и, затем 1x1 conv до 128x32x32; три actor-головы (workers 19 действий, carts 17, city tiles 4) + один critic в [-1,1]. Алгоритм: IMPALA (FAIR/TorchBeast) + **UPGO + TD(λ)** + **KL к замороженному teacher** (стабилизация self-play). Награда: shaped первые 20M шагов (за города/юниты/research/топливо), затем разреженная ±1; **прогрессия 8→16→24 блока с меньшей сетью как teacher**. Совместная вероятность: действия юнитов сэмплируются **без возвращения до no-op**, log-prob — вероятность этой последовательности; политический лосс — **сумма log-prob выбранных действий**. Маскирование нелегальных действий `-inf`. Инференс: 180° поворот, усреднение вероятностей с оригиналом, greedy; rule-based разрешение конфликтов по порядку вероятностей; 2–2.5 с на матч при batch=2. Железо: личный ПК, 8 ядер/16 потоков, 2 GPU (число шагов обучения в README не указано). URL: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021 ; writeup (контент не загрузился, только заголовок): https://www.kaggle.com/c/lux-ai-2021/discussion/294993
- **Frog Parade (Lux S3, gold; автор тот же Isaiah Pressman)**: SE-ResNet, 8 блоков 3x3, d_model=256, **~10M параметров**, 430 шагов/с; ~300M игровых шагов (600M per-player наблюдений), ~8 суток, плато около 200M; PPO (clip, entropy, teacher-KL), GAE-λ с γ≈0.9999–1.0, награда ±1 в конце матча (100 шагов) с ранней остановкой при 3 очках матча; actor head даёт `n_units × n_actions`, энергия юнита подаётся и на вход, и в actor head (чтобы различать совмещённые юниты); sap-цель — общая 2-слойная CNN-голова 1x24x24 с per-unit маской; **«суммируем log-prob всех юнитов по main и sap распределениям — совместный log-prob для policy loss»**. ~80 глобальных + ~100 пространственных признаков, 10 кадров истории; Rust-симулятор (~110k шагов/с без инференса). TTA: две диагональные рефлексии + поворот 180° с усреднением политики. Замечание автора: маскирование было излишне строгим. Железо: Ryzen 9950X, 64GB RAM, RTX 3090 + RTX 2070 Super. URL: https://raw.githubusercontent.com/IsaiahPressman/kaggle-lux-2024/main/write-up.md
- **Lux S3 официальный API**: функциональная JAX-среда, действия `[max_units, 3]` на команду, максимум 16 юнитов, матч 100 шагов, best-of-5 на одной карте, fog of war. URL: https://github.com/Lux-AI-Challenge/Lux-Design-S3 .
- **Другие Lux S3 RL-решения** (вторичный источник — японский обзор, **цифры внутренне противоречивы и не совпадают с писаниями авторов, поэтому [не подтверждено]**): утверждается, что 1-е место Flat Neurons — IMPALA, ResNet+ConvLSTM+Transformer, 3–4 суток на 8xH100; 6-е EcoBangBang — IMPALA/ResNet с action space (16,230); 10-е Boey — PPO с BPTT на JAX, 75% latest + 25% PFSP; все команды зеркалили карту в перспективу игрока 0; награда сошлась к win/loss, нормированная в [-5,+5]. URL: https://zenn.dev/kurupical/articles/61dbeedf89a29d . (Их числа вроде «300M×2 параметров» и «23B» противоречат Frog Parade writeup — не использовать.)
- **AlphaStar (Nature 2019)**: 139M весов (55M нужны на инференсе); наблюдения кодируются, объединяются, проходят глубокий LSTM; аргументы действия сэмплируются **авторегрессивно**; ablation: scatter connections, transformer, pointer network. Источник: https://www.nature.com/articles/s41586-019-1724-z (PDF разобран локально, Methods).
- **OpenAI Five (arXiv 1912.06680)**: один общий LSTM на 5 героев, **4096 юнитов, ~158.5M параметров, 84% — LSTM**; идентификатор героя подаётся в observation; ~16 000 входных значений, до 189 юнитов на карте; 8 000–80 000 дискретных действий в зависимости от героя. URL: https://ar5iv.labs.arxiv.org/html/1912.06680
- **Halite IV**: 4-е место (0Zeta) — **100% rule-based**, linear sum assignment для одновременного назначения действий кораблям, авторы пишут, что не нашли способа заставить «player-based» RL работать в 4-игроковом формате. URL: https://raw.githubusercontent.com/0Zeta/HaliteIV-Bot/master/README.md . Обзор nagiss: «чемпион — более 11 000 строк Python, позиции 2–6 тоже rule-based» (https://speakerdeck.com/nagiss/kagglesimiyuresiyonkonpenodong-xiang). Репозиторий победителя ttvand/Halite содержит отдельные папки rule-based и Deep Learning агентов (детали не описаны): https://github.com/ttvand/Halite
- **Kore 2022**: по обзору nagiss, все топ-5 — rule-based (победитель 5000+ строк). Writeup победителя не найден [не подтверждено в первоисточнике].
- **Lux S2**: 1-е место ryandy — https://github.com/ryandy/Lux-S2-public (подход по README не определён; считаем [не подтверждено]). Базовые RL-решения: PPO, pixel-to-pixel U-Net с invalid action masking (в wandb-отчёте sgoodfriend) — отчёт не загрузился (JS), по сводке поиска: PPO ~20M шагов, self-play + исторические оппоненты, награда линейно сдвигается от shaped к итоговой, U-Net-подобная сеть: https://wandb.ai/sgoodfriend/rl-algo-impls-benchmarks/reports/Lux-AI-Season-2-Training--Vmlldzo0MTc4NDQz [не подтверждено].

#### A.2. Log-prob, энтропия и PPO-клиппинг при сотнях юнитов

Факты:
- Все найденные реализации для «одного бота — много юнитов» берут **сумму log-prob по юнитам** (Gym-μRTS/CleanRL: `.sum(1).sum(1)`; Toad Brigade и Frog Parade: сумма по юнитам; GridNet: произведение вероятностей клеток).
- Энтропия в CleanRL-GridNet суммируется так же (`entropy.sum(1).sum(1)`) — значит, **эффективный коэффициент энтропии растёт с числом юнитов** [вывод].
- **Проблема произведения отношений**: DISC (arXiv 1905.02363) — IS-вес ρ_t=∏_d ρ_{t,d} при росте размерности действия уходит от 1; для Humanoid (17 измерений) более 60% сэмплов получают нулевой градиент от клиппинга против 15–20% у Hopper (3 измерения); решение — клиппинг по измерению. https://ar5iv.labs.arxiv.org/html/1905.02363
- **«Trust Region Bounds for Decentralized PPO Under Non-stationarity»** (arXiv 2202.00082): совместный (joint) vs независимый (independent) клиппинг; рекомендуемые ε падают с числом агентов: **5 агентов ε=0.1, 10 — 0.05, 27 — 0.03**; trust region при ε=0.1 «растёт от <0.3 до >0.5» с ростом числа агентов; в экспериментах joint-ratio клиппинг на картах с большим числом агентов работает лучше independent. https://ar5iv.labs.arxiv.org/html/2202.00082
- **MAPPO (Yu et al. 2021)**: ε<0.2, 5–15 эпох, без дробления на много минибатчей, value normalization (бегущие оценки), общий для однородных агентов набор параметров лучше отдельных. https://ar5iv.labs.arxiv.org/html/2103.01955 (прочитано 100 тыс. символов из 111 тыс.).
- **AlphaStar**: «V-trace-коррекции неэффективны в больших структурных пространствах действий, т.к. разные действия дают сходное поведение»; поэтому **политика — V-trace, value — TD(λ) без off-policy коррекции** (ablation Fig. 3I); value-функции получают на вход и наблюдения оппонента (Fig. 3K). Дополнительно UPGO и KL к supervised-политике. Nature PDF.
- Gym-μRTS ε=0.1; Toad Brigade обходится IMPALA/V-trace (клиппинг ρ).

Рекомендации (все [вывод]):
1. Сделать редукцию configurable: `logp_reduction ∈ {sum, mean, sum/sqrt(N)}`, `entropy_reduction` отдельно (по умолчанию mean по активным юнитам × coef), логировать распределение log-ratio, `clipfrac`, `approx_kl`, эффективное число юнитов N_eff.
2. Режим «per-unit ratio»: каждая компонента клиппится независимо, advantage общий (IPPO с parameter sharing / DISC) — как альтернатива joint-ratio (оба режима есть в литературе, joint лучше на больших картах по 2202.00082 — значит, делать **A/B** на задаче, а не выбирать по умолчанию).
3. ε = f(N): ε_eff = ε_0 · min(1, N_ref/N) как стартовая эвристика (подтверждена эмпирикой 2202.00082 только для 5–27 агентов; для сотен — [не подтверждено], нужна проверка).
4. Для V-trace считать ρ, c̄ на уровне юнита (с агрегацией через min/mean) или использовать гибрид AlphaStar (V-trace для policy-lag, TD(λ)/GAE для value). Joint ρ на сотнях юнитов практически всегда усечён до 1/до 0 [вывод].

#### A.3. Маскирование и переменное число юнитов

- **Invalid action masking** (Huang & Ontañón 2020, arXiv 2006.14171; Gym-μRTS): логиты нелегальных действий заменяются большим отрицательным числом (M=-1e8), градиент по ним нулевой; в Gym-μRTS полная маска даёт «значимый» прирост для GridNet, для UAS — умеренный; частичная маска (как в PySC2: маска только типов действий, без параметров) хуже. https://arxiv.org/pdf/2006.14171 (PDF скачан, текст не разбирался; утверждения о нулевом градиенте и сравнении — из Gym-μRTS paper).
- **Пустые клетки / отсутствующие юниты**: GridNet — пустые клетки «маскируются из совместной политики», так что сложность не зависит от разрешения сетки.
- **Фиксированный максимум + паддинг**: Lux S3 (16 юнитов; `[max_units,3]`), PufferLib — «приводит переменную популяцию к фиксированному числу агентов и сохраняет порядок» (Neural MMO) (https://pufferai.github.io/build/html/rst/blog.html); Toad Brigade — паддинг карты до 32x32 с маской, чтобы не было утечки информации между разными размерами досок.
- **Hide-and-seek**: self-attention по сущностям, маскированная, «перестановочно-инвариантна и обобщается на переменное число сущностей» (https://ar5iv.labs.arxiv.org/html/1909.07528).
- **Конфликты действий** (два юнита идут в одну клетку): Toad — сэмплирование без возвращения + rule-based разрешение по вероятностям; Halite 4-е место — одновременное назначение (linear sum assignment). Нужен хук `ActionResolver` после сэмплирования [вывод].

#### A.4. Credit assignment

- Все победившие RL-решения юнит-команд используют **одну командную награду и один командный value** (Toad: critic скаляр в [-1,1]; Frog Parade: ±1 по матчу; Gym-μRTS: shaped reward на игрока; GridNet: централизованный value). Per-unit value в найденных первоисточниках не использовался [вывод: низкий приоритет].
- **Team spirit (OpenAI Five)**: r_i=(1−τ)·ρ_i+τ·ρ̄, τ: 0.3→0.8 (rerun) / 0.3→1.0 (OpenAI Five): ранний индивидуальный сигнал, затем полностью командный. https://ar5iv.labs.arxiv.org/html/1912.06680
- **Value-функции**: AlphaStar — value принимает наблюдения оппонента (уменьшает дисперсию); hide-and-seek — «омнисциентный value function критична»: с маскированным value эксперимент не дошёл до стадии 4 за отведённое время; MAPPO — agent-specific global state (AS) и feature-pruned (FP).
- **UPGO** (AlphaStar; Toad Brigade) и **TD(λ)** (Toad Brigade, AlphaStar) — рабочие инструменты для ударной разреженной награды; HandyRL поддерживает value/policy-target MC, TD, VTRACE, UPGO как конфиг (https://raw.githubusercontent.com/DeNA/HandyRL/master/docs/parameters.md).
- **HAPPO** (Kuba et al. 2021, arXiv 2109.11251): доказывает, что parameter sharing для **гетерогенных** агентов может быть экспоненциально хуже; для однородных юнитов — нормально. https://ar5iv.labs.arxiv.org/html/2109.11251

#### A.5. Decentralized (общая политика на юнит) vs «командир»

- **Командир на всю карту** (GridNet/ResNet) — Lux S1/S3, Gym-μRTS, AlphaStar: лучший учёт координации, дороже на вход (вся карта), но не зависит от числа юнитов.
- **Общая политика на юнит** (IPPO/MAPPO, hide-and-seek, Neural MMO, Pommerman Skynet): простая, масштабируется по числу юнитов, но координация только через критик/наблюдения.
- GridNet обосновывает гибрид: общие свёрточные веса + большое receptive field = «коммуникация» без явного канала.
- Pommerman: коммуникация — 2 слова из словаря 8 между напарниками (среда поддерживает), но Skynet (2-е место среди learning-агентов на NeurIPS 2018) играл **без явной координации** двумя идентичными сетями. https://ar5iv.labs.arxiv.org/html/1809.07124 , https://ar5iv.labs.arxiv.org/html/1905.01360

#### A.6. Аугментации для карт

- **AlphaGo Zero**: случайные повороты/отражения (8-кратная симметрия Go) при обучении и в MCTS; **AlphaZero в шахматах/сёги аугментации не использует** (правила асимметричны). https://ar5iv.labs.arxiv.org/html/1712.01815
- **Test-time augmentation**: Toad Brigade — 180° поворот; Frog Parade — две диагональные рефлексии + 180° (с усреднением политики); требует **перестановки каналов направлений** в действиях [вывод].
- **Канонизация перспективы** (зеркалить карту к игроку 0) — все Lux S3 RL-команды (вторичный источник [не подтверждено]); 1 нейросеть вместо двух для симметричных стартов.
- **Тороидальные карты** (Hungry Geese): HandyRL GeeseNet использует `TorusConv2d` (wrap-around padding). https://raw.githubusercontent.com/DeNA/HandyRL/master/handyrl/envs/kaggle/hungry_geese.py
- **Осторожно с аугментациями в PPO**: DrAC (arXiv 2006.12862) показывает, что наивная аугментация ломает importance ratio π(a|f(s))/π_old(a|s); решение — аугментированные данные только в регуляризаторах (KL между политиками на s и f(s), равенство value). https://ar5iv.labs.arxiv.org/html/2006.12862 [вывод: для Colosseum — аугментировать через DrAC-регуляризацию или каноникализацию, а не подмешивать в ratio].

### B. Соло-игры

Что заменяет self-play и рейтинги:
1. **Ranked Reward (R2)** (Laterre et al., arXiv 1807.01672): self-play для одиночных игр — результат агента ранжируется относительно буфера его собственных недавних результатов (буфер 250 эпизодов), награда бинаризуется в ±1 по порогу перцентиля α (тестировали 50/75/90; 75% лучше на малых задачах, 50% на больших); использовалась с MCTS (300 симуляций на ход), пермутационно-инвариантная сеть, Adam, batch 32, 50 шагов градиента на итерацию; V100, до 48 часов на эксперимент; +6% к Gurobi, до +15% на крупных 2D/3D bin packing. https://ar5iv.labs.arxiv.org/html/1807.01672
2. **Curriculum по сидам/уровням — PLR** (Jiang et al., arXiv 2010.03934): приоритет уровней по среднему |GAE| (аппроксимация L1 value-loss) + staleness; Procgen: 128% нормированного test return против 100% (176% в комбинации с UCB-DrAC). Для соло-соревнований с процедурной генерацией карт. https://ar5iv.labs.arxiv.org/html/2010.03934
3. **Нормировка награды / value**: MAPPO — value normalization по бегущим оценкам (https://ar5iv.labs.arxiv.org/html/2103.01955); Melting Pot — min-max нормировка score между случайным агентом и эталонными «exploiter»-ами (https://ar5iv.labs.arxiv.org/html/2107.06857); PopArt — [не подтверждено в этой сессии, из памяти]. Toad Brigade: critic в [-1,1] (ограниченная награда).
4. **Статистика оценки**: Agarwal et al. (rliable, arXiv 2108.13264): мало прогонов + точечные оценки вводят в заблуждение; использовать **IQM, performance profiles, optimality gap, probability of improvement, stratified bootstrap CI**. https://ar5iv.labs.arxiv.org/html/2108.13264
5. **Сравнение с прошлыми версиями**: Frog Parade мониторил «winrate против предыдущей лучшей модели»; для соло — [предложение] парные сиды (одни и те же seed-ы для версий k и k-1), доля сидов, где новая версия лучше, и разница перцентилей (p10/p50/p90) со bootstrap-CI.
6. **Соло-eval Neural MMO** (одиночные политики): PvE — 32 эпизода на 4 seed-картах (в PvP-этапе — 1 800 эпизодов на 256 отложенных картах). https://arxiv.org/html/2508.12524v1
7. **Нечто по «чистому» соло в Kaggle** — Halite I–III и подобные оптимизационные игры в основном выигрывались поиском/правилами (по обзору nagiss: Halite IV rule-based); RL-пайплайн для таких игр должен уметь использовать scripted baseline как референс [вывод].

Вывод для фреймворка [вывод]: для N=1 PFSP и «рейтинг» вырождаются; нужен `SoloMatchmaker` (сид-сэмплер + PLR) и `SoloEvaluator` (распределение score на фиксированном наборе сидов, IQM/перцентили/CI, парное сравнение версий, сравнение со scripted-референсом).

### C. 1v1: пошаговые и с одновременными ходами

#### C.1. Построение переходов

- **PettingZoo AEC**: агенты ходят по очереди, `agent_iter()`, `last()` возвращает (obs, reward, termination, truncation, info) для ходящего; мёртвые агенты исключаются из `agents` через dead-step convention (`_was_dead_step`); авторы аргументируют AEC против POSG-API **атрибуцией награды** (сообщают о 22% улучшении производительности в Pursuit от корректного учёта источников награды) и устранением race condition. https://ar5iv.labs.arxiv.org/html/2009.14471 ; docs через Context7: https://github.com/farama-foundation/pettingzoo/blob/main/docs/content/basic_usage.md . Награды копятся в `_cumulative_rewards` (по Context7-сниппету dead-step).
- **PettingZoo Parallel**: `step(actions: dict)` → (obs, rewards, terminations, truncations, infos) dict-ы по агентам; `possible_agents` неизменен, `agents` сокращается; при конце эпизода `self.agents=[]`. Context7: https://github.com/farama-foundation/pettingzoo/blob/main/docs/api/parallel.md
- **RLCard**: пошаговую игру N игроков превращает в последовательность переходов «задержкой наблюдения»: игрок видит **следующее состояние только после того, как все остальные сходили**; в конце — payoffs всем. https://ar5iv.labs.arxiv.org/html/1910.04376
- **HandyRL** (`generation.py`): в turn-based режиме сохраняются только данные **действующих** игроков (`turn()`), у недействующих — только если включён флаг `observation`; в simultaneous — все игроки на каждом шаге; награды из `env.reward()`, финал из `env.outcome()`; returns считаются назад с γ. API среды: `players()`, `turn()` / `turns()`, `legal_actions(player)`, `play(action, player)` / `step(actions)`, `reward()`, `outcome()` (−1/0/1), `observation(player)`. https://raw.githubusercontent.com/DeNA/HandyRL/master/docs/custom_environment.md , https://raw.githubusercontent.com/DeNA/HandyRL/master/handyrl/generation.py
- **AlphaZero**: исход z назначается **с перспективы игрока, чей ход**; терминальные позиции −1/0/+1; самоигра всегда идёт последней сетью (нет gating); аугментации — нет (шахматы/сёги). https://ar5iv.labs.arxiv.org/html/1712.01815
- **OpenSpiel**: extensive-form представление, simultaneous-move через специальный «simultaneous player», chance-узлы, N-player, returns() по игрокам; алгоритмы: MCTS, CFR, PSRO, α-Rank, NFSP. https://ar5iv.labs.arxiv.org/html/1908.09453

Рекомендуемая схема [вывод, опирающаяся на указанное выше]:
- Для каждого слота строить **собственную цепочку переходов** на его *решениях*: (s_t, a_t) → награда, накопленная между его решениями (как `_cumulative_rewards` в PettingZoo) → s_next в **следующий свой ход** (как RLCard). Терминальный переход добавлять **каждому слоту**, включая не ходившего последним (он получает outcome со своей перспективы).
- Финальная награда: outcome_i ∈ [-1,1]; в 1v1 zero-sum outcome_B = −outcome_A; draws → 0.
- Для simultaneous: acting-set = все живые слоты, переход общий.

#### C.2. Первый игрок и ротация мест

- Единого цитируемого исследования про first-player advantage в самоигре найти не удалось (поиск вернул только AlphaZero). Практика:
  - **HandyRL evaluation**: для 2 игроков позиции чередуются систематически (первая половина игр — агент 0 ходит первым, вторая — агент 1), результаты ведутся отдельно по паттернам `-F`/`-S`; для N>2 — случайная перестановка `random.sample`. https://raw.githubusercontent.com/DeNA/HandyRL/master/handyrl/evaluation.py
  - nagiss: в ConnectX «у первого игрока выигрышная стратегия», поэтому в соревновании побеждал поиск. https://speakerdeck.com/nagiss/kagglesimiyuresiyonkonpenodong-xiang
  - Frog Parade/Lux S3: зеркалирование к перспективе игрока 0 (вторичный источник).
- [предложение] Фреймворк: матч задаёт `seat_permutation`; matchmaker сэмплирует места равномерно (или балансированно по парам); результаты агрегируются **по местам** и в среднем; в eval показывать win-rate на каждом месте отдельно (разрыв = мера first-player advantage), рейтинг считать по усреднённому.

#### C.3. Когда MCTS (AlphaZero) вместо model-free

Факты:
- **ConnectX** — поиск (первый игрок выигрывает), **Pommerman 2018** — tree search выиграл 1-е и 3-е места (hakozakijunctions, dypm), «learning»-агенты 4–5; в Pommerman окно решения ~100 мс, известен симулятор, важна точная безопасность. https://ar5iv.labs.arxiv.org/html/1902.10870 ; сводка результатов: NeurIPS 2018 — 1-й hakozakijunctions, 2-й eisenach, 3-й dypm; learning-категория — Navocado 1-й, skynet955 2-й (https://proceedings.mlr.press/v101/osogami19a.html ; https://rbcborealis.com/publications/skynet-top-deep-rl-agent-inaugural-pommerman-team-competition/ ).
- **Hungry Geese**: победитель — HandyRL; по сводке nagiss — линейные оценочные функции + ансамбли нейросетей/MCTS (вторичные пересказы, [не подтверждено в первоисточнике]); автор одного zenn-обзора (идентичность не установлена) в последний месяц добавил MCTS в стиле AlphaZero, ограничив паттерны появления еды, и пишет, что **офлайн-оценка модели почти невозможна из-за дисперсии** (https://zenn.dev/ktechb/articles/e2394bc27358c4).
- **Halite IV / Kore 2022** — rule-based/optimization; RL не был лидером.
- **AlphaZero** — ресурсоёмок: 5 000 TPU первого поколения для self-play + 64 TPU второго поколения для обучения, 700k шагов, batch 4096, 800 симуляций на ход (ar5iv 1712.01815).
- **Simultaneous moves MCTS**: Lisý et al. (NeurIPS 2013) — шаблон MCTS для одновременных ходов; при ε-Hannan-consistent методах выбора (Exp3, regret matching) сходится к приближённому равновесию Нэша; по сводке поиска empirical лучший — Decoupled UCT. https://ar5iv.arxiv.org/html/1310.8613 (аннотация; Decoupled UCT — по поисковой сводке [не подтверждено полностью]).
- **Gumbel AlphaZero/MuZero** (Danihelka 2022) — улучшение политики при малом числе симуляций: страница не загрузилась, [не подтверждено в этой сессии].

Эвристика выбора [вывод]: model-free (APPO) — когда игра большая/стохастическая/частично наблюдаемая и симулятор медленный или недоступен при инференсе (GRF, Lux, Dota, FFA с хаосом); MCTS/поиск — когда есть быстрый точный симулятор (clone/step), игра детерминированная малой ширины, ограничение по времени позволяет (ConnectX, Pommerman, 2-игроковые настольные); гибрид (сеть как prior+value для MCTS, Hungry Geese) — как «plug-in» поверх обученной политики.

### D. FFA на N игроков

#### D.1. Схемы наград

- **Попарный ранговый outcome (HandyRL Hungry Geese)**: для игрока i сравнивается с каждым из N−1 соперников: победа +1/(N−1), поражение −1/(N−1) (ничья 0) → outcome ∈ [-1,1] (−1 — последнее место, +1 — первое). https://raw.githubusercontent.com/DeNA/HandyRL/master/handyrl/envs/kaggle/hungry_geese.py
- Нормировка: zero-sum центрирование ранга; Kaggle-среды выдают каждому агенту reward, а ранжирование идёт по сравнению наград в эпизодах (https://raw.githubusercontent.com/Kaggle/kaggle-environments/master/README.md).
- Shaped: Skynet Pommerman Table 2 — 0.001 за исследование новой клетки, 0.5 за убийство врага, −0.5 за смерть напарника, 0.02 за kick, 0.01 за ammo/blast (https://ar5iv.labs.arxiv.org/html/1905.01360); Pommerman population-paper — адаптивный коэффициент отжига α=1−tanh(k·x), k=1.2, x — качество устранения врагов (https://ar5iv.labs.arxiv.org/html/2407.00662).
- Neural MMO: награды — через **task-условные** предикаты (1297 train задач, 63 eval), reward shaping доминировал в топ-решениях; «reward design важнее архитектуры» (https://arxiv.org/html/2508.12524v1).

#### D.2. Подбор N−1 оппонентов (обобщение PFSP)

Факты:
- **AlphaStar PFSP**: оппонент B из множества кандидатов 𝒞 сэмплируется с вероятностью f(P[A бьёт B]) / Σ f(P[A бьёт C]); f_hard(x)=(1−x)^p — сосредоточиться на самых сильных; f_var(x)=x(1−x) — на соразмерных (для main exploiters и «застрявших» main-агентов); main agents: **35% self-play, 50% PFSP против всех прошлых игроков лиги, 15% PFSP против «забытых» main-игроков и прошлых main exploiters**; League exploiters — PFSP, с вероятностью 25% сбрасываются на supervised; main exploiters играют против main-агентов (если P(win)<20% — PFSP с f_var), сбрасываются. Nature PDF, Methods.
- **OpenAI Five**: 80% игр против последней версии, 20% против прошлых; для прошлых — softmax по quality score q_i, p_i ∝ e^{q_i}, обновление q_i ← q_i − η/(N·p_i), η=0.01. https://ar5iv.labs.arxiv.org/html/1912.06680
- **FTW**: population 30; **stochastic matchmaking** по Elo так, чтобы исход был достаточно неопределён; компаньоны и соперники сэмплируются из живой популяции; PBT (lr, KL weight, timescale τ, entropy cost): агент с вероятностью победы <70% копирует лучшего с возмущением ±20%; 1920 арен; V-trace. https://ar5iv.labs.arxiv.org/html/1807.01281
- **Pommerman population-based self-play** (arXiv 2407.00662): 8 агентов (3 rule-based + 5 learning); матчмейкинг — softmax по ожидаемым win-rate на Elo; агенты с win-rate <45% заменяются; curriculum 3 фазы (static → moving without bombs → bomb-placing) с порогом 55%; сеть: 4 conv 3x3 → 256 → +scalars → LSTM (10 шагов) → policy/value (6 действий); результат: 98.85% против baseline, 96.23% против Skynet (по сводке). https://ar5iv.labs.arxiv.org/html/2407.00662
- **Sample Factory**: `--num_policies=N`, `agent_policy_mapping`, при `pbt_mix_policies_in_one_env=True` политики на агентах **периодически перемешиваются**, чтобы агенты не встречали одних и тех же оппонентов. https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/docs/07-advanced-topics/multi-policy-training.md
- **Hungry Geese (автор zenn)**: «простая смесь прошлых агентов» вместо лиги AlphaStar; оффлайн-оценка почти невозможна. 

[предложение] Обобщение PFSP на N−1 слотов:
1. **iid-схема**: каждый из N−1 слотов независимо сэмплируется из PFSP-распределения над пулом (чекпоинты self, других агентов, скрипты); «win-rate» определяется попарно из FFA-результатов (P[ранг A лучше ранга B]), т.е. из тех же попарных исходов, что в HandyRL outcome.
2. **Table-схема**: один «сложный» оппонент по PFSP + остальные слоты — смесь (self-latest / прошлые версии / скрипты); сохраняет сигнал, но разнообразит стол.
3. **Quality-balanced**: выбирать столы, где предсказанное распределение рангов (OpenSkill `predict_rank`) максимально неопределённое (аналог FTW).
4. Доля «старых»/скриптовых слотов ≥ 20% (OpenAI Five: 20% прошлых).
5. Вес `collect` на слотах: собирать траектории только у обучаемых; в арене нескольких обучаемых — у всех (как уже в Colosseum).

#### D.3. Рейтинг для FFA

- **OpenSkill (Weng–Lin)**: 5 моделей — PlackettLuce (рекомендована), BradleyTerryFull/Part, ThurstoneMostellerFull/Part; μ по умолчанию 25, σ 8.33, ordinal = μ−3σ; `model.rate([team1, team2, team3], ranks=[4,1,3,2])`, `scores=[...]`, ничьи (равные ранги), `weights`, `margin`; `predict_win`, `predict_draw`, `predict_rank`. https://openskill.me/en/stable/manual.html
- **Weng & Lin (JMLR 2011)**: k-командная игра рассматривается как несколько двухкомандных; аналитические правила обновления без численного интегрирования; точность на данных конкурентна с TrueSkill при меньших времени и объёме кода; модели Bradley–Terry, Thurstone–Mosteller, Plackett–Luce. https://www.jmlr.org/papers/volume12/weng11a/weng11a.pdf (разобран локально).
- **TrueSkill** (Herbrich et al.): Гауссово убеждение, моделирует ничьи, мульти-команда/мульти-игрок, вывод через message passing; OpenAI Five использовал TrueSkill с 83 эталонными агентами (от 0 — случайная игра до 254 — версия, победившая чемпионов мира) (https://ar5iv.labs.arxiv.org/html/1912.06680 ; https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/ ). TiKick также отчитывается TrueSkill 22.84±3.25 (https://arxiv.org/pdf/2110.04507).
- **Нетранзитивность**: Balduzzi et al. 2019 — любая игра раскладывается на транзитивную и циклическую части; Elo/средний win-rate не годятся в циклических играх; предлагаются Nash-based метрики, PSRO_N / PSRO_rN. https://ar5iv.labs.arxiv.org/html/1901.08106 ; α-Rank/α-PSRO — для общей суммы и многих игроков (Muller et al., arXiv 1909.12823): https://ar5iv.labs.arxiv.org/html/1909.12823 .
- **Kaggle** использует собственную Gaussian-рейтинговую схему для симуляций — **детали не подтверждены** (страницы недоступны).
- [предложение] Colosseum: основной рейтинг — OpenSkill PL по рангам (+ weights для командных слотов), Elo оставить для 1v1 и мониторинга; дополнительно: матрица попарных win-rate (уже есть), Nash-averaging/α-Rank как диагностика нетранзитивности; фиксированный **референс-пул** (OpenAI Five) для сопоставимости между запусками.

#### D.4. Эффекты мест и практика конкретных игр

- **Seat/стартовая позиция**: HandyRL — случайная перестановка при N>2. Для игр с неравными стартами — ротация, и усреднение.
- **Hungry Geese** (4 игрока, 7x11 тор): HandyRL GeeseNet — obs 17x7x11 (головы/хвосты/тела/прошлые головы по 4 + еда), вход TorusConv2d 17→32, **12 residual-блоков TorusConv2d 32→32 3x3 с batch norm**, policy-голова Linear(32→4), value-голова Linear(64→1) от конкатенации «голова + среднее по карте»; outcome — попарные ранги (см. D.1). 875 команд, 1 039 участников (https://aismiley.co.jp/ai_news/kaggle-hungry-geese-dena-quantum/ ); состав победившей команды DeNA + QUANTUM; обучение «в основном на машинах quantum». 1st place writeup: https://www.kaggle.com/c/hungry-geese/discussion/263279 (не прочитан). DQN-варианты (vanilla/Double/Dueling) на Hungry Geese сходились плохо из-за стохастики (arXiv 2109.01954).
- **Halite 4p / Kore**: rule-based (см. A.1); для RL-фреймворка это означает «скриптовые боты в пуле оппонентов и как baselines» (Halite-4 players: 4-е место — rules).
- **Lux S3**: 1v1, но best-of-5 на одной карте — нужна «мета»-память между матчами (внутри серии) [вывод: matchmaker должен уметь запускать серию матчей с общим state/seed между матчами].
- **Neural MMO** (v2.0): PettingZoo ParallelEnv, до 128 агентов, **команды по 8**; `GameState` в плоском тензорном формате; ~3000 agent-steps/ядро/с (250M+ agent steps в сутки); бейслайн CleanRL PPO + PufferLib, обучаемо за 8 A100-часов; соревнование NeurIPS 2023: PvE (32 эпизода × 4 карты) и PvP (топ-9 политик в общей среде, 128 агентов по 9 группам, 1 800 эпизодов на 256 картах); победители 25.21% и 24.88% выполнения задач против 6.39% бейслайна; обучение 15–22M шагов. https://ar5iv.labs.arxiv.org/html/2311.03736 ; https://arxiv.org/html/2508.12524v1
- **Neural MMO v1**: популяции агентов с **общей архитектурой и несвязанными весами между популяциями**; 16–128 агентов; популяции 1–8; агенты из больших популяций сильнее; наблюдение 15x15 тайлов; простая FC-архитектура. https://ar5iv.labs.arxiv.org/html/1903.00784

### E. Команды и асимметричные роли

#### E.1. Parameter sharing, role embedding, отдельные сети

| Решение | Подход |
|---|---|
| OpenAI Five | одна общая сеть на 5 героев; кто управляется — через часть observation (hero id) |
| Hide-and-seek | общие параметры политики (децентрализованное исполнение, централизованный value); отдельные параметры «тоже достигают всех шести стадий, но менее sample-efficient» |
| Diplomacy SearchBot | одна сеть, **60-мерный embedding державы** в decoder (7 асимметричных ролей) |
| AlphaStar | **отдельный агент на каждую расу** (3 main, 3 main exploiter, 6 league exploiter) |
| Neural MMO v1 | популяции с общей архитектурой и разными весами |
| TiKick (GRF 10 игроков) | один набор параметров на всех (центрально обучаемый), 4x256 FC + GRU, вход 268 |
| Skynet (Pommerman 2v2) | 2 идентичные сети |
| MAPPO | sharing для однородных агентов лучше; HAPPO — для гетерогенных sharing может быть экспоненциально хуже |

Источники: https://ar5iv.labs.arxiv.org/html/1912.06680 ; https://ar5iv.labs.arxiv.org/html/1909.07528 ; https://ar5iv.labs.arxiv.org/html/2010.02923 ; Nature (AlphaStar); https://ar5iv.labs.arxiv.org/html/1903.00784 ; https://arxiv.org/pdf/2110.04507 ; https://ar5iv.labs.arxiv.org/html/1905.01360 ; https://ar5iv.labs.arxiv.org/html/2103.01955 ; https://ar5iv.labs.arxiv.org/html/2109.11251

Рекомендация [вывод]: по умолчанию **одна сеть на роль** (отдельный `network_id` на роль, как уже есть группировка инференса по `(agent_id, network_id)`), с опцией **shared trunk + role/slot embedding** для ролей с одинаковыми пространствами; различающиеся obs/action spaces → отдельные сети обязательно.

#### E.2. Матчмейкинг с ролями, PSRO для асимметричных игр

- **Теория**: Tuyls et al. 2018 — двухигровая асимметричная игра раскладывается на симметричные «контрпарты»: если (x,y) — Нэш в асимметричной игре (A,B), то y — Нэш в симметричной игре с матрицей A, x — в игре с B^T (для полной опоры; Th. 2 — при совпадении опор) — то есть анализ идёт по **популяциям ролей**. https://ar5iv.labs.arxiv.org/html/1711.05074
- **PSRO**: для симметричных игр достаточно одной популяции; игры с ролями — «multi-population» (отдельные популяции на каждого игрока); α-PSRO даёт сходимость в single-population к sink-SCC, для multi-population нужен novelty-bound oracle (Muller et al.). По ссылке поиска на survey: https://arxiv.org/html/2403.02227v2 ; https://ar5iv.labs.arxiv.org/html/1909.12823
- **AlphaStar league**: PFSP по ролям (расам) + exploiters, центральный координатор поддерживает payoff-матрицу, сэмплит матчи по запросу, сбрасывает exploiters.
- **Melting Pot** (arXiv 2107.06857): оценка через **focal vs background population**: режимы Resident (фокальных больше), Visitor (фоновых больше), Universalization (все фокальные); основная метрика — per-capita return фокальной популяции, нормировка min-max между random-агентом и exploiter-ами; фоновые «боты» тренируются RL по заданным reward-функциям (~85 сценариев). https://ar5iv.labs.arxiv.org/html/2107.06857 — полезная модель для **слотов фокальных/фоновых**.
- [предложение] **Role-aware PFSP**: статистика win-rate ведётся по ключу (agent_i, role_i) против (agent_j, role_j); для каждой роли — своя популяция/пул; матч собирается как «обучаемый (A, роль r) против оппонентов по PFSP над пулами **противоположных** ролей» + сбалансированная ротация ролей (счётчики, чтобы каждая пара (агент, роль) получала равные доли); рейтинг: skill на (agent, role) + агрегированный по ролям. Асимметричные игры типа hide-and-seek обучают обе стороны одновременно (hiders и seekers — самоигровая «автокурикулярная» пара), что требует *одновременно* обучаемых слотов на разных ролях — это уже есть в Colosseum (арена из нескольких обучаемых).

#### E.3. Примеры

- **OpenAI hide-and-seek** (arXiv 1909.07528): сущностная политика — embedding слои с общими весами по типу, **masked residual self-attention (4 головы по 32)**, пулинг, LSTM 256, embedding 128, MLP 256; ~1.6M параметров; награда zero-sum командная (hiders +1 если все скрыты, −1 если кого-то видят; seekers наоборот), −10 за выход из 18-метровой зоны; PPO + GAE (λ=0.95, γ=0.998), lr 3e-4, ε=0.2, batch 64 000 цепочек по 10 шагов, буфер 320 000, 60 SGD-подшагов; **132.3M эпизодов (31.7B кадров) за 34 часа** до стадии ramp-defense; омнисциентный value критичен; 6 эмерджентных стадий. Метрики Elo/TrueSkill слабо информативны о *почему* растёт качество (авторы предлагают набор «intelligence test» задач). https://ar5iv.labs.arxiv.org/html/1909.07528
- **Pommerman (team)**: 2v2, 11x11, коммуникация 2 слова из 8, 6 действий (https://ar5iv.labs.arxiv.org/html/1809.07124); Skynet: 4 conv-слоя 3x3 с 64 фильтрами [формулировка в извлечении искажена: «44 layers»/«33×33» — вероятно, 4 слоя, 64 ядра 3x3; [не подтверждено]], 14 входных плоскостей 11x11, PPO, 120 игр за итерацию от 12 акторов, «retrospective board» вместо LSTM, ActionFilter ≈3 мс; результаты: ~70% побед против Static и ~20% против SmartRandomNoBomb (много ничьих). Победители — tree search с pessimistic scenarios; в dypm учитываются только собственные 6 действий, напарник — в детерминированном сценарии.
- **GRF**: Kaggle-формат — управляется **один игрок** (держатель мяча или ближайший защитник); WeKick (1-е место 2020): imitation learning + multi-head value trick + распределённая league training, **не расширяется на много игроков**; HandyRL — 5-е место из 1138 команд (https://speakerdeck.com/ikki407/...); **TiKick** (arXiv 2110.04507): управление **всеми 10 полевыми игроками** с общими параметрами из данных одиночного WeKick (21 947 эпизодов self-play по 3 000 шагов): вход 268 (115 + относительные позы + offside-флаги), 4x256 FC + 1 GRU, lr 1e-4 (Adam), 64 параллельных env, loss — α-balanced BC + minimization «build-in» действия + buffer ranking + advantage-weighted regression; 94.4% побед против встроенного ИИ (500 игр), разница голов +3.096, TrueSkill 22.84±3.25; пишут, что наивное управление всеми игроками BC-моделью ломается: в одиночном режиме нужно управлять ближайшим к мячу, а в мульти — дальним игроком с постоянным ракурсом. Среда: 19 дискретных действий, 115-мерный float-вектор, SMM 4x72x96, шаг ~140M шагов/сутки. https://arxiv.org/pdf/2110.04507 ; https://ar5iv.labs.arxiv.org/html/1907.11180
- **Dota 5v5**: одна политика, hero id во входе, team spirit τ→1.0, 80/20 оппоненты, 770±50 PFlops/s·days (по сводке; ~500–1440 GPU для forward, ~80 000–172 800 CPU для rollouts, батчи ≈1–3M шагов, до 1 536 optimizer GPU, GAE λ=0.95, clip 0.2) https://ar5iv.labs.arxiv.org/html/1912.06680
- **Capture the Flag (FTW)**: 2v2, популяция 30, внутренние награды (обучаемое преобразование 13 игровых событий), двухтемповый RNN (быстрое и медленное ядро), PBT. Агенты внутри команды — независимые, децентрализованное управление. https://ar5iv.labs.arxiv.org/html/1807.01281
- **Neural MMO teams**: команда из 8; общий API PettingZoo Parallel (см. D.4).

#### E.4. Централизованный критик vs независимые, коммуникация

- MAPPO: IPPO ≈ MAPPO на простых задачах, но «явное преимущество MAPPO растёт с числом агентов» (Hanabi, 5 игроков); критик на «agent-specific global state» (глобальное состояние + собственные наблюдения) и feature-pruned; value normalization. 2202.00082: при правильно подобранном clip IPPO и MAPPO сопоставимы — **trust region важнее архитектуры критика**.
- Hide-and-seek: омнисциентный critic критичен (A.4).
- Коммуникация: GridNet — через receptive field; Pommerman — словарь из 8 слов (среда); Skynet не использовал; TiKick — без явных каналов. [вывод]: коммуникация как часть action space (дискретные токены) — nice-to-have.

### F. Обобщение: минимальный универсальный интерфейс среды

#### F.1. Сравнение интерфейсов

| Интерфейс | Модель хода | Агенты/роли | Переменные агенты | Результат матча | Заметки |
|---|---|---|---|---|---|
| **PettingZoo AEC** | последовательный: `agent_iter`, `last`, `step` | `possible_agents`, `agents`; роли — через разные spaces на агента | мёртвые удаляются из `agents` (dead-step: `step(None)`) | нет «результата матча»; reward+termination на агента, `_cumulative_rewards` | лучшая атрибуция награды; нет команд как сущности |
| **PettingZoo Parallel** | одновременный: `step(dict)` | то же | `agents` сокращается; `possible_agents` фиксирован | как выше | NMMO и PufferLib строятся на нём |
| **OpenSpiel** | extensive-form; `current_player` / simultaneous / chance | игроки 0..N-1; симметрия не обязательна | число игроков фиксировано в игре | `returns()` по игрокам | есть MCTS/CFR/PSRO/α-Rank, но API игры, а не RL-среды с батчами |
| **Gymnasium** | single-agent | один | — | — | для multi-agent не предназначен [не подтверждено: из памяти] |
| **RLlib MultiAgentEnv** | dict по agent_id; mix turn/simultaneous | **policy_mapping_fn(agent_id, episode)**: N агентов → M политик | агенты появляются/исчезают | `"__all__"` для конца | Context7: https://github.com/ray-project/ray/blob/master/doc/source/rllib/multi-agent-envs.rst |
| **Melting Pot** | одновременный DM-Env, список игроков [не подтверждено, из памяти] | focal/background популяции, сценарии | фиксировано число игроков (до 8 на substrate) | per-capita focal return | модель фокальных vs фоновых |
| **Neural MMO 2.0** | PettingZoo Parallel | до 128 агентов, команды по 8, task-conditional | спавн/смерть динамические | task completion | `GameState` + `Predicates/Tasks` |
| **Kaggle environments** | `agent(observation, configuration)`; `make/run/step` | N агентов; статусы ACTIVE/INACTIVE/DONE | INACTIVE/DONE | reward на агента, ранжирование по рейтингу из эпизодов | агент — функция, путь, URL, `"random"` |
| **Lux S1/S2/S3** | одновременный | 2 команды; **команда = много юнитов** | юниты рождаются/гибнут | win/loss + match points | S3: `[max_units,3]`, best-of-5, JAX |
| **HandyRL Env** | `turn()` / `turns()`, `play` / `step` | `players()` | через `legal_actions` | `outcome()` −1/0/1 или вектор | наиболее близок к Colosseum |
| **PufferLib** | эмуляция PettingZoo → фикс. число агентов, плоские obs | паддинг и сортировка | да (паддинг) | — | https://pufferai.github.io/build/html/rst/blog.html |

Источники: PettingZoo/RLlib — Context7; Kaggle env — https://raw.githubusercontent.com/Kaggle/kaggle-environments/master/README.md ; Lux — https://github.com/Lux-AI-Challenge/Lux-Design-S3 ; HandyRL — docs выше.

#### F.2. Предлагаемый минимальный универсальный интерфейс [предложение]

Принципы: (1) отделить **участника матча (slot)** от **управляемых сущностей (entities)**; (2) acting-set явно в каждом шаге; (3) результат матча — отдельный объект; (4) всё, что нужно только при обучении (глобальное состояние, наблюдения оппонентов), — необязательные поля.

```python
# Схема (псевдокод интерфейса)
class RoleSpec:        # одна на роль; роли могут иметь РАЗНЫЕ пространства
    role_id: str
    obs_spec: ObsSpec         # global:[...] , spatial:[C,H,W], entities:[E_max,F] + entity_mask
    act_spec: ActSpec         # per_slot | per_entity[E_max, A] | per_cell[A,H,W]
    min_slots, max_slots
    team_id_policy: ...

class EnvSpec:
    roles: dict[str, RoleSpec]
    num_slots: int | range
    slot_roles: list[str]           # роль каждого слота
    slot_teams: list[int]           # команда каждого слота
    turn_model: Literal["simultaneous","sequential","mixed"]
    supports_clone: bool            # для MCTS/поиска
    has_global_state: bool          # для централизованного критика

reset(seed, options) -> StepResult
step(actions: dict[slot_id, Action]) -> StepResult

class StepResult:
    obs: dict[slot_id, Obs]                 # только для acting slots + (опционально) наблюдатели
    acting: set[slot_id]                    # acting mask: кто ходит СЕЙЧАС
    masks: dict[slot_id, ActionMask]        # per-entity/per-cell/per-action
    entity_ids / entity_mask                # переменное число юнитов
    rewards: dict[slot_id, float]           # dense, накапливаются между решениями слота
    done: bool                              # по матчу
    alive: dict[slot_id, bool]              # слот выбыл (FFA)
    global_state: Optional[Tensor]          # train-only (критик)
    info: dict

class MatchOutcome:
    scores: dict[slot_id, float]            # сырой score (в т.ч. соло)
    ranks: dict[slot_id, int]               # 1 = лучший, ничьи = равные ранги
    outcome: dict[slot_id, float]           # [-1,1]: пары (±1/(N-1) как в HandyRL) или zero-sum
    team_of: dict[slot_id, int]             # командная агрегация
    # WDL = производная (sign(outcome)), не хранимая
```

Соответствия: PettingZoo Parallel (`dict` + `agents`), HandyRL (`turn/turns`, `outcome`), RLCard (отложенное next_state по слоту), Kaggle (reward на агента → ранг), Lux (команда → юниты), Melting Pot (focal/background).

#### F.3. Слот: обучаемая политика / замороженный чекпоинт / скрипт

[предложение; прецеденты: Kaggle agent = function/path/URL/«random»; RLlib `policy_mapping_fn`; Sample Factory `agent_policy_mapping` + periodic resample; HandyRL `agent_map = {env.players()[p]: agents[ai]}` с RandomAgent/RuleBasedAgent/NetworkAgent; Melting Pot focal/background.]

```python
class SlotAssignment:
    slot_id: int
    role: str; team: int
    controller: Union[
        Trainable(agent_id, network_id),                  # собирает траектории, свой learner
        Frozen(checkpoint_ref, network_id, sampling="greedy|sample|temp"),
        Scripted(bot_ref, params),                        # rule-based / поиск, без нейросети
        Search(policy_ref, simulator, sims, ...),         # MCTS поверх сети (опционально)
        External(callable | path | url),                  # kaggle-like agent
        Random(),
    ]
    collect: bool                    # писать ли траектории слота
    weight_version: Optional[int]    # для frozen/lag-метрик
    seat: Optional[int]              # ротация мест
```

`MatchConfig = {env_id, env_config, seed, slots: list[SlotAssignment], series: n_matches_in_series}`; `MatchResult = {MatchOutcome, per-controller stats (по agent_id, role), episode_length, per-slot diagnostics}`. Для текущего Colosseum (по CLAUDE.md): `PlayerSlot`/`MatchConfig`/`MatchResult`, `slot_agent_map[env_idx][player_idx]` — естественная база; требуется добавить **role/team**, **entity-уровень**, **acting_mask**, **rank/outcome**, **controller-kind**.

---

## 3. Таблица «тип игры → рекомендации»

| Тип игры | Сеть | Построение траекторий | Награда | Матчмейкинг | Рейтинг | Eval-метрики |
|---|---|---|---|---|---|---|
| **1. Соло** | MLP/CNN/ResNet (+LSTM при частичной наблюдаемости); для комбинаторных — перм.-инвариантная; Ranked Reward + MCTS при наличии симулятора | чанки T шагов, эпизод = игра; seed-сэмплер (PLR) | score (нормированный), либо R2 (±1 по перцентилю буфера 250 эпизодов, α=75%) | нет оппонентов: сэмплер сидов/уровней + curriculum | распределение score на фиксированных сидах (перцентили p10/50/90, IQM) | IQM + bootstrap CI (rliable), парные сиды vs предыдущая версия, vs scripted baseline |
| **2. 1v1 пошаговая** | CNN/ResNet(+attention) policy+value; MCTS при быстром точном симуляторе | переходы **на решениях слота**; next_state = следующий свой ход; терминал каждому слоту с его перспективы | outcome ±1/0 (zero-sum); shaped опционально | self-play: latest + прошлые (OpenAI Five 80/20), PFSP f_hard/f_var; **ротация мест** | Elo / OpenSkill; фикс. референс-пул | win-rate по местам (F/S), CI Уилсона, первый-игрок-gap, Elo-кривая |
| **2b. 1v1 одновременная** | тот же; поиск: Decoupled UCT / Exp3 | общий шаг, оба acting | outcome ±1 | как выше | как выше | как выше + exploitability-прокси (лига exploiters) |
| **3. Команда юнитов (один бот)** | GridNet/ResNet/U-Net (Toad: 24 бл. 128 ch 5x5 ~20M; Frog: 8 бл. 256 ~10M) или entity-attention; per-entity head + маски | чанки T=128–512; выход на юнит/клетку + `entity_mask`; hidden state LSTM по команде | ±1 в конце (sparse) + shaped на старте (20M шагов), team value | self-play + teacher-KL (Toad) / PFSP; scripted в пуле | Elo/OpenSkill | win-rate, длина эпизода, число юнитов, доля masked/клиппинга, %illegal |
| **4. Команда независимых ботов (2v2, GRF)** | shared params + slot/role embedding (TiKick 4x256+GRU; Skynet CNN) + централизованный критик | траектории на слот; team value над глобальным состоянием | командная + team spirit τ (0.3→1.0) | популяция (PBT), союзники и оппоненты из пула; Elo-softmax | TrueSkill/OpenSkill multi-team | win-rate, goal diff (GRF), TrueSkill с ± |
| **5a. FFA N игроков** | CNN/ResNet (тор: TorusConv; Hungry Geese 12 бл. 32 ch) + value от head+pool | на каждого слота; терминал всем | попарный ранг: ±1/(N−1) → [-1,1] | PFSP над столами (iid / table / quality-balanced), смесь self/прошлые/скрипты; ротация мест | OpenSkill PL по рангам / TrueSkill | средний ранг, % побед/топ-2, попарные win-rate, рейтинг с σ |
| **5b. 1 vs N / асимметричные роли** | отдельная сеть на роль (AlphaStar) или shared + role embedding (Diplomacy 60-d); критик с global state | отдельные цепочки по ролям | zero-sum командная (hide-and-seek ±1) | role-aware PFSP, баланс ролей, популяции по ролям; PSRO/α-Rank для нетранзитивности | рейтинг на (agent, role) + агрегат | win-rate по роли, баланс ролей, Nash-averaging/α-Rank |

---

## 4. Рекомендации для фреймворка (по приоритету)

### Must-have

1. **Слот/роль/команда** в `MatchConfig`: `role`, `team`, `controller` (Trainable/Frozen/Scripted/External/Random), `collect`, `seat`. (Опора: HandyRL agent_map, RLlib policy_mapping_fn, Kaggle agent forms, Melting Pot focal/background.)
2. **Acting mask + поллотные цепочки переходов**: отложенный next_state на следующем решении слота (RLCard), накопление награды между решениями (PettingZoo `_cumulative_rewards`), терминал для всех слотов, поддержка turn-based/simultaneous/mixed; обработка выбывших (FFA).
3. **MatchOutcome**: `scores` + `ranks` + `outcome ∈ [-1,1]` (попарный ранг, HandyRL) + производные WDL; командная агрегация; `seat_permutation` в MatchConfig и в eval (first/second для 2p, перестановки для N>2 — как HandyRL).
4. **Entity/grid-структурные наблюдения и действия**: `entity_mask`, `entity_ids`, per-entity `[E_max, A]` и per-cell `[A,H,W]` режимы (расширение ActionSpec/CompositeDist), маски на уровне юнита; хук `ActionResolver` для конфликтов.
5. **Редукция лоссов по юнитам**: `logp_reduction`, `entropy_reduction`, режим per-unit ratio, ε(N) (0.1/0.05/0.03 для 5/10/27 агентов — подтверждено только в этих пределах), V-trace на уровне юнита или гибрид «V-trace для policy + TD(λ)/GAE для value» (AlphaStar); логирование clipfrac/KL/|log-ratio|.
6. **Централизованный критик**: поле `global_state` (+ наблюдения оппонентов) только на обучении (AlphaStar, hide-and-seek: критично; MAPPO).
7. **Скриптовые боты и внешние агенты как first-class** (Halite/Kore — rule-based лидеры, Pommerman — поиск): интерфейс `Scripted`/`External`, использование в пуле оппонентов и как референс-baseline в eval.
8. **Eval-ядро**: фиксированный референс-пул, CI (Уилсон), по-местам статистика, парные сиды, рейтинг OpenSkill (Plackett-Luce) с rank-ами и weights.

### Should

1. **Role-aware PFSP + балансировка ролей** и PFSP на множества оппонентов (iid/table/quality-balanced); f_hard/f_var, доли «self/PFSP/forgotten/exploiters» (AlphaStar 35/50/15).
2. **Value-инфраструктура**: multi-head value (WeKick, AlphaStar), team-spirit τ-отжиг (OpenAI Five), value normalization (MAPPO), UPGO/TD(λ) цели (AlphaStar, Toad Brigade, HandyRL).
3. **Соло-пакет**: Ranked Reward, PLR-сэмплер сидов, rliable-статистика.
4. **Канонизация перспективы + TTA (D4-группа с перестановкой каналов действий) + DrAC-регуляризатор** вместо подмешивания аугментаций в ratio.
5. **Teacher-KL / progressive scaling** (Toad: 8→16→24 блока; teacher KL) поверх существующего kickstart; BC из реплеев (Offline BC уже есть).
6. **Plug-in поиска**: опциональный интерфейс `supports_clone`/`clone_state`, MCTS/Decoupled UCT как «контроллер слота», сеть как prior/value (Hungry Geese, Pommerman, AlphaZero).
7. **Поддержка серий матчей** (Lux S3 best-of-5) в `MatchConfig`.
8. **PBT гиперпараметров** по образцу FTW (lr, entropy, KL weight; copy-with-perturbation ±20% при win prob < 70%).

### Nice-to-have

1. AlphaZero/MuZero-стиль learner (Tier 3), Gumbel-варианты [не подтверждено].
2. PSRO/α-PSRO мета-решатели и Nash-averaging для нетранзитивных игр (Balduzzi; Muller).
3. Коммуникационные каналы как часть action space (Pommerman 2 слова из 8).
4. Адаптеры: PettingZoo Parallel/AEC, Kaggle-environments, Lux S2/S3 (JAX), PufferLib-подобная эмуляция фиксированного числа агентов (паддинг+сортировка).
5. Per-unit value/counterfactual baseline (COMA-подобно) — в найденных победивших решениях не использовался.
6. Гетерогенный HAPPO-подобный последовательный апдейт для ролей с общим trunk (теория Kuba et al.: sharing для гетерогенных агентов может быть экспоненциально хуже).

### Открытые вопросы и ограничения исследования

- Не прочитаны Kaggle-writeups: Hungry Geese 1-е место (263279), Lux S3 Flat Neurons, Kore 2022 победитель, Halite IV победитель (ttvand); Lux S2 победитель (ryandy) без описания подхода в README.
- Не проверены: Gumbel AlphaZero, PopArt, детали рейтинга Kaggle, точный API Melting Pot.
- Числа Lux S3 из японского обзора не согласуются с writeup Frog Parade — использовать только данные самих авторов.
- ε(N) для сотен юнитов — экстраполяция, требует эксперимента.

---

## 5. Источники (URL)

Архитектуры/юниты:
- GridNet (Han et al. 2019): https://proceedings.mlr.press/v97/han19a/han19a.pdf , https://proceedings.mlr.press/v97/han19a.html
- Gym-μRTS (Huang et al. 2021): https://arxiv.org/pdf/2105.13807 , https://ar5iv.labs.arxiv.org/html/2105.13807
- CleanRL/gym-microrts `ppo_gridnet.py`: https://raw.githubusercontent.com/vwxyzjn/gym-microrts/master/experiments/ppo_gridnet.py
- Invalid action masking (Huang & Ontañón 2020): https://arxiv.org/pdf/2006.14171
- Toad Brigade (Lux S1): https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021 ; https://raw.githubusercontent.com/IsaiahPressman/Kaggle_Lux_AI_2021/main/README.md ; Kaggle: https://www.kaggle.com/c/lux-ai-2021/discussion/294993
- Frog Parade (Lux S3): https://github.com/IsaiahPressman/kaggle-lux-2024 ; https://raw.githubusercontent.com/IsaiahPressman/kaggle-lux-2024/main/write-up.md
- Lux S3 RL-сводка (вторичный): https://zenn.dev/kurupical/articles/61dbeedf89a29d
- Lux S3 дизайн: https://github.com/Lux-AI-Challenge/Lux-Design-S3 ; Lux S2: https://github.com/Lux-AI-Challenge/Lux-Design-S2 ; https://github.com/ryandy/Lux-S2-public ; https://github.com/RoboEden/Luxai-s2-Baseline
- Lux S2 wandb (sgoodfriend): https://wandb.ai/sgoodfriend/rl-algo-impls-benchmarks/reports/Lux-AI-Season-2-Training--Vmlldzo0MTc4NDQz
- Halite IV: https://github.com/0Zeta/HaliteIV-Bot ; https://github.com/ttvand/Halite
- Обзор Kaggle-симуляций (nagiss, DeNA): https://speakerdeck.com/nagiss/kagglesimiyuresiyonkonpenodong-xiang
- Lux S1 обзор (kenmatsu4): https://speakerdeck.com/kenmatsu4/lux-ai-season-2gashi-matutanode-season-1wozhen-rifan-ru
- AlphaStar (Nature): https://www.nature.com/articles/s41586-019-1724-z ; PDF: https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf
- OpenAI Five: https://ar5iv.labs.arxiv.org/html/1912.06680
- Hide-and-seek: https://ar5iv.labs.arxiv.org/html/1909.07528
- FTW / Capture the Flag: https://ar5iv.labs.arxiv.org/html/1807.01281

PPO/MARL:
- MAPPO: https://ar5iv.labs.arxiv.org/html/2103.01955
- HAPPO/HATRPO: https://ar5iv.labs.arxiv.org/html/2109.11251
- Trust Region Bounds for Decentralized PPO: https://ar5iv.labs.arxiv.org/html/2202.00082
- DISC: https://ar5iv.labs.arxiv.org/html/1905.02363
- DrAC: https://ar5iv.labs.arxiv.org/html/2006.12862
- PLR: https://ar5iv.labs.arxiv.org/html/2010.03934
- rliable: https://ar5iv.labs.arxiv.org/html/2108.13264
- Ranked Reward: https://ar5iv.labs.arxiv.org/html/1807.01672

Игры/среды:
- Hungry Geese (HandyRL env): https://raw.githubusercontent.com/DeNA/HandyRL/master/handyrl/envs/kaggle/hungry_geese.py ; Kaggle writeup: https://www.kaggle.com/c/hungry-geese/discussion/263279 ; DQN-исследование: https://arxiv.org/abs/2109.01954 ; zenn-обзор: https://zenn.dev/ktechb/articles/e2394bc27358c4 ; пресса: https://aismiley.co.jp/ai_news/kaggle-hungry-geese-dena-quantum/
- HandyRL: https://github.com/DeNA/HandyRL ; docs: https://raw.githubusercontent.com/DeNA/HandyRL/master/docs/parameters.md , https://raw.githubusercontent.com/DeNA/HandyRL/master/docs/custom_environment.md ; код: https://raw.githubusercontent.com/DeNA/HandyRL/master/handyrl/evaluation.py , https://raw.githubusercontent.com/DeNA/HandyRL/master/handyrl/generation.py ; презентация: https://speakerdeck.com/ikki407/ming-ri-karashi-jian-qiang-hua-xue-xi-deqiang-idui-zhan-gemuaiwozuo-rou-daredemoshi-seruzhuang-tai-womezasite-6a49ff3a-882f-4a1b-b475-50e1d16b93b3
- Pommerman: https://ar5iv.labs.arxiv.org/html/1809.07124 ; Skynet https://ar5iv.labs.arxiv.org/html/1905.01360 ; pessimistic tree search https://ar5iv.labs.arxiv.org/html/1902.10870 , https://proceedings.mlr.press/v101/osogami19a.html ; population self-play https://ar5iv.labs.arxiv.org/html/2407.00662
- Google Research Football: https://ar5iv.labs.arxiv.org/html/1907.11180 ; TiKick (WeKick): https://arxiv.org/pdf/2110.04507
- Neural MMO: v1 https://ar5iv.labs.arxiv.org/html/1903.00784 ; 2.0 https://ar5iv.labs.arxiv.org/html/2311.03736 ; результаты NeurIPS 2023 https://arxiv.org/html/2508.12524v1
- Melting Pot: https://ar5iv.labs.arxiv.org/html/2107.06857
- Diplomacy SearchBot: https://ar5iv.labs.arxiv.org/html/2010.02923
- Sample Factory multi-policy: https://raw.githubusercontent.com/alex-petrenko/sample-factory/master/docs/07-advanced-topics/multi-policy-training.md
- PufferLib: https://pufferai.github.io/build/html/rst/blog.html

Интерфейсы:
- PettingZoo paper: https://ar5iv.labs.arxiv.org/html/2009.14471 ; docs (Context7): https://github.com/farama-foundation/pettingzoo/blob/main/docs/api/parallel.md , https://github.com/farama-foundation/pettingzoo/blob/main/docs/content/basic_usage.md
- OpenSpiel: https://ar5iv.labs.arxiv.org/html/1908.09453
- RLCard: https://ar5iv.labs.arxiv.org/html/1910.04376
- RLlib multi-agent (Context7): https://github.com/ray-project/ray/blob/master/doc/source/rllib/multi-agent-envs.rst
- Kaggle environments: https://raw.githubusercontent.com/Kaggle/kaggle-environments/master/README.md

Рейтинги/теория игр:
- OpenSkill: https://openskill.me/en/stable/manual.html ; Weng & Lin 2011: https://www.jmlr.org/papers/volume12/weng11a/weng11a.pdf ; TrueSkill: https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/
- Open-ended learning in symmetric zero-sum games: https://ar5iv.labs.arxiv.org/html/1901.08106
- Symmetric decomposition of asymmetric games: https://ar5iv.labs.arxiv.org/html/1711.05074
- α-PSRO (Generalized Training Approach): https://ar5iv.labs.arxiv.org/html/1909.12823 ; PSRO survey: https://arxiv.org/html/2403.02227v2
- AlphaZero: https://ar5iv.labs.arxiv.org/html/1712.01815
- MCTS в одновременных ходах: https://ar5iv.arxiv.org/html/1310.8613

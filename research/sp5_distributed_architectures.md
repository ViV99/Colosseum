# SP5 «Distributed»: control plane и системная архитектура распределённых RL / self-play лиг

Дата исследования: 2026-10-17. Язык: русский (названия, URL, API, цитаты в оригинале).

Область: системная инженерия (hub / координатор, регистрация и обнаружение узлов, лизы и heartbeat, раздача матчей, хранение весов, отказоустойчивость, деплой). Алгоритмический уровень (GORILA, A3C, IMPALA, Ape-X, R2D2, SEED RL, Podracer, Sample Factory, PureJaxRL, PufferLib, Cleanba, V-trace) уже разобран в `research/01_distributed_rl_architectures.md`; здесь он **не повторяется**, только ссылки на него там, где нужен контраст. Про матчмейкинг как алгоритм (PFSP, PSRO, Elo) см. `research/02_competitive_rl_framework_research.md`, про отбор и логи матчей `research/sp4_selection_practice.md`.

Условные обозначения (те же, что в `research/sp4_selection_practice.md`):
- **[F]** — факт, прочитанный в первоисточнике (статья, код репозитория, официальная документация); URL указан.
- **[S]** — вторичный источник или пересказ (поисковая выдача, агрегатор, краткое изложение страницы). Проверять перед использованием.
- **[И]** — мой вывод или оценка (не из источника).
- «Не публично» — источник прочитан, и такой детали в нём нет; это не то же самое, что «в системе этого нет».

## 0. Что удалось и что не удалось прочитать

Прочитано целиком или по нужным разделам (PDF скачаны и разобраны `pdftotext`, код читался из репозиториев):
AlphaStar (Nature, текст Methods), OpenAI Five (arXiv 1912.06680, разделы 3.2, App. A, C, J, N), Dactyl/Rapid (1808.00177, §4.3), SEED RL, Sample Factory (статья и документация), Podracer (только фактура про Sebulba), Acme, Launchpad, Reverb, moolib (whitepaper + `broker.h`), TorchBeast (§5.2), TLeague (arXiv 2011.12895 целиком по системным разделам), TStarBot-X, MALib (через ar5iv, пересказом), DI-engine (`coordinator.py`, `base_league.py`, документация, DI-orchestrator `architecture.md`), DI-star (`league_api.py`, `league.py`, `actor_comm.py`, `coordinator.py`), Tencent HoK / «KaiWu» / JueWu (arXiv 2011.12692, 1912.09729, 2209.08483), Menger (блог Google Research), SRL (ICLR 2024), Ray RLlib и Ray Core (документация), PufferLib (статья), HandyRL (код `train.py` / `worker.py` / `connection.py`), Lux AI S1 победитель (README репозитория).

Не удалось или только частично:
- AlphaStar: Extended Data Fig. 6 (схема инфраструктуры) в текстовом виде отсутствует; детали хранения чекпойнтов, формата протокола, отказоустойчивости **не публичны**.
- OpenAI Five: исходники Rapid закрыты; как хранятся блобы весов, как устроены лизы и перезапуск, **не публично**; известно только то, что написано в статье (Redis-«controller», метаданные).
- MALib: статья читалась через краткое изложение ar5iv-версии (v1 arXiv); JMLR-версия отличается. Отказоустойчивость в тексте не обсуждается.
- Kaggle write-up страницы рендерятся JS-ом и не читаются через WebFetch; для Kaggle-победителей использованы README репозиториев и ранее собранный `research/sp4_selection_practice.md`.
- Поисковик плохо находит документацию (moolib, DI-engine, Sample Factory); на каждую такую систему идёт прямое чтение первоисточника.
- Не искал и не читал: Slurm-специфику (кроме упоминаний в статьях), Mesos/Borg-детали (Borg у Menger/Launchpad упомянут только в блогах).

---

## 1. Короткое резюме (15 пунктов)

1. **Control plane везде тонкий, data plane отделён.** Хаб/координатор/лига раздаёт задания и копит результаты, а траектории идут напрямую actor→learner (AlphaStar, TLeague, Rapid через Redis на каждый оптимизатор, KaiWu через memory pool). Хаб никогда не видит траекторий [F: AlphaStar, TLeague, Dactyl, JueWu — §2].
2. **Раздача матчей: pull, а не push.** Actor сам просит задание в начале эпизода (TLeague `LeagueMgr`, DI-star `actor_ask_for_job`, AlphaStar «samples new matches on request», HandyRL `'args'`, MALib «semi-passive»). Исключения: DI-engine-координатор назначает задачи зарегистрированным collectors, PolyBeast наоборот держит env-серверы пассивными, а learner сам «звонит» им.
3. **Динамическое добавление/удаление actors — это почти бесплатно, learners — почти нет.** Actors: Deployment/ReplicaSet в K8s (TLeague), «anytime and from anywhere» (HandyRL), moolib-пиры через broker. Learners в TLeague синхронны через Horovod, в moolib через Accumulator с лидером; меняется состав только с рестартом или ресинком группы.
4. **Единого «изящного» протокола членства нет.** Явные лизы/heartbeat встречаются редко: `moolib` Broker (ping с таймаутом, `syncId`, лидер), Ray (raylet↔GCS, `RAY_gcs_rpc_server_reconnect_timeout_s` 60 с), остальные полагаются на разрыв соединения, таймаут задачи или на K8s (TLeague, Launchpad). DI-engine-координатор **не имеет heartbeat**, только таймауты задач.
5. **Отказоустойчивость по принципу «actors — скот, learner — питомец, хаб — сохраняется на диск».** Launchpad формулирует это прямо [F]: actors stateless, learner чекпойнтится, планировщик (K8s) перезапускает. DI-star пишет `league_resume` раз в 3600 с, Rapid кладёт метаданные запуска в Redis-controller. Точной (exactly-once) учёности env-шагов нет нигде; «at-least-once + потеря незавершённого» — норма.
6. **Staleness — главный практический параметр.** OpenAI Five: цель 0–1 версий отставания, около 8 версий уже заметно замедляет обучение; решение — отправлять данные сегментами по 256 шагов и держать цикл коротким. Sample Factory: `--max_policy_lag` (по умолчанию 1000 версий) просто выбрасывает старый опыт. AlphaStar: actors обновляют веса каждые 10 с. Acme/Reverb: `RateLimiter` как backpressure.
7. **Веса: версионированный пул + реплики или кэши, а не «learner отдаёт каждому».** TLeague ModelPool (реплики, случайная реплика, в памяти), Menger (кэши между learner и actors: средняя латентность чтения −4×), HandyRL (прокси `Gather` на машину кэширует модели), SRL (NFS как parameter server). Это ровно пункт Colosseum «per-machine weight cache».
8. **Лига = сервис с персистентным состоянием и интерфейсом `get_job / report_result / register_learner / snapshot`.** У TLeague (`LeagueMgr` + `GameMgr` + `HyperMgr`), DI-star (`League`: `actor_ask_for_job`, `actor_send_result`, `register_learner`, `learner_send_train_info`), DI-engine (`BaseLeague.get_job_info/finish_job/judge_snapshot`), AlphaStar (central coordinator) интерфейс по сути один.
9. **Оценка — отдельная роль узла.** AlphaStar: «Additional evaluator workers (running on CPU)», OpenAI Five: выделенные машины TrueSkill играют текущую версию против 83 эталонных агентов (только против близких по силе, ±10 TrueSkill), HandyRL: доля заданий `eval_rate` идёт на оценку. Это прямое обоснование «pulse»/«tournament» как типов заданий хаба.
10. **Масштаб в публикациях огромен, а победители соревнований скромны.** Dota: до 1 536 GPU-оптимизаторов и 172 800 CPU; AlphaStar: 12 копий × (16 000 матчей + 128-ядерный TPU-learner); HoK: 320 GPU + 35 000 CPU на эксперимент. Но Lux AI S1 (1-е место) обучен на **одном личном ПК с двумя GPU** [F: README], а Hungry Geese 1-е место и GRF 5-е использовали HandyRL — GPU-learner + CPU-машины без Ray [F/S].
11. **Деплой: K8s-нативный (TLeague, DI-orchestrator) или «просто процессы + адрес хаба» (HandyRL, moolib, DI-star).** TLeague сознательно выбрал «yaml+jinja2+kubectl» вместо оператора и это «suffices» [F]. Launchpad отделяет программу от платформы запуска (launcher), но в открытой версии есть только локальные launchers [F].
12. **Безопасность в исследовательских системах почти отсутствует** (pickle поверх TCP в HandyRL и moolib, Flask без аутентификации в DI-star, собственный Python-протокол на ZeroMQ в TLeague). Для LAN-деплоя это терпимо, для интернета нет [F по коду; вывод об отсутствии auth — [И], см. §4].
13. **Центральный inference (SEED) окупается только на ускорителях с быстрой сетью.** В самой статье SEED на 176 окружениях и P100 даёт 0.63× от IMPALA; выигрыш приходит на TPU (до 2.4M FPS, 80×). У Colosseum worker'ы выводят политику на CPU; центральный inference — отдельная будущая опция (SP6).
14. **Чужие ошибки повторяются:** unbounded очереди (DI-star: deque 1000, HandyRL: Queue(256) — они хотя бы ограничены), pickle-совместимость версий, single learner как точка раздачи весов, потеря результатов при перегрузке (в Colosseum это уже было и вылечено батчами в FX1).
15. **Для Colosseum рекомендуется поэтапность:** сначала «hub = расширенный главный процесс» (всё, что уже есть: лига, рейтинги, match log, pulse, отчёты), с протоколом `Register / Heartbeat / LeaseJobs / ReportResults / AnnounceVersion / GetWeights`, спроектированным как на отдельный сервис; затем при необходимости вынести hub в отдельный процесс с SQLite-состоянием. K8s — слой поверх, не основа (§5).

---

## 2. По системам

### 2.1. AlphaStar (DeepMind, 2019)

Источник: [Nature 2019, Methods — текст unformatted](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf) (DOI [10.1038/s41586-019-1724-z](https://www.nature.com/articles/s41586-019-1724-z)).

**Control plane.**
- Один **central coordinator** «maintains an estimate of the payoff matrix, samples new matches on request, and resets main and league exploiters» [F]. «On request» = pull от actors/learners. Сброс эксплойтеров — решение координатора.
- «Additional evaluator workers (running on CPU) are used to supplement the payoff estimates» [F] — оценка отдельной ролью; платёжная матрица постоянно переоценивается («continuously re-evaluate the internal payoff estimation») [F].
- Лига: «regularly saving the parameters from our agents … as new players (which have fixed, frozen parameters)» [F]. Главные агенты добавляют копию каждые 2·10⁹ шагов; эксплойтеры добавляются по условию (winrate > 70 % против лиги) или по таймауту [F]. Всего создано «almost 900 distinct players» [S — число из пересказа, в прочитанном тексте Methods его не нашёл].
- **12 независимых копий actor-learner** (по расам: 1 main + 1 main exploiter + 2 league exploiters на расу) [F]. То есть «одновременных обучаемых агентов» — 12, у каждого свой learner.

**Data plane.** На каждого обучаемого агента: 16 000 одновременных матчей на preemptible CPU (~150 процессоров по 28 физических ядер), 16 actor-задач (каждая на TPU v3 с 8 ядрами) для inference, запросы шагов «batched together dynamically», learner — 128 ядер TPU, батч 512, ~50 000 шагов агента/с, данные «buffered in memory and replayed twice»; actors обновляют веса с learner **каждые 10 секунд** [F]. Итого inference централизован (на actor-задаче с TPU), а игры — на CPU: контраст с Colosseum, где inference на самих CPU-воркерах.

**Хранение, версии, отказоустойчивость, мониторинг, деплой.** Хранение чекпойнтов, формат RPC, обработка падений learner/координатора, перезапуски **не публичны** [F: в Methods нет]. Известно только: preemptible CPU для игр (подразумевается, что падения игр терпимы).

**Масштаб.** 32 TPU v3 на агента, 44 дня [F]. TStarBot-X сообщает, что скорости потребления и генерации данных в их конфигурации составляют лишь 1/30 и 1/73 от AlphaStar ([arXiv 2011.13729](https://arxiv.org/abs/2011.13729)) [F].

### 2.2. OpenAI Five и «Rapid» (OpenAI, 2018–2019)

Источники: [OpenAI Five, arXiv 1912.06680 / PDF](https://cdn.openai.com/dota-2.pdf) (§3.2, App. A, C, J, N); [Dactyl «Learning Dexterous In-Hand Manipulation», arXiv 1808.00177](https://arxiv.org/abs/1808.00177) (§4.3, описание Rapid).

**Роли машин** (4 типа, Fig. 2) [F]: Rollout (игра Dota на CPU), Forward Pass GPUs (отвечают rollout-ам на запросы действий батчами ≈ 60), Optimizer GPUs, Controller. «The entire system runs on our custom distributed training platform called Rapid, running on Google Cloud Platform» [F].

**Control plane / «controller».** Это **центральное Redis-хранилище**: «Every 32 gradient steps, the optimizers publish a new version of the parameters to a central Redis storage called the controller. The controller also stores all metadata about the state of the system, for stopping and restarting training runs» [F]. Forward-pass GPU «frequently poll the controller to gather the newest parameters» [F]. То есть: pull весов по версии, метаданные запуска переживают рестарт хаба (если Redis переживает).

**В Dactyl-версии Rapid** [F]: «Worker machines randomly connect to a Redis server from which they pull new policy parameters and to which they send new experience. The optimizer machine has one MPI process for each GPU, each of which gets a dedicated Redis server»; воркеры «are assigned to an instance randomly» для балансировки; пул 384 worker-машин по 16 ядер.

**Лига/оппоненты** [F, App. N]: 80 % игр против последних весов, 20 % против прошлых версий; прошлые оппоненты с «quality score» q_i, выбор softmax(q), каждые 10 итераций текущий агент добавляется в пул, обновление q после каждой игры (η = 0.01). Выбор оппонента делает rollout-сторона по таблице, раздаваемой через controller (деталей протокола нет — **не публично**).

**Оценка** [F, App. J]: «a dedicated set of computers continually download the latest agent parameters and plays games between the latest trained agent and the reference agents»; пул из 83 эталонных агентов (TrueSkill 0…254); играют только с агентами не дальше 10 пунктов TrueSkill; после 750 игр версия логируется; обновление «approximately once every two hours».

**Staleness (главный урок)** [F, §3.2, App. M.2]: данные отправляются сегментами по 256 шагов (раньше — целыми играми, возраст данных — часы, тысячи шагов градиента); метрика staleness = M − N (версия оптимизатора минус версия генерации); «increasing staleness by ~8 versions causes significant slowdowns»; «Our final system design targeted a staleness between 0 and 1»; дополнительно метрика sample reuse (потреблено/поступило).

**Отказоустойчивость / эксплуатация** [F]: «surgery» (перенос обучения на изменённую модель/среду без перезапуска) — свыше 20 раз за 10 месяцев; обучение «spread over 10 months of real time due to restarts and reverts»; учёт compute ведётся по упрощённой истории, «rather than keeping track of every change and every time something crashed and needed to be restarted» (то есть падения были рутиной). Детали автоперезапуска машин (K8s и т.п.) **не публичны**.

**Масштаб** (Table в App. C) [F]: Rerun — 512 оптимизаторных GPU, 512 rollout-GPU, 51 200 rollout-CPU; OpenAI Five — 480↔1 536 оптимизаторных GPU, 500↔1 440 rollout-GPU, 80 000↔172 800 CPU; общий батч до 3 145 728 шагов. Стоимость: оптимизация ≈ 30 %, forward-pass GPU ≈ 30 %, rollout CPU ≈ 30 %, остальное (controllers, TrueSkill evaluators, …) ≈ 10 % [F, App. A].

### 2.3. SEED RL (Google, 2019) — контраст: центральный inference

Источник: [SEED RL, arXiv 1910.06591](https://arxiv.org/abs/1910.06591).

- Actors только шагают среду; **inference и накопление траекторий — на learner** («conceptually a single-machine setup»); связь — gRPC **streaming RPC**, соединение остаётся открытым, метаданные шлются один раз, на learner батчирующий модуль [F].
- Learner с тремя типами потоков: inference / prefetching / training; actors раздаются по TPU-хостам round-robin [F].
- Масштаб (Table): 64 ядра TPU v3 → 12 480 окружений, 4 160 CPU актёров, **2.4M FPS (80× относительно IMPALA-базы)**; но на 2×P100 при 176 окружениях SEED даёт 19K FPS против 30K у IMPALA (0.63×) [F]. [И]: центральный inference зависит от быстрого ускорителя и быстрой сети; на CPU/Ethernet-кластере выигрыш не гарантирован.
- Деплой: Docker-образ и скрипт запуска на Google AI Platform, приложение A.4 [F]. Отказоустойчивость, динамический состав — в статье **не обсуждаются**.
- Для лиги неудобно: один learner = одна политика на весь inference; несколько разных архитектур в одном матче потребуют нескольких inference-серверов (на практике так делают TLeague `InfServer`, KaiWu Inference Server).

### 2.4. Sample Factory (Petrenko et al., 2020; v2 документация)

Источники: [arXiv 2006.11751](https://arxiv.org/abs/2006.11751); документация: [PBT](https://www.samplefactory.dev/07-advanced-topics/pbt/), [multi-policy](https://www.samplefactory.dev/07-advanced-topics/multi-policy-training/); флаги: [`cfg.py`](https://github.com/alex-petrenko/sample-factory/blob/master/sample_factory/cfg/cfg.py).

- Архитектура оптимизирована под **одну машину** («optimized for a single-machine scenario») [F, статья]; в документации PBT: «Current version of PBT is implemented for a single machine» [F].
- Популяция: `--num_policies=N`; на многогрупповых GPU-машинах «each policy gets its own GPU», при нехватке GPU политики раскладываются по кругу (4 политики на 2 GPU: GPU0 = {0,2}, GPU1 = {1,3}) [F]. Назначение политик агентам в одной среде «periodically resampled at random», логика в `agent_policy_mapping` [F]. Полное смешивание политик в одной среде требует `--async_rl=True` и `--batched_sampling=False` [F].
- PBT: ранжирование в конце «meta-training epoch», нижние K % заменяются копией случайного верхнего с мутацией гиперпараметров; текущие параметры в `policy_XX_cfg.json` в директории эксперимента [F] (механизм передачи весов в документации не описан [S/не публично]).
- Staleness: **`--max_policy_lag` (default 1000)** — «Discard all experience that is older than this» [F, `cfg.py`]. Лаг определяется в версиях политики; значение по умолчанию практически «выключено», то есть опора на V-trace/PPO-клиппинг.
- Масштаб: 36-ядерный ПК с 4 GPU обучает популяцию из 8 агентов (VizDoom Duel/Deathmatch) [F, статья].
- Control plane в смысле хаба/лизов **отсутствует** (процессы одного узла, очереди FIFO на политику) [F].

### 2.5. Podracer / Sebulba (DeepMind, 2021) — только контраст

Источник: [arXiv 2104.06272](https://arxiv.org/abs/2104.06272); алгоритмическая часть в `research/01_distributed_rl_architectures.md` §2.1.

Sebulba: среда на CPU-хосте, действия и обучение на TPU-ядрах **внутри одной TPU Pod-«шайбы»**, синхронизация градиентов `pmap` по всем learner-устройствам (всех слайсов) [F]. Control plane как отдельный сервис не нужен: состав фиксирован на время задания (gang scheduling). Урок для Colosseum: **когда топология жёсткая и симметричная, хаб не нужен**, а когда нужны динамические присоединения/отключения, приходится платить сложностью (ниже).

### 2.6. Acme + Launchpad + Reverb (DeepMind)

Источники: [Acme, arXiv 2006.00979](https://arxiv.org/abs/2006.00979); [Launchpad, arXiv 2106.04516](https://arxiv.org/abs/2106.04516); [Reverb, arXiv 2102.04736](https://arxiv.org/abs/2102.04736).

**Launchpad** [F]: программа = **ориентированный граф сервисных узлов**, рёбра = RPC (Courier поверх gRPC); `Program.add_node` возвращает `Handle`, передача handle другому узлу = ребро; resource groups задают CPU/RAM/accelerators; `ColocationNode` явно говорит «эти узлы на одной машине»; launcher'ы платформенные (в open-source — только локальные: многопроцессный/многопоточный) [F]. Адреса — плейсхолдеры, резолвятся на этапе launch (`address table`).

**Отказоустойчивость Launchpad — сознательно не своя** [F]: «Unlike Ray, Launchpad does not itself implement mechanisms providing fault tolerance. Instead, we assume only that the underlying job scheduling system (e.g. Kubernetes) has the ability to restart failing jobs. In practice, this requires that stateful nodes—often learning nodes holding model parameters—have the ability to restore themselves. … actor processes … are stateless … For learning processes … periodic checkpointing allows us to restart the process with minimal losses». Авторы аргументируют: «exact» механизмы (lineage Ray) дороги, а в ML важны агрегированные поведения.

**Acme** [F]: `make_distributed_experiment` строит Launchpad-программу из узлов: environment loops (actor+env), learners, сервис Reverb. Actor периодически опрашивает `VariableSource` (в роли источника — Learner) через `VariableClient` в отдельном потоке (pull весов). Evaluator — такой же узел с выключенной эксплорацией. Счётчики (`Counter`) — общий узел для метрик и учёта шагов.

**Reverb** [F]: сервер таблиц с `Table`/`Item`/`Chunk`; **`RateLimiter`** (`SampleToInsertRatio`, `MinSize`, `Queue`) ограничивает соотношение вставок и выборок: «if … one of the processes starts lagging behind the other, the rate limiter will block the latter while the former catches up» [F, Acme] — готовый механизм backpressure/«max lag» без явных версий. Чекпоинты таблиц на диск по gRPC-вызову клиента, на время чекпоинта сервер блокируется (3.7 в статье Reverb) [F]. Шардирование между серверами — на стороне клиента (Menger так и делает, §2.12).

Для лиги и динамического состава **Acme/Launchpad ничего готового не даёт**: граф статичен в момент запуска [F — «program … launched»; динамическое присоединение узлов не описано].

### 2.7. moolib (Meta / FAIR, 2022)

Источники: [whitepaper PDF](https://research.facebook.com/file/268346465322053/moolib--A-Platform-for-Distributed-RL.pdf) ([страница публикации](https://research.facebook.com/publications/moolib-a-platform-for-distributed-rl/)); [репозиторий](https://github.com/facebookresearch/moolib) (архивирован 31.10.2023 [F]); [`src/broker.h`](https://github.com/facebookresearch/moolib/blob/main/src/broker.h), [`py/moolib/broker.py`](https://github.com/facebookresearch/moolib/blob/main/py/moolib/broker.py).

- **Модель: группа однородных пиров**, каждый крутит один и тот же цикл; а не IMPALA-разделение на learner/actors. Авторы: «a dynamically-sized group of homogeneous peers is often easier to reason about and manipulate» [F].
- **Accumulator** — state machine («wants gradients» → «reducing» → «has gradients»); редукция градиентов асинхронная, виртуальный батч набирается теми пирами, кто успел; **лидер** распределяет своё состояние (веса, состояние оптимизатора) новым пирам и периодически, чтобы «new peers join ad-hoc» [F].
- **Broker** — отдельный процесс (порт по умолчанию 4431), один на любое число экспериментов, группы различаются по имени `group` [F]. Из кода: пиры шлют `BrokerService::ping(group, name, timeoutMilliseconds)`, брокер хранит `lastPing`/`timeoutDuration`, при изменении состава запускает синхронизацию `syncId` («Sync %s %d/%d in %gs»), активные пиры получают `sortOrder` (отсюда лидерство) [F]. То есть **настоящие лизы с клиент-задаваемым TTL и перенумерацией группы**.
- RPC на TensorPipe с автоматическим выбором транспорта (shared memory / InfiniBand / TCP), RPC «tried again, if necessary via a different transport type» при временных сбоях; произвольные Python-объекты через pickle [F] (то есть безопасность = доверенная сеть).
- Имя пира уникально: «hostname, process id и случайный slug» (`learnmachine7:4020365:mega-dragon`) [F] — хороший паттерн идентификаторов узлов.
- Масштаб: «from single GPUs to hundreds» (заявление) [F: README]; конкретных чисел в тексте нет.
- Лиги, матчмейкинга, весов по версиям — **нет**; moolib — транспорт и редукция.

### 2.8. TorchBeast / PolyBeast (FAIR, 2019)

Источники: [arXiv 1910.03552 §5.2](https://arxiv.org/abs/1910.03552); [репозиторий](https://github.com/facebookresearch/torchbeast).

- **Инверсия ролей**: environment servers «wait for incoming gRPC connections», learner-процесс запускает C++-потоки actors, которые **сами подключаются к env-серверам** по двунаправленному gRPC-стриму; для каждого подключения сервер создаёт новую копию среды [F].
- Inference на learner с динамическим батчингом (inference queue → `DynamicBatcher`), траектории по `unroll_length` в learner queue [F]. То есть это «SEED-подобный» дизайн в PyTorch.
- Плюс: добавить машину со средами = запустить env-сервер и перечислить его адрес у learner (список `--server_addresses`); минус: **динамически он не обнаруживается** (адреса статичны) [F по README: `num_actors` должен совпадать с числом серверов].
- Отказоустойчивость и масштабные числа в README не приводятся; на уровне статьи — «on par with TensorFlow IMPALA» по качеству [F].

### 2.9. DI-engine и DI-star (OpenDILab)

Источники: [`coordinator.py`](https://github.com/opendilab/DI-engine/blob/main/ding/worker/coordinator/coordinator.py), [`base_league.py`](https://github.com/opendilab/DI-engine/blob/main/ding/league/base_league.py), [docs: distributed](https://di-engine-docs.readthedocs.io/en/latest/03_system/distributed.html), [DI-orchestrator `architecture.md`](https://github.com/opendilab/DI-orchestrator/blob/main/docs/architecture.md), [DI-star](https://github.com/opendilab/DI-star) (`distar/ctools/worker/league/league_api.py`, `league.py`, `actor_comm.py`, `coordinator/coordinator.py`).

**DI-engine, «parallel mode» (legacy `ding/worker/coordinator`).** [F, по коду]
- Один `Coordinator` с `CommCoordinator` и словарём callback'ов (`deal_with_collector_send_data`, `deal_with_learner_get_data`, `deal_with_learner_finish_task`, …); явного шага регистрации нет, **heartbeat не реализован**, вместо него таймауты задач `collector_task_timeout` (по умолчанию 30) и `learner_task_timeout` (600).
- Задачи производит `Commander` (`get_collector_task`, `get_learner_task`) под `_commander_lock`; координатор **назначает** задачи коллекторам/learners (push), с повторами во временном окне (0.3·timeout для collector, 0.1·timeout для learner; пауза 3 с), затем помечает задачу как failed.
- Replay buffer живёт **в координаторе** (по `buffer_id`), коллекторы шлют данные через координатор — **data plane не отделён** (отличие от TLeague/AlphaStar).
- Динамика: `deal_with_increase_collector` / `deal_with_decrease_collector` меняют `task_space` коммандера.
- `operator_server` объявлен в конфиге, но в классе не используется.

**DI-engine, новый режим (`ditask` / `Parallel`).** [F, документация] Процессы маршрутизируют события через встроенный router (сети: mesh, star или Redis как middleware), распределённый режим = многопроцессный (`ditask --parallel-workers N`); в K8s ресурс `DIJob` раскрывает IP всех подов в переменных окружения. Динамическое присоединение/отключение узлов, отказоустойчивость: «не описаны» на этой странице.

**DI-orchestrator (K8s-оператор).** [F] `di-operator` (контроллер CRD `DIJob`, Allocator `Allocate`/`Optimize`, фазы Pending/Starting/Running/Succeeded/Failed/Restarting/Rescheduling) + `di-server` (HTTP: `GET/POST/DELETE /v2alpha1/[job_id]/replicas`, «Server will only operate on preemptible DIJobs»; реальные операции делает operator). Рестарт: `backoffLimit` по умолчанию 3, `cleanPodPolicy`. Каждая `Task` имеет тип `learner|collector|evaluator|none`.

**DI-star (Flask-HTTP лига для StarCraft II).** [F, по коду]
- `League` как HTTP-сервис: `POST /league/register_learner`, `/league/learner_send_train_info`, **`/league/actor_ask_for_job`**, **`/league/actor_send_result`**; отладочные `/league/show_payoff` и т.п., `GET /league/save_resume`.
- **Задание содержит `checkpoint_paths`** (пути на **общей ФС**) и `player_ids`, `side_ids` и т.д.; actor делает `torch.load(job['checkpoint_paths'][idx])`, а свежие веса подтягивает через `Adapter.pull(token=player_id + 'model')` с HTTP-координатора, по таймеру `actor_model_update_interval`; повторы запросов `Retry(total=20, backoff_factor=1)`.
- Координатор (`/coordinator/register`, `/push`, `/pull`) — буферный посредник на `deque(maxlen=1000)` на токен; удаление сервера только после >5 запросов `remove_server` на один ip:port; **heartbeat нет** [F по коду].
- Персистентность: поток сохраняет `league_resume` раз в `save_resume_freq` (по умолчанию 3600 с), при старте `resume_path` загружает лигу [F].
- Выбор сложности: Flask + `requests` + файловая система = просто, но всё доверенное и на одной сети.

### 2.10. TLeague (Tencent Robotics X, 2020) и TStarBot-X

Источники: [TLeague, arXiv 2011.12895](https://arxiv.org/abs/2011.12895) (§3.2–3.4, Table 3), [репозиторий проекта](https://github.com/tencent-ailab/tleague_projpage), [TStarBot-X, arXiv 2011.13729](https://arxiv.org/abs/2011.13729).

**Модули** [F]: `Actor` (+ Env, Agt), `Learner` (+DataServer, ReplayMem; несколько learners одного агента синхронны через Horovod/NCCL), `InfServer` (опциональный GPU-сервер batch inference), `ModelPool`, `LeagueMgr` (+ `GameMgr` с платёжной матрицей и алгоритмами выбора оппонентов, + `HyperMgr` с гиперпараметрами моделей, мутации для PBT).

**Протокол заданий (pull)** [F]: «At each episode beginning, the Actor requests a task from the LeagueMgr to know what the current (itself) learning policy πθ and what the opponent policy πφ. At each episode ending, it also reports the game outcome to LeagueMgr»; траектории — Learner'у напрямую; «Periodically, the Actor pulls up-to-date policy parameters θ and φ from ModelPool». Learner в начале «learning period» получает задачу от LeagueMgr, согласованную с задачами actors; запрашивает её только rank-0 learner; в конце периода θ **замораживается в ModelPool** (= снимок для лиги).

**ModelPool** [F]: хранит параметры всего пула оппонентов **в памяти**; «up to M_M ModelPool replicas can be run simultaneously and a random one is picked to do the concrete response»; read/write — и у learners, и у actors.

**Транспорт** [F]: микросервисы, RPC на **ZeroMQ** с собственным Python-протоколом сообщений; сноска: «Other scheme is possible, e.g., using protobuf and gRPC».

**Деплой на K8s** [F]: LeagueMgr, ModelPool, Learner, InfServer — K8s **Service** (endpoint вида `tcp://signature-league-mgr:9003`), Actor — **Deployment/ReplicaSet** («scale-up/-down the number of Actors … automatically restarts the Actor in case it encounters an unrecoverable error»; в качестве причины приводят «occasional core-dump of the Env binary»), размещение через `nodeSelector`; общая директория через **PVC** (CFS/NFS-подобная ФС Tencent); спецификация: один yaml через jinja2, `kubectl apply/delete`; оператор (kubeflow) осознанно не использован: «the yml-jinja2-kubectl way suffices». Пример спецификации «56 Learners, 8 InfServers, each Learner corresponds to 16 actors … every 7 Learners and 1 InfServer co-located in the same GPU machine» [F].

**Масштаб** (Table 3, на одного обучаемого агента, M_G — число параллельных агентов) [F]: Dota 2 1v1 — 60 000 CPU, 256 GPU; StarCraft II AlphaStar-вариант (данные из AlphaStar) — 12 агентов; TStarBot-X — 3 агента, 4 200 CPU и 96 GPU на агента, rfps 1.7K, cfps 4.2K; ViZDoom — 1 152 CPU, 32 GPU, rfps 6.0K; Pommerman — 100 CPU, 2 GPU. Метрики rfps/cfps (приём/потребление кадров) — удобный индикатор перекоса actors/learner (cfps/rfps = число повторов кадра) [F].

**Отказоустойчивость/мониторинг/безопасность**: actor — рестарт K8s; LeagueMgr/ModelPool — Service (K8s перезапустит под, но восстановление **состояния** лиги в статье не описано) [F/не публично]; аутентификации нет [F по описанию протокола, подтверждения auth-слоя в тексте нет].

### 2.11. MALib (SJTU / UCL, 2021–2023)

Источники: [arXiv 2106.07551](https://arxiv.org/abs/2106.07551) (читалась ar5iv-версия [ar5iv](https://ar5iv.arxiv.org/html/2106.07551), пересказом), [JMLR 24(22-0169)](https://jmlr.org/papers/v24/22-0169.html).

- **Centralized Task Dispatching (CTD)**: центральный контроллер («Coordinator Server») планирует задачи обучения и симуляции, рабочие процессы (Actors, Learners, data servers) «semi-passive»: «after finishing a task, they send a task request back to the coordinator» [S]. Задачи порождаются алгоритмом PB-MARL (PSRO: N обучающих задач на N агентов) [S].
- Роли: **Actor–Evaluator–Learner**; **Evaluator** ведёт payoff table по сочетаниям политик (PSRO берёт из неё Nash) [S]. Параметры — Parameter Server, опыт — OfflineDataset Server [S].
- Реализовано на Ray (low-level stack) [S].
- Масштаб: «>40K FPS» на одной 32-ядерной машине (формулировка аннотации; в ar5iv-тексте число искажено) и ускорение в 5× над RLlib / ≥3× над OpenSpiel по их собственным замерам [S, самоотчёт]; RLlib в их тесте не стартовал при >32 CPU-воркеров [S].
- Отказоустойчивость, лизы, сохранение состояния координатора — в тексте **не обсуждаются** [S].

### 2.12. Tencent Honor of Kings: KaiWu / JueWu и Menger (Google)

Источники: HoK: [Towards Playing Full MOBA Games, arXiv 2011.12692](https://arxiv.org/abs/2011.12692) (§3.5, §8.1), [JueWu / «Mastering Complex Control in MOBA Games», arXiv 1912.09729](https://arxiv.org/abs/1912.09729), [Honor of Kings Arena, arXiv 2209.08483](https://arxiv.org/abs/2209.08483); Menger: [блог Google Research](https://research.google/blog/massively-large-scale-distributed-reinforcement-learning-with-menger/).

**KaiWu / JueWu (2019–2020)** [F]: четыре модуля — **AI Server** (актор: взаимодействие среды и модели; один AI Server привязан к одному CPU-ядру, inference на CPU через FeatherCNN), **Dispatch Module** (сбор, сжатие и пересылка выборок; привязан к нескольким AI Server на той же машине), **Memory Pool** (сервер, кольцевая очередь для выборок разной длины), **RL Learner** (ring-allreduce, обмен с Memory Pool через shared memory). Модели синхронизируются на AI Servers **peer-to-peer от мастера-learner'а**. Версия 2011.12692 добавляет «centralized inference module on the GPU side … similar to … Seed RL» (Inference Server). Масштаб: 600 000 CPU-ядер в Docker и 1 064 GPU (JueWu, 1v1) [F]; в полной MOBA-версии «320 GPUs and 35,000 CPUs, referred to as one resource unit», кластер выдерживает «6 to 7 such experiments in parallel» [F]. HoK Arena (открытая среда) [F]: 200 машин / 2 000 одновременных сред дают «over 800 million samples per» (формулировка обрывается в прочитанном тексте; единица времени не прочитана — не использовать без проверки) и ~4.34 млн выборок/ч с одной 10-ядерной машины.

**Menger (Google, 2020)** [F, блог]: actors с локальным inference («similar to Acme»), **сегментированный Reverb** (каждый шард обслуживает группу actors, обычно из той же Borg cell), **кэширующий слой** между learner и actors: learner шлёт обновления только кэшам, каждый кэш обслуживает actors своей cell; для модели 16 МБ и 512 actors средняя задержка чтения падает примерно в 4.0×; рост записи в один буфер 6.2× (16 МБ) и 18.9× (512 МБ) при росте actors с 16 до 2048; тысячи actors в нескольких Borg cells; до 8.6× ускорение на задаче chip placement (авторские цифры). Отказоустойчивость в блоге **не обсуждается**.

### 2.13. SRL (Tsinghua, ICLR 2024) — дополнительно, из-за системного дизайна

Источник: [SRL, arXiv 2306.16688](https://arxiv.org/abs/2306.16688).

[F] Абстракция: **actor workers, policy workers, trainer workers** + **data streams** (inference stream, sample stream) + сервисы (**parameter server — NFS**, «has enough throughput according to our experience»). **Controller + scheduler**: при запуске эксперимента «submits a controller and all workers to the cluster»; контроллер «monitoring and managing the lifetime of workers»; воркеры на одной ноде автоматически получают shared-memory соединение. Динамический батчинг на policy worker: «enable fault tolerance when some actor worker crashes due to unstable environment implementations». «Threaded environment ring» — среды в отдельных потоках actor worker, что «can largely benefit fault tolerance». Заявления: решает hide-and-seek на уровне качества Rapid с ускорением 3× (CPU inference) и 5× (GPU inference), доходит до 15K CPU-ядер [F]. Sample/inference streams выбирают самый быстрый транспорт (сокеты или shared memory) [F].

### 2.14. Ray RLlib (только как ориентир)

Источники: [RLlib fault tolerance](https://docs.ray.io/en/latest/rllib/rllib-fault-tolerance.html), [Ray GCS fault tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/gcs.html), [RLlib key concepts](https://docs.ray.io/en/latest/rllib/key-concepts.html), [Ray fault tolerance overview](https://docs.ray.io/en/latest/ray-core/fault-tolerance.html).

- Архитектура новой API-стека [F]: `Algorithm`, `EnvRunnerGroup` (n EnvRunner-акторов), `LearnerGroup` (m Learner-акторов, DDP), `RLModule`/`MultiRLModule`; «The algorithm synchronizes model weights after an update» (механизм — в самой Ray).
- Отказоустойчивость EnvRunner [F]: `restart_failed_env_runners`, `restart_failed_sub_environments`, `validate_env_runners_after_construction`, `num_consecutive_env_runner_failures_tolerance`; «Training continues with the remaining healthy EnvRunners, though at reduced speed»; для learners «The page does not cover learner failures» (в документации); эксперимент целиком — Tune с периодическими чекпойнтами.
- Ray Core [F]: **GCS** хранит метаданные в памяти, «If it fails, the entire Ray cluster fails» — если не подключить внешний Redis (официально поддержан только с KubeRay); таймаут переподключения raylet — 60 с (`RAY_gcs_rpc_server_reconnect_timeout_s`); во время восстановления GCS нельзя создавать акторов и регистрировать узлы, но уже работающее продолжает работать.
- Лига: в RLlib есть пример `self_play_league_based_with_open_spiel` [S: зеркала репозитория; пример на старом API; в форуме Ray есть сообщение о том, что лига-вариант «stops learning after first generation»] — лига реализована как **callback в процессе алгоритма**, `policy_mapping_fn` меняется при добавлении снимка [S]. Это пример «лига внутри главного процесса», как в Colosseum сегодня.
- Вывод [И]: Ray даёт общий механизм акторов и рестарта, но **не даёт** ни лиги, ни учёта матчей, ни версионированных весов, а размер зависимости и сложность отладки — именно то, чего Colosseum избегает (решение «No Ray»).

### 2.15. PufferLib

Источник: [PufferLib, arXiv 2406.12905](https://arxiv.org/abs/2406.12905).

Векторизация сред: serial / multiprocessing / Ray-backend с единым API; shared-memory массивы и флаги, busy-wait воркеры [F]. Межмашинной сети, хаба и лиги в системе **нет** [F]. Для SP5 неинтересен, кроме идеи «`vec_env` как сменный backend».

### 2.16. HandyRL (DeNA) и победители Kaggle

Источники: [HandyRL](https://github.com/DeNA/HandyRL) (`handyrl/train.py`, `worker.py`, `connection.py`, [docs/large_scale_training.md](https://github.com/DeNA/HandyRL/blob/master/docs/large_scale_training.md)); [Lux AI S1, 1-е место, README](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021); итоги соревнований по `research/sp4_selection_practice.md` §1.1 (там же GRF-5 [thread](https://www.kaggle.com/c/google-football/discussion/203412)).

**HandyRL — самый близкий по духу к «просто и гибко» пример.** [F, по коду]
- Learner (`--train-server`) слушает **два порта**: 9999 «entry server» — воркер присылает свою конфигурацию (`address`, `num_parallel`), в ответ получает `args` с уникальным `base_worker_id` (динамическая регистрация узла); 9998 — рабочие соединения. Воркер-машина запускается `python main.py --worker` с `server_address`, «anytime and from anywhere … multiple workers can connect».
- На каждой машине между воркерами и learner стоит **`Gather` — прокси**: он (1) запрашивает у сервера пачку заданий `'args'` (`buffer_length = 1 + n_workers // 4`) и раздаёт локально, (2) **кэширует модели по `model_id`** (`data_map['model']`), (3) буферизует результаты и отправляет пачкой. Число gathers = `1 + (num_parallel − 1) // 16`. Это ровно «per-machine weight cache + batched results» Colosseum.
- Задание (`args`) создаёт сервер: роль `'g'` (генерация) или `'e'` (оценка) по доле `eval_rate`; `model_id` по игроку (`-1` = случайная модель); `req == 'model'` отдаёт запрошенную эпоху или последнюю при ошибке загрузки.
- Транспорт: сокет + `pickle` (`PickledConnection`), очереди ограничены (`Queue(maxsize=256)`); отключение соединения (`EOFError`, `ConnectionResetError`, `BrokenPipeError`) = просто удалить из множества; **нет ни heartbeat, ни перераспределения потерянных заданий, ни аутентификации.**
- Пример размещения из документации: GPU-машина для learner, 64-CPU машина без GPU для генерации эпизодов [F].
- Результаты: 1-е место Hungry Geese, 5-е место Google Research Football (с Man City) [F: README]. GRF-5: в последние 3 дня ~4000 CPU и 20 GPU, часть воркеров постоянно оценивает чекпойнты [F: Kaggle thread, как процитировано в `research/sp4_selection_practice.md`].

**Lux AI S1, 1-е место (Toad Brigade).** [F, README] «All training was done on my personal PC - an 8-core/16-thread dual-GPU system»; FAIR-реализация IMPALA (TorchBeast-подобный `run_monobeast.py`), замороженный учитель-KL, архив сабмитов `internal_testing/hall_of_fame`. Вывод [И]: масштаб соревнования определяется не кластером, а итерационной скоростью; лиге на нескольких машинах нужно **не мешать** одиночному режиму.

**Lux AI S2 (академическая статья, не победитель).** [F: [arXiv 2304.13004](https://arxiv.org/abs/2304.13004)] PFSP с пулом чекпойнтов в виде очереди фиксированного размера, «60 environment steps per second running on a single A6000 GPU», итоговый запуск «8 RTX8000 GPUs in parallel for 220M environment steps during 124 hours»; связь с победителем не заявлена.

---

## 3. Сравнительные таблицы

Таблицы разбиты по группам вопросов. «—» = не публично или в источнике нет. Ссылки на источники даны в разделе соответствующей системы (§2).

### 3.1. Hub, регистрация, динамика, размещение

| Система | Hub / координатор | Регистрация / обнаружение | Heartbeat / лизы | Динамическое добавление/удаление | Размещение learner'ов |
|---|---|---|---|---|---|
| AlphaStar | один central coordinator (матрица выплат, матчи по запросу, сброс эксплойтеров) | — | — | — | 12 независимых actor-learner копий; статически |
| OpenAI Five / Rapid | «controller» = Redis (параметры + метаданные) | воркеры подключаются к Redis, случайный экземпляр | — | размер пулов менялся по ходу проекта (80 000↔172 800 CPU) | оптимизаторы: MPI + NCCL allreduce, по Redis на GPU-процесс |
| SEED RL | learner = хаб (inference) | actors подключаются к learner по gRPC | — | — | actors раздаются по TPU-хостам round-robin |
| Sample Factory | нет (один узел) | — | — | PBT на одной машине | политики по GPU по кругу |
| Acme + Launchpad | нет; граф статичен | handles + address table при launch | делегируется планировщику | при запуске | resource groups, colocation |
| moolib | **Broker** (discovery) | уникальное имя пира + группа | **ping с TTL**, `syncId`, перенумерация, лидер | пиры присоединяются «ad-hoc» | однородные пиры |
| TorchBeast / PolyBeast | learner | learner дозванивается до адресов env-серверов (статический список) | — | вручную перезапуском | один learner |
| DI-engine (legacy) | `Coordinator` + `Commander` | неявно через comm | **нет**, таймауты задач | `increase/decrease_collector` | через `DIJob`/Allocator в K8s |
| DI-star | Flask-`League` + `Coordinator` | `register_learner`, `/coordinator/register` | нет | actors приходят и спрашивают задание | learner на ранг, `torch.distributed` init |
| TLeague | `LeagueMgr` (+GameMgr, HyperMgr) | K8s Service (endpoint по DNS) | K8s | K8s Deployment для actors | K8s + `nodeSelector`, co-location |
| MALib | Coordinator Server (CTD, Ray) | Ray | Ray | semi-passive workers по запросу задач | Ray scheduler |
| HoK/KaiWu | — (в статье модули без координатора) | — | — | — | master learner синхронизирует модели |
| Menger | — | — | — | actors в нескольких Borg cell | — |
| SRL | controller + scheduler | scheduler при запуске | monitoring lifetime | — | scheduler (колокация, ресурсы) |
| RLlib | Algorithm + Ray (GCS) | Ray | raylet↔GCS (60 с) | `EnvRunnerGroup` эластична | `LearnerGroup` |
| HandyRL | learner-процесс (entry 9999 / workers 9998) | **entry server выдаёт `base_worker_id`** | нет (разрыв сокета) | «anytime and from anywhere» | один learner |

### 3.2. Лига, матчмейкинг, результаты

| Система | Кто выбирает матч | Как возвращаются результаты | Evaluators | Сколько агентов / популяций |
|---|---|---|---|---|
| AlphaStar | координатор по запросу | через координатор → матрица выплат | отдельные CPU evaluator workers | 12 обучаемых + почти 900 заморожённых игроков [S] |
| OpenAI Five | rollout-сторона по таблице q_i (80 % latest / 20 % прошлые) | через controller | отдельные TrueSkill-машины, 83 эталона | 1 агент (self-play с историей) |
| TLeague | `LeagueMgr`/`GameMgr` по запросу actor (pull) | `Actor` → `LeagueMgr` по окончании эпизода | через те же actors / отдельные задания | M_G параллельных агентов (3 в TStarBot-X) |
| DI-star | `League.actor_ask_for_job` (pull) | `actor_send_result` | ladder/eval-job типы в `League` | множество active players |
| DI-engine | `BaseLeague.get_job_info` (+commander) | `finish_job` (payoff + TrueSkill при `eval_flag`) | evaluator role | active/historical lists |
| MALib | coordinator по алгоритму (PSRO) | evaluator → payoff table | Evaluator | policy pool на агента |
| HandyRL | learner отдаёт `args` (роль g/e, `model_id`) | `episode` / `result` через Gather-пачки | доля заданий `eval_rate` | одна сеть + пул прошлых эпох |
| RLlib (пример лиги) | `policy_mapping_fn` в процессе алгоритма | callback | — | 3 группы политик |
| Sample Factory | `agent_policy_mapping` (случайно, периодически) | внутри процесса | — | популяция на одной машине |

### 3.3. Веса, версии, staleness

| Система | Хранение/раздача весов | Версии | Контроль staleness | Что делают со старыми данными |
|---|---|---|---|---|
| AlphaStar | learner → actors (по сети) | — | обновление каждые 10 с | реплей данных дважды [F] |
| OpenAI Five | Redis «controller» (каждые 32 шага) | номер версии | **цель 0–1, 8 уже плохо** | сегменты по 256 шагов; потреблённые/поступившие ≈ reuse |
| TLeague | **ModelPool** (память, реплики) | политики пула; замораживание в конце периода | период обучения «long enough» | blocking queue у learner → on-policy-ность (cfps ≈ rfps) |
| DI-star | общая ФС (`checkpoint_paths`) + HTTP-pull свежих | `model_last_iter` | интервал pull | — |
| Sample Factory | in-process shared memory | версия политики | **`max_policy_lag`** (default 1000) | выбрасывается |
| Acme/Reverb | `VariableSource`/`VariableClient` | — | **RateLimiter** | блокирует писателя/читателя |
| Menger | learner → кэши → actors | — | — | шардированный replay |
| SRL | NFS parameter server | — | — | — |
| HandyRL | `model_id` = эпоха; `Gather` кэширует | epoch | `update_episodes` | V-trace/UPGO |
| moolib | лидер раздаёт состояние пирам | — | виртуальный батч | — |

### 3.4. Отказоустойчивость, учёт, мониторинг, деплой

| Система | Actor падает | Learner падает | Хаб падает | Учёт env-шагов / результатов | Мониторинг | Деплой | Безопасность |
|---|---|---|---|---|---|---|---|
| AlphaStar | — (preemptible CPU) | — | — | — | evaluators, payoff matrix | Google (TPU) | — |
| OpenAI Five | рестарт (подразумевается) | restarts/reverts, surgery | метаданные в Redis | упрощённая история | TrueSkill evaluators | GCP, кастомный Rapid | — |
| Launchpad/Acme | планировщик перезапускает | чекпойнт + рестарт | — | `Counter`-узел | логгеры | локально; платформенные launchers | — |
| TLeague | **K8s Deployment перезапускает** | K8s Service; рестарт состояния не описан | не описано | rfps/cfps | — | **K8s + yaml/jinja2 + kubectl** | — |
| DI-star | retry с backoff | — | `league_resume` раз в час | — | tensorboard по игрокам | процессы + `--type` | нет auth [F по коду] |
| DI-engine | таймауты задач | — | — | — | — | DIJob в K8s (оператор) | — |
| RLlib | `restart_failed_env_runners`, эластично | не описано в доке | Ray GCS: без Redis кластер падает | — | Tune | KubeRay/ray up | — |
| HandyRL | разрыв сокета, задание теряется | процесс умирает — обучение теряется | — | эпизоды считает learner | лог | ssh/Docker | pickle, нет auth |
| moolib | RPC retry, ping-timeout, ресинк группы | лидер переизбирается | broker: перезапуск, пиры переподключаются | — | — | процессы | pickle |
| SRL | потоки внутри actor worker | — | controller | — | — | Slurm [S: благодарность в статье] | — |

### 3.5. Масштаб (как заявлено)

| Система | Масштаб | Источник |
|---|---|---|
| OpenAI Five | до 1 536 GPU-оптимизаторов, 1 440 rollout-GPU, 172 800 CPU | [dota-2.pdf](https://cdn.openai.com/dota-2.pdf) App. C |
| AlphaStar | на агента: 32 TPU v3, 16 000 матчей, ~50k шагов/с learner; 12 агентов | [AlphaStar Methods](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf) |
| SEED | 4 160 CPU + 64 ядра TPU v3: 2.4M FPS | [SEED](https://arxiv.org/abs/1910.06591) |
| TLeague | Dota 1v1: 60k CPU/256 GPU; ViZDoom: 1 152 CPU/32 GPU; TStarBot-X: 4 200 CPU/96 GPU на агента ×3 | [TLeague Table 3](https://arxiv.org/abs/2011.12895) |
| HoK (Tencent) | 320 GPU + 35 000 CPU = 1 «unit»; JueWu 1v1: 600 000 CPU + 1 064 GPU | [2011.12692](https://arxiv.org/abs/2011.12692), [1912.09729](https://arxiv.org/abs/1912.09729) |
| Menger | «several thousand actors», тест до 2048 actors | [блог](https://research.google/blog/massively-large-scale-distributed-reinforcement-learning-with-menger/) |
| SRL | до 15K CPU-ядер | [SRL](https://arxiv.org/abs/2306.16688) |
| Sample Factory | 36 ядер + 4 GPU, популяция 8 | [SF](https://arxiv.org/abs/2006.11751) |
| Dactyl (Rapid) | 384 воркера × 16 ядер = 6 144 ядра; 8 GPU оптимизатор | [Dactyl](https://arxiv.org/abs/1808.00177) |
| Lux S1 победитель | один ПК, 8c/16t + 2 GPU | [README](https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021) |
| GRF-5 (HandyRL) | ~4000 CPU + 20 GPU в последние 3 дня | [Kaggle thread](https://www.kaggle.com/c/google-football/discussion/203412) (через `sp4_selection_practice.md`) |

---

## 4. Типичные ошибки и уроки

1. **Данные устаревают быстрее, чем кажется.** Rapid первоначально отправлял целые игры; возраст данных — часы. Исправление — сегменты 256 шагов и метрика staleness [F: dota-2.pdf §3.2, App. M.2]. [И] Для Colosseum: у нас уже чанки фиксированной длины (T = 128–512) и `behavior_policy_version` в `TrajectoryChunkProto` (`proto/colosseum.proto`); не хватает **измерения лага на приёме у learner** и политики «drop vs weight» с порогом `max_policy_lag`.
2. **Раздача весов от одного learner'а — узкое место.** Menger поставил кэши (латентность чтения −4×, 512 actors, модель 16 МБ); TLeague — реплики ModelPool; HandyRL — прокси на машину; SRL — NFS [F]. [И] Colosseum сегодня: «newest-wins weight-queue eviction unpickles the stale payload» (пункт SP5 в CLAUDE.md) и каждый воркер тянет веса сам.
3. **Pull с лизом лучше push.** Push-задачи (DI-engine) требуют знания «кто жив»; в DI-engine поэтому появились таймауты и retry-окна. Pull с TTL на задание (как moolib ping или `Retry`/`backoff_factor` в DI-star) проще: умер воркер — задание вернётся по таймауту [И, на основании §2.7, §2.9].
4. **Результаты без идемпотентности.** HandyRL/DI-star при разрыве теряют незавершённое; никто не пишет exactly-once. [И] В Colosseum у `MatchResult` есть `uid` — достаточно «at-least-once доставка + дедупликация по `uid` в match log» (SQLite `INSERT OR IGNORE`).
5. **Состояние лиги — это данные.** DI-star пишет `league_resume` раз в час (потеря до часа состояния), Rapid держит метаданные в Redis, Ray без внешнего Redis теряет всё при падении GCS [F]. [И] В Colosseum состояние лиги уже пишется (`league_state.json`, `matches.db` WAL, `ratings.json`) — хаб должен лишь **стартовать с него**.
6. **Падения сред — норма, а не исключение.** TLeague обосновывает K8s именно «occasional core-dump of the Env binary»; SRL — «actor worker crashes due to unstable environment implementations»; Rapid-эксплуатация — «restarts and reverts» [F]. [И] Это соответствует идее Colosseum `PlayerError`/`EnvContractError`, но для удалённого узла нужен режим «убить окружение и продолжить с другими» (счётчик падений, а не падение всего прогона).
7. **Pickle и единая версия Python/torch.** HandyRL (pickle), moolib (pickle), TLeague (собственный Python-протокол сообщений) связывают версии кода узлов [F]. Colosseum уже отказался от pickle (JSON + `np.savez`, проект «no pickle»); хаб должен добавить **проверку версии протокола и хэша конфига при регистрации**.
8. **Центральный inference не бесплатен.** SEED на P100 0.63× от IMPALA [F]. Рекомендация: оставить CPU-inference на воркерах; центральный inference делать опциональным на GPU-узле (TLeague `InfServer` и HoK Inference Server — это именно «опциональный» вариант).
9. **K8s-оператор — не обязательное условие.** TLeague без оператора («suffices»); Launchpad не реализует отказоустойчивость сам; DI-orchestrator полноценный оператор — это уже отдельный проект. [И] Для пользователя Colosseum (1 GPU-машина + пара CPU-машин) `docker compose` / ssh важнее K8s.
10. **Лидер/хаб — единственная точка отказа без явной стратегии.** Ray: падение GCS = падение кластера (без Redis); DI-star: час потери; TLeague: не описано. Нужна явная стратегия: «хаб падает → узлы продолжают текущие задания до истечения лизов, затем ждут» (см. §5).
11. **Backpressure нужен с первого дня.** Reverb RateLimiter, `Queue(maxsize=256)` в HandyRL, `deque(maxlen=1000)` в DI-star; в Colosseum очередь результатов раньше теряла записи при быстрых играх (FX1: батчи, очередь на 1 000 батчей) — тот же класс проблем на уровне хаба [CLAUDE.md, «Parked during SP4»].
12. **Совместимость старых снимков при смене среды/кода.** OpenAI Five обновлял код у старых агентов, чтобы они играли новую версию игры; «если surgery не удаётся, …» [F: dota-2.pdf App. B]. [И] В Colosseum `meta.json` с архитектурой уже есть; хабу нужен реестр `{agent, version, arch_hash, env_hash}`.
13. **Метрики дисбаланса actors/learner.** TLeague вводит rfps/cfps; OpenAI — sample reuse [F]. [И] Это те числа, по которым хаб должен подсказывать «добавьте воркеров» или «learner не успевает» — основа для (ручного) автоскейла.
14. **Динамика состава learners — самое сложное.** TLeague и moolib синхронизируют learners (Horovod/Accumulator); добавление требует остановки или ресинка группы (moolib делает ресинк через broker, TLeague — рестарт). [И] У Colosseum «один learner = один агент» (CLAUDE.md), поэтому добавление агента не требует ресинка существующих learners — **это преимущество, его надо сохранить** (не вводить DDP между агентами).

---

## 5. Выводы для Colosseum

### 5.1. Что у нас уже есть (контекст)

[F из репозитория `/home/viv/dev/repos/Colosseum`]: `colosseum serve-weight-store`, `run-learner`, `run-workers` на gRPC (`proto/colosseum.proto`: `WeightStoreService` с `PutWeights/GetWeights/GetVersion`, `TrajectoryService.SendChunks` стримом); `src/colosseum/distributed.py` оборачивает очереди в gRPC-адаптеры; `check_distributed_scope` отказывается от всего SP3/SP4 (скриптованные/замороженные агенты, init, кастомный матчмейкер, match log, рейтинги, pulse). Вся лига, рейтинги, match log (SQLite), pulse, tournament, report/dashboard живут в **главном процессе** локального `train` и являются библиотеками без транспорта (SP4 сделал их transport-free). Цель владельца: inference/сбор и обучение на разных машинах, динамика машин и агентов, простота деплоя, арена из разных архитектур.

### 5.2. Что брать

1. **Pull-протокол заданий («лизы матчей»)**: воркер просит `N` заданий; задание = `{uid, lineup (layout+seat assignments), seed, weight refs (agent, version), collect flags, ttl}`. Совпадает с `Lineup`/`SeatAssignment`/`schedule_lineups`, которые уже есть; прецеденты: TLeague, DI-star, HandyRL, AlphaStar [§2.1, §2.9, §2.10, §2.16].
2. **Отделить control plane от data plane**: хаб никогда не обрабатывает чанки; адрес learner'а агента берётся из реестра хаба, воркер шлёт чанки напрямую (TLeague, AlphaStar, KaiWu) — оставляет `TrajectoryService` как есть.
3. **Прокси/кэш на машине** (`Gather` HandyRL + кэши Menger): один процесс на машину тянет версионированные веса от хаба/weight store и раздаёт локальным воркерам (shared memory/файлы), заодно батчирует результаты и метрики — закрывает пункты SP5 «per-machine weight cache» и «shared-memory weight store».
4. **Лизы и heartbeat как в moolib**: узел регистрируется (`Register`), присылает `heartbeat` с TTL, который сам же задаёт; хаб считает узел живым, пока лиз не истёк; задания и подписки узла — под этим же лизом. Идентификатор узла вида `host:pid:slug` [§2.7].
5. **Идемпотентные результаты**: `ReportResults(batch)` по `MatchResult.uid` с дедупликацией в match log; доставка at-least-once; незавершённые задания после истечения лиза либо возвращаются в очередь (с тем же seed), либо молча списываются — выбор по типу (обучающие эпизоды можно списывать, tournament-блоки — повторять) [§4 п.4].
6. **Оценка как тип задания**: `kind ∈ {train, pulse, tournament_block}`; узел заявляет, какие kind умеет (CPU-узел — все, GPU-узел — ещё `learner`). Прецеденты: AlphaStar evaluators, OpenAI Five TrueSkill, HandyRL `eval_rate`. Позволяет без отдельной подсистемы распределить pulse и tournament по машинам.
7. **Learner-authoritative учёт бюджета**: `training.total_timesteps` считается по `consumed` на learner'ах (+ отчётам хаба), а не по локальным счётчикам воркеров — единая семантика бюджета (пункт SP5 «one budget semantics»); метрики rfps/cfps и staleness как у TLeague/OpenAI.
8. **Staleness**: два уровня — (а) V-trace как сейчас; (б) жёсткий `max_policy_lag` (версии learner'а минус `behavior_policy_version` при приёме) с режимами `drop` / `warn`; плюс метрика распределения лага в `train`-метриках и backpressure (ограниченные очереди, как RateLimiter/`maxsize`). Прецеденты: Sample Factory `max_policy_lag`, OpenAI staleness [§2.2, §2.4].
9. **Персистентное состояние хаба = то, что уже на диске**: `league_state.json`, `matches.db`, `ratings.json`, `checkpoints/…/meta.json`, `run_state.json`; хаб при старте загружает их (аналог `league_resume` DI-star, но без часовой потери — писать на каждый metrics tick, как делает SP4).
10. **Безопасность по минимуму**: токен-аутентификация в метаданных gRPC (pre-shared token в `--hub-token`/переменной окружения) и необязательный TLS; никакого pickle (уже так); проверка версии протокола и хэша `ColosseumConfig`. Для LAN/SSH-туннелей достаточно; открытый интернет не поддерживать. [И]

### 5.3. Чего избегать

- **Ray / Ray-подобного GCS** (владелец исключил; к тому же GCS — SPOF без внешнего Redis, §2.14).
- **DDP/Horovod между learners разных агентов** (сохраняем «один learner = один агент»; moolib и TLeague платят за ресинк групп).
- **Хаб как посредник данных** (DI-engine-координатор с replay buffer внутри — узкое место и SPOF для данных).
- **Статический граф процессов при запуске** (Launchpad/PolyBeast): противоречит приоритету динамического состава.
- **K8s-оператор как обязательный путь** (DI-orchestrator): K8s — слой поверх, а не основа.
- **Pickle поверх сокета** (HandyRL, moolib).
- **Автоматическое перераспределение агентов между learner-машинами на первом этапе** (никто, кроме DI-orchestrator Allocator и SRL scheduler, этого не делает; все остальные — ручное размещение). Начать с `--agent A` + проверка хабом и подсказкой «свободных GPU-узлов».

### 5.4. Три варианта архитектуры hub-and-nodes

Общие элементы всех вариантов (протокол v0, один proto-файл): `Register(node caps) → node_id, lease`, `Heartbeat(node_id, load, held versions) → commands`, `LeaseJobs(node_id, n, kinds) → [Job]`, `ReportResults(batch)` (идемпотентно по `uid`), `AnnounceVersion(agent, version, blob_ref)` (от learner'а), `GetWeights(agent, version)` (с кэшем машины), `Drain(node_id)`/`Deregister`. Узлы заявляют `capabilities {cpu_cores, gpus, memory, tags, kinds}`.

**Вариант A. «Hub = расширенный главный процесс» (минимальная миграция).**
- Главный процесс `colosseum train --hub :PORT` (как сегодня) поднимает gRPC-сервер хаба; локальные дочерние процессы становятся «узлом по умолчанию», а удалённые узлы (`colosseum node --hub host:PORT --role workers|learner|evaluator`) подключаются теми же вызовами. Лига, рейтинги, match log, pulse, report остаются как есть (в одном процессе, без сетевого слоя), удалённые воркеры выглядят для `RolloutLoop`/matchmaker как ещё один пул.
- Плюсы: наименьший риск и объём работ (SP4-библиотеки не меняются, `check_distributed_scope` снимается почти без изменения семантики), одиночный режим не меняется (приоритет 2), единый `run dir`, простой деплой (`ssh host 'colosseum node --hub ...'`).
- Минусы: хаб = главный процесс (SPOF, но у нас и сейчас так; падение → resume из каталога прогона); узел присоединяется только к живому прогону; нагрузка на GIL главного процесса (результаты, рефиты BT, метрики) растёт с числом узлов — нужно измерить (в CLAUDE.md помечено «not measured»); не подходит для «хаб обслуживает несколько независимых прогонов».
- Прецеденты: RLlib-лига (callback в процессе алгоритма), HandyRL (learner = хаб), DI-star (`League` внутри одного процесса).

**Вариант B. «Standalone hub с персистентным состоянием» (TLeague / DI-star / AlphaStar-координатор).**
- Отдельная служба `colosseum hub` владеет: реестром узлов и лизами, очередью заданий и матчмейкером (`BaseMatchmaker` + PFSP-таблицы), реестром версий/блобов (метаданные + ссылки на хранилище), match log, рейтингами, `league_state`; пишет в `runs/<name>/` (SQLite WAL). Learners и воркеры — узлы; главный `train` становится тонким клиентом/запускалкой. Хаб можно перезапустить: узлы переподключаются (лиз истёк → `Register` заново), состояние поднимается с диска.
- Плюсы: чистое разделение; узлы и сами прогоны пережимают падение главной машины; очевидная точка для `colosseum dashboard`/`report`; легко сделать несколько прогонов на одном хабе позже; естественное место для distributed tournament/pulse.
- Минусы: больше работы (вынос лиги/рейтингов/журнала в сервис, схема миграции, консистентность при двойном хозяине), риск регрессий по SP3/SP4-фичам, перенос «горячих» путей (результаты матчей 4 000+/с) через сеть → нужны батчи и бэкофф (урок FX1); хаб остаётся SPOF для заданий (но без потери данных).
- Прецеденты: TLeague `LeagueMgr`, DI-star `League` + `Coordinator`, AlphaStar coordinator, MALib CTD.

**Вариант C. «K8s-нативный» (TLeague на K8s + DI-orchestrator).**
- Хаб — Deployment/StatefulSet с PVC; learner — StatefulSet/Job на GPU-нодах; воркеры — Deployment с HPA по глубине очереди заданий или по rfps/cfps; веса — S3/MinIO или общая PVC; сервисное обнаружение — DNS Service; конфиг/кластер — Helm/jinja2 + kubectl.
- Плюсы: перезапуск actors, автоскейл, размещение (`nodeSelector`, gang) «из коробки»; соответствует `deployment/k8s` и Dockerfile'ам в репозитории.
- Минусы: самая тяжёлая операционка для маленькой команды (1 GPU-машина + пара CPU-коробок — K8s overkill), отладка сложнее; всё равно требует хаба варианта B (K8s не знает про лигу). Вариант C — не отдельная архитектура, а **способ запуска варианта B**.
- Прецеденты: TLeague (§2.10), DI-orchestrator (§2.9), Menger/Borg (§2.12).

**Сравнение (для владельца):**

| Критерий | A: hub = main | B: standalone hub | C: K8s-нативный |
|---|---|---|---|
| Объём работ SP5 | малый | средний-большой | большой (поверх B) |
| Изменение одиночного режима | почти нет | заметное (общий код лиги) | как B |
| Динамические узлы | да (присоединение к живому прогону) | да + переживает рестарт хаба | да + автоскейл |
| SPOF | главный процесс (resume с диска) | хаб (resume с диска, узлы переподключаются) | хаб (K8s перезапускает) |
| Подходит под «1 GPU-машина + CPU-ящики» | отлично | хорошо | избыточно |
| Несколько прогонов на одном хабе | нет | возможно | да |
| Совместимость с SP4-библиотеками | полная | полная, но с переносом в сервис | как B |
| Риск | низкий | средний | высокий |

### 5.5. Рекомендация и порядок работ [И]

1. Зафиксировать **протокол** (общие элементы выше) как отдельный proto-пакет с версией и валидатором хэша конфига. Никакой логики лиги в протоколе: только задания, результаты, версии, лизы.
2. Реализовать **вариант A** как первый релиз: он сразу даёт приоритет 1 (inference/сбор и обучение на разных машинах), динамику воркеров и снятие большинства отказов `check_distributed_scope` (скриптованные/замороженные, match log, рейтинги, pulse/tournament как типы заданий) при минимуме переписывания.
3. Параллельно сделать **per-machine кэш весов + пачки результатов** (HandyRL `Gather`, Menger-кэш) и **learner-authoritative бюджет**; это независимо от выбора A/B и снимает три пункта списка «Open items → SP5».
4. Заложить **вынос в вариант B** без изменения протокола: все обращения узлов идут через клиентский интерфейс, который в варианте A реализован внутри процесса, а в B — по gRPC. Решение о переходе на B принимать по трём измерениям: нагрузка хаба на главный процесс (GIL), частота падений главной машины, потребность в нескольких прогонах на одном хабе.
5. K8s — **только упаковка** (образы, манифесты в `deployment/`), после того как A/B работает на `docker compose` и ssh; автоскейл по метрикам rfps/cfps/глубине очереди — после.
6. Динамика агентов: добавление агента на лету = новая запись в реестре агентов (конфиг + архитектура) + свободный learner-узел, подтверждающий `claim`; воркеры узнают о новом агенте из следующих заданий (spec подтягивается по хэшу). Удаление — `Drain` (прекратить выдавать задания с этим агентом, дождаться возврата). Это соответствует тому, что TLeague делает перезапуском, а мы можем без остановки, пока «один learner = один агент».

### 5.6. Что нужно измерить до детального дизайна [И]

- Поток результатов и метрик хаба при 10–50 воркерах (результатов/с, пачка, задержка); нагрузка на GIL главного процесса с рефитами BT и журнальным писателем.
- Стоимость раздачи весов: размер `state_dict` примерных моделей × число воркеров × частота `weight_sync_interval_sec`; эффект кэша на машину (Menger — 4×; наш эффект неизвестен).
- Размер лиза и пачки заданий (HandyRL берёт `1 + n_workers // 4`; у AlphaStar, DI-star — по эпизоду): как это влияет на латентность реакции матчмейкера на новую версию.
- Накладные расходы gRPC-стрима чанков при мультимашинной сети (10 Гбит/с против 1 Гбит/с): в `CLAUDE.md` оценка 300 МБ/с для 100 воркеров — проверить на реальной линии.
- Поведение при потере узла посреди эпизода: сколько данных теряется, не искажает ли PFSP (потеря длинных игр смещает статистику в сторону коротких) [И, гипотеза].

---

## 6. Реестр непроверенных или частично проверенных утверждений

- AlphaStar «almost 900 distinct players»: из пересказа (поисковая выдача), в прочитанном Methods фразы нет [S].
- MALib: все детали из ar5iv-пересказа; «>40K FPS» в аннотации и искажённый «4040K» в тексте; разница 5×/3× — самоотчёт [S].
- HoK Arena «800 million samples per …»: фраза обрывается в прочитанном тексте, единица времени не установлена; использовать только «≈4.34 млн выборок/ч на одной 10-ядерной машине» [F, частично].
- Лига RLlib (`self_play_league_based_with_open_spiel`): описание по зеркалам репозитория, пример на старом API; жалоба на «stops learning» — отдельный пользовательский отчёт на форуме Ray [S].
- DI-engine: описание `Coordinator` — legacy «parallel mode»; новый режим (`ditask`) отличается; DI-orchestrator v2 интегрировал coordinator/collector/learner в один worker (по `architecture.md`) [F, но версия кода может отличаться от описываемой].
- SRL «Slurm»: только строка благодарности в статье [F/слабое].
- Отсутствие аутентификации у TLeague/Ray и других: **не проверялось** по документации безопасности; утверждения об отсутствии auth даны только там, где виден код (HandyRL, DI-star Flask, moolib pickle) [F по коду], в остальных случаях это [И].
- Для всех «—» в таблицах: деталь не найдена в прочитанных источниках, а не доказанное отсутствие.

## 7. Источники

- AlphaStar: https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf ; https://www.nature.com/articles/s41586-019-1724-z
- OpenAI Five: https://cdn.openai.com/dota-2.pdf ; Dactyl / Rapid: https://arxiv.org/abs/1808.00177
- SEED RL: https://arxiv.org/abs/1910.06591
- Sample Factory: https://arxiv.org/abs/2006.11751 ; https://www.samplefactory.dev/07-advanced-topics/pbt/ ; https://www.samplefactory.dev/07-advanced-topics/multi-policy-training/ ; https://github.com/alex-petrenko/sample-factory/blob/master/sample_factory/cfg/cfg.py
- Podracer: https://arxiv.org/abs/2104.06272
- Acme: https://arxiv.org/abs/2006.00979 ; Launchpad: https://arxiv.org/abs/2106.04516 ; Reverb: https://arxiv.org/abs/2102.04736
- moolib: https://research.facebook.com/file/268346465322053/moolib--A-Platform-for-Distributed-RL.pdf ; https://github.com/facebookresearch/moolib ; https://github.com/facebookresearch/moolib/blob/main/src/broker.h
- TorchBeast: https://arxiv.org/abs/1910.03552 ; https://github.com/facebookresearch/torchbeast
- DI-engine: https://github.com/opendilab/DI-engine ; https://di-engine-docs.readthedocs.io/en/latest/03_system/distributed.html ; DI-orchestrator: https://github.com/opendilab/DI-orchestrator ; DI-star: https://github.com/opendilab/DI-star
- TLeague: https://arxiv.org/abs/2011.12895 ; https://github.com/tencent-ailab/tleague_projpage ; TStarBot-X: https://arxiv.org/abs/2011.13729
- MALib: https://arxiv.org/abs/2106.07551 ; https://jmlr.org/papers/v24/22-0169.html
- Tencent HoK: https://arxiv.org/abs/2011.12692 ; https://arxiv.org/abs/1912.09729 ; https://arxiv.org/abs/2209.08483
- Menger: https://research.google/blog/massively-large-scale-distributed-reinforcement-learning-with-menger/
- SRL: https://arxiv.org/abs/2306.16688
- Ray RLlib / Core: https://docs.ray.io/en/latest/rllib/rllib-fault-tolerance.html ; https://docs.ray.io/en/latest/ray-core/fault_tolerance/gcs.html ; https://docs.ray.io/en/latest/rllib/key-concepts.html
- PufferLib: https://arxiv.org/abs/2406.12905
- HandyRL: https://github.com/DeNA/HandyRL ; https://github.com/DeNA/HandyRL/blob/master/docs/large_scale_training.md
- Lux AI: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021 ; https://arxiv.org/abs/2304.13004
- Внутренние: `research/01_distributed_rl_architectures.md`, `research/02_competitive_rl_framework_research.md`, `research/sp4_selection_practice.md`, `proto/colosseum.proto`, `src/colosseum/distributed.py`, `CLAUDE.md`.

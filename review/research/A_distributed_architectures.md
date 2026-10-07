# R1. Распределённые RL-архитектуры: обзор 2023–2026 и оценка архитектуры Colosseum

Дата среза: 2026-10-07. Автор: исследовательский агент (Claude). Язык: русский, термины английские.

Метки достоверности, которые используются в тексте:

- **[подтв.]** — цифра или утверждение прочитаны мной (или субагентом) в первоисточнике: arXiv-PDF (скачан и прочитан через `pdftotext`), README/исходники на GitHub, официальные доки.
- **[док.]** — из страницы документации, пересказанной WebFetch (маленькой моделью). Возможны неточности, ключевые цифры по возможности перепроверены.
- **[LOCAL]** — измерено мной/субагентом на loopback этой машины (WSL2, 8 vCPU, Python 3.12, grpcio 1.84, pyzmq 27.2, lz4 4.4.5, zstandard 0.25, safetensors 0.8, torch 2.14 CPU). Это не сеть: нет NIC, RTT, MTU.
- **[не подтв.]** — только вторичный источник/сниппет поиска/моё умозаключение без измерений.
- **[устарело]** — проект архивирован или давно не обновлялся.
- **[вывод]** — моё рассуждение, а не факт из источника.

---

## 0. Резюме (главное для владельца)

1. **Базовый выбор Colosseum (IMPALA-style: CPU-воркеры с локальным инференсом, асинхронные лёрнеры, V-trace/APPO, pull-веса) подтверждается литературой для вашего класса задач** (маленькие сети в МБ-масштабе, соревновательные игры). Центральный инференс (SEED RL) выигрывает только при дорогом инференсе и быстром ускорителе: в самой статье SEED на P100 дал 0.63x от IMPALA, а выигрыш 2.5x–80x получен на TPU v3 [подтв.: arXiv 1910.06591, Table 1]. OpenAI Five использовал отдельные Forward-Pass GPU, но с LSTM на 159M параметров [подтв.: arXiv 1912.06680].
2. **Для соревновательных игр одна мощная машина — норма, мультимашинность нужна реже, чем кажется.** Победитель Lux AI S3 обучался около 8 суток на одной рабочей станции; победители Neural MMO 2023 уложились в 8 часов на одной RTX 4090 [подтв. субагентом по write-up/статье]. Sample Factory держит ~130–146K FPS на одной 36-ядерной машине с одной GPU, PufferLib 3.0 заявляет 4M sps, 4.0 — 15–20M sps на одной RTX 5090 (на C-средах). Мультимашинность окупается в трёх случаях: (a) среда медленная (Python, десятки–тысячи шагов/с на ядро), (b) лига из многих обучаемых агентов с разными сетями (каждому нужен свой лёрнер/GPU), (c) большой бюджет self-play (AlphaStar, OpenAI Five). Именно (a)+(b) соответствуют вашему продукту, так что multi-machine — осмысленная фича, но **single-machine fast path должен оставаться первоклассным**.
3. **Самые ценные идеи для Colosseum приходят из LLM-RL, а не из игровых фреймворков: версия весов в каждом чанке + ограничение staleness по версиям (rate limit) + отбрасывание данных старше k версий + decoupled/proximal PPO objective** (AReaL, ROLL Flash, Prime-RL, StaleFlow, A-3PO). В текущем коде поле `behavior_policy_version` в `TrajectoryChunk` есть, но ни лёрнер, ни воркер его не используют для контроля staleness (grep по `learner.py`, `rollout_worker.py`, `appo.py`: фильтра нет).
4. **Транспорт: gRPC на Python годится, но с оговорками, и в текущей реализации есть конкретные потери.** [LOCAL] grpcio даёт 0.7–1.1 GB/s на поток для сообщений 4–32 MB, но всего ~226 MB/s на unary-вызовах по 0.3 MB (≈1.3 мс на вызов), а дробление на кадры по 64 KB убивает пропускную способность (60–170 MB/s). Текущий `GRPCTransport.send_chunk` открывает новый stream на каждый чанк (~300 KB) — это худший режим. `torch.save` для чанков в 5–10 раз медленнее pickle5/raw-буферов на малых объектах (715 MB/s на 0.27 MB [LOCAL]). lz4 на fp32-весах даёт ratio 1.00 [LOCAL] — сжимать веса нет смысла, лучше fp16/bf16 (2x) или delta+int8.
5. **Главный пробел относительно ваших приоритетов — не транспорт, а control plane**: нет регистрации нод, capability-based размещения ролей, heartbeat/lease, version-gate кода среды, распределённого координатора (в `distributed.py` прямо сказано, что PFSP/исторические оппоненты работают только на одной машине). Лучшие паттерны: «агент-нода сама регистрируется у координатора» (Nomad client, GitHub runner, Buildkite agent, k3s join, Ray `ray start --address`), join-token, исходящие соединения, lease, drain, content-hash пакет кода.
6. **Не стоит**: возвращаться к Ray; брать torch.distributed.rpc/TensorPipe (в режиме maintenance, CVE-2024-5480); NATS как data plane для тензоров (лимит payload 1 MB по умолчанию, ≤64 MB максимум); Redis как data plane для весов; torchrun elastic (рестарт всех при смене состава); gang scheduling (нужен только DDP-лёрнеру).

Приоритизированный список изменений — в разделе 7.

---

## 1. Обзор систем и архитектур

### 1.1 Классика: IMPALA, SEED RL, Podracer, Cleanba

**IMPALA** (Espeholt et al., 2018, arXiv 1802.01561): акторы с локальным инференсом тянут параметры перед каждым unroll, шлют траектории лёрнеру, V-trace корректирует лаг. DMLab: single-machine 24K FPS (48 CPU), распределённо 80K FPS (150 CPU), 200K (375 CPU), **250K FPS (500 CPU, batch 128)** [подтв., цифры также цитирует Sample Factory].

**SEED RL** (arXiv 1910.06591, ICLR 2020; репозиторий архивирован [устарело]):
- Инференс переносится на лёрнер (TPU/GPU); акторы только шагают среду и стримят obs по gRPC streaming (соединение держится открытым, метаданные один раз; на одной машине — unix domain sockets). Модель в единственной копии → нет pull весов, нет policy lag между воркерами.
- Цифры (DMLab, Table 1) [подтв., прочитано в PDF]: IMPALA (P100) 30K FPS; SEED на **той же P100 — 19K (0.63x)**; SEED TPU v3 2 ядра — 74K (2.5x); 8 ядер — 330K (11x); 64 ядра, 12 480 envs, 4 160 actor-CPU — **2.4M FPS (80x)**. Экономия стоимости 40–80% (DMLab, 1B фреймов ≈ $90 IMPALA против ≈ $25 SEED; ratio ~4).
- Арифметика пропускной способности из сноски статьи: при 100 000 obs/с (96×72×3 B), траектории 20 шагов и модели 30 MB IMPALA-схема требует **148 GB/s** (из-за постоянной рассылки весов), а передача наблюдений — 2 GB/s [подтв.]. Это аргумент в пользу SEED только при крайне частом pull весов; у вас pull раз в несколько секунд, так что аргумент ослабевает.
- Когда оправдан [вывод по статье]: инференс дорог (большая сеть), среда лёгкая, есть быстрый ускоритель и интерконнект. Недостаток: один узел инференса/лёрнера — узкое место (SRL: 32 CPU-ядра «легко перегружают единственный GPU лёрнер SEED»).

**Podracer** (Hessel et al., 2021, arXiv 2104.06272) [подтв.]:
- **Anakin**: среда на JAX, весь цикл act+learn внутри `jit`/`pmap`. >5M шагов/с на 8-ядерном TPU (малые сети, grid-world); >3M шагов/с на 16 ядрах в meta-RL (~$100 за 24 ч на preemptible).
- **Sebulba**: среды на CPU хоста, 8 TPU-ядер делятся на actor- и learner-ядра (часто 3x больше learner-ядер), инференс батчем на TPU. 200K FPS на 8 ядрах (actor batch 128), **43M FPS на 2048 ядрах** (Pong меньше минуты). Авторы предупреждают: наивный рост batch ради throughput сильно снижает data efficiency.

**Cleanba** (Huang et al., ICLR 2024, arXiv 2310.00036) [подтв.]: JAX-платформа, Sebulba-подобная, детерминированная схема actor/learner split. Главное для нас: **асинхронность не бесплатна для PPO** (PPO делает 16 градиентных шагов на rollout → падает data efficiency), а для IMPALA разницы нет. IMPALA-версия Cleanba: 1 A100 + 10 CPU — в 6.8x быстрее monobeast и 1.2x быстрее moolib; 8 A100 + 40 CPU — 5x и 2x. Это подтверждает формулу из CLAUDE.md («APPO = PPO + V-trace»), но предупреждает: проверять data efficiency APPO при росте лага.

### 1.2 Sample Factory 2.x

Источники: arXiv 2006.11751 (ICML 2020), www.samplefactory.dev, github.com/alex-petrenko/sample-factory.

- **Архитектура**: rollout workers (среды), inference (policy) workers, learner, batcher; обмен через shared-memory тензоры и очереди индексов; **double-buffered sampling** (k сред на воркер делятся на две группы, пока GPU считает действия для одной, CPU шагает другую) [подтв.].
- **Цифры (paper)** [подтв.]: «до 130 000 FPS на одной multi-core машине с одной GPU» (abstract); Table A.3: VizDoom Battle PBT, 72 воркера, 2304 env, **146–154K FPS** независимо от числа агентов (4/8/12); в сравнении IMPALA приводится 24K FPS на 48 ядрах и 250K FPS на кластере 500 CPU. Субагент видел в таблице 146 551 FPS (VizDoom), 135 893 (Atari), 42 149 (DMLab) на 36-ядерной CPU + RTX 2080 Ti.
- **Сериализации нет**: параметры в shared GPU memory, обновление у policy worker «менее 1 мс»; собственная C++ очередь `faster-fifo` (стандартный `multiprocessing.Queue` становится узким местом выше ~10^5 FPS) [подтв. по PDF].
- **Policy lag** [док.: samplefactory.dev/07-advanced-topics/policy-lag]: измеряется в SGD-шагах (метрики `train/version_diff_{min,avg,max}`); формула оценки `lag ≈ (num_epochs · num_workers · num_envs_per_worker · agents_per_env · rollout) / batch_size`; рекомендация держать lag **ниже 20–30 SGD-шагов**; RNN и сложные action spaces чувствительнее; режим «нулевого лага» `--async_rl=False --num_batches_per_epoch=1 --num_epochs=1` превращает PPO в A2C. В конфигах моделей на HF присутствует параметр `max_policy_lag` (отбрасывание данных старше порога по версии) [подтв. наличие ключа в cfg.json на HuggingFace; дефолт и точная семантика мной не проверены]. В статье наблюдаемый lag в экспериментах 5–10 SGD-шагов.
- **Multi-agent/self-play/PBT** [док.]: «single- & multi-agent training, self-play, multiple policies at once on one or many GPUs», PBT; все среды трактуются как multi-agent (single-agent оборачивается автоматически); multi-agent среды обязаны делать auto-reset по агентам; есть «inactive agents». Self-play и PBT продемонстрированы на мультиплеерной Doom (статья).
- **Multi-node**: статья позиционирует систему как **single-machine** («optimized for a single-machine scenario»). Доки упоминают multi-node/Slurm, но детали (как синхронизируются лёрнеры и воркеры между узлами) я не подтвердил [не подтв.].
- **Статус**: PyPI `sample-factory` 2.1.1 от 2023-06; репозиторий жив (коммит 2026-10-01, по данным субагента).
- **Для Colosseum**: SF — эталон single-node производительности и double-buffering. League/PFSP «из коробки» нет; PBT есть.

### 1.3 PufferLib (2.x → 3.0 → 4.0 → 5.0-experiments)

Источники: puffer.ai/blog/{2.0,3.0,4.0,engineering-4.0}, arXiv 2406.12905 и RLC 2025 paper «PufferLib 2.0: RL at 1M steps/s», GitHub releases.

| Версия | Что изменилось | Цифры |
|---|---|---|
| 1.0 | стабильный API, эмуляция Gymnasium/PettingZoo | — |
| 2.0 | PufferEnv/C-среды, векторизация: shared memory, EnvPool-подобный async, «no observation copies»; native multiagent | среды >1M sps на одном ядре, PPO-демо 300K–1.2M sps на одной RTX 4090 [подтв., paper + blog] |
| 3.0 | новый трейнер (default hyperparams решают то, что 2.0 не решало за 200 запусков), Protein (тюнинг гиперпараметров), PufferEnv C API, 22 Ocean-среды | **4M sps на одной RTX 5090** [подтв., blog 3.0] |
| 4.0 | PyTorch заменён 5 000 строк детерминированного CUDA-C (torch как fallback), MinGRU вместо LSTM, highway-связи вместо нормализации, OpenMP-потоки на чанках сред, асинхронные rollout-воркеры на отдельных CUDA streams | **15M sps (стандартная модель), >20M (малые) на одной RTX 5090** [подтв., blog 4.0]; «+2M sps» от async rollout, pinned memory вдвое сократила H2D-издержки [подтв., engineering-4.0] |
| 5.0 | на GitHub есть релиз «5.0 Experiments» (13 сентября); docs-страница сообщает «PufferLib 5.0, up to 60M step/second training in ~10k lines of CUDA C» | **60M sps — [не подтв.]**: цифра получена через summarizer страницы docs, независимого поста/измерения не нашёл; в блоге список заканчивается на 4.0 |

Что важно для Colosseum:
- **Весь дизайн PufferLib — «одна машина, быстрые среды на C, маленькие сети, векторизация в потоках внутри одного процесса»**. Multi-node отсутствует (в README/доках не упоминается) [не подтв. отсутствие, но подтв. отсутствие упоминаний]. Multi-GPU упомянут только для sweep (`--sweep.gpus`).
- **Multi-agent: native** (агенты как часть батча; fixed число агентов на env с padding), Ocean-среды Battle/MOBA/NMMO3/Slime Volleyball поддерживают competitive-режимы [док. по вторичной выдаче]. **Self-play/league/PFSP как фича фреймворка я не нашёл** [не подтв.].
- Урок: узкие места на масштабе — память/копии/Python-overhead, а не сеть; «pinned memory + async rollout + статические буферы» дали больше, чем распределение. Применимо к вашему локальному fast path.
- Neural MMO 3: «петабайт сжатых данных (12 000 лет игры) на одном сервере» [подтв. по блогу game-rl].
- Для Python-сред уровня Kaggle (медленные) PufferLib-ускорения C-сред не применимы напрямую; применимы идеи векторизации (shared-memory буферы, EnvPool-async, double-buffer).

### 1.4 RLlib (new API stack), TorchRL, moolib, Acme/Launchpad, SRL, TLeague, EnvPool

**RLlib (Ray 2.59.0, 2026-10-02)** [подтв. по исходникам и релизам, ссылки в разделе 9]:
- Новый стек по умолчанию с Ray 2.39–2.40 (PPO — 2.39; APPO/IMPALA — 2.40); старый стек формально не удалён, помечен `@OldAPIStack`, даты удаления не нашёл [не подтв.].
- Компоненты: `EnvRunner` (Ray-акторы; инференс локально: forward RLModule на CPU или GPU), `ConnectorV2` (env-to-module, module-to-env, learner pipeline; доки называют work in progress), `RLModule/MultiRLModule`, `LearnerGroup` (акторы с torch DDP; модель обязана помещаться в одну GPU), опциональные `AggregatorActors` (собирают батчи, GAE/V-trace, загрузка на GPU).
- Веса: pull-модель через глобальный актор `EnvRunnerStateServer` (default `use_env_runner_state_server=True`): Algorithm раз в `broadcast_interval` пушит ObjectRef на merged state с `WEIGHTS_SEQ_NO`, EnvRunner в начале `sample()` делает `pull_if_newer(weights_seq_no)`. Класс помечен `@DeveloperAPI(stability="alpha")`. `WEIGHTS_SEQ_NO` пишется в `extra_model_output` каждого сэмпла (можно мерить лаг). Backpressure: `max_requests_in_flight_per_env_runner=1`, `..._per_aggregator_actor=3`.
- Fault tolerance: `restart_failed_env_runners=True`, `delay_between_env_runner_restarts_s=60`, `env_runner_restore_timeout_s=1800`; после рестарта EnvRunner принудительно получает последние веса; с 2.42–2.43 улучшена толерантность к node failure и spot preemption. Динамически добавлять агентов в работающий эксперимент штатно нельзя [не подтв. отсутствие].
- Multi-agent: `MultiAgentEnv`, `policy_mapping_fn`, `MultiRLModuleSpec`. **League-пример** (`self_play_league_based_with_open_spiel.py`) — main/main-exploiter/league-exploiter по AlphaStar, но это **один Algorithm на одном драйвере с одним MultiRLModule**, а матчмейкинг реализован пересозданием `policy_mapping_fn` при каждом добавлении в лигу. Независимые агенты с разными архитектурами/алгоритмами/лёрнерами — это то, что делаете вы.
- Цифры: единственная найденная — **100K ts/s на Atari с 400 EnvRunner и 16 multi-node GPU Learners** (release notes 2.43). Блог Anyscale с бенчмарками не найден. Открытый баг про замедление APPO на >1 GPU (issue #50221), актуальность не проверена [не подтв.]. SRL измерял RLlib в 6.3–21.6x медленнее своей системы (старая версия RLlib).
- Расхождение: release 2.43 заявляет векторизацию multi-agent на новом стеке, а доки пишут, что она пока не поддерживается — проверять руками.

**TorchRL (v0.14.0, 2026-09-10)** [подтв. по доке/релизам]: коллекторы `Collector`, `MultiCollector` (sync/async), `AsyncBatchedCollector`, `DistributedCollector` (torch.distributed, gloo/nccl/mpi/ucc, launcher submitit/mp), `RPCCollector`, `RayCollector`. Weight sync — **push, инициирует отправитель** (`SharedMemWeightSyncScheme`, `MultiProcess…`, `Distributed…`, `RPC…`, `Ray…`); staleness: `update_after_each_batch`, `max_weight_update_interval`. С 0.12–0.14: auto-batching inference server (threading/multiprocessing/Ray/Monarch), `ProcessInferenceServer`, `PolicyClientModule`, remote learners. Есть `PolicyAgeFilter(max_policy_lag)` — отбрасывание по версии (док.). Лиги/PFSP/self-play-оркестрации нет; fault tolerance в доках distributed collectors не описана. Это набор блоков, а не система.

**moolib** (Meta, 2022): peers-группа с all-reduce градиентов, elasticity через раздачу state лидером, RPC поверх TensorPipe (авто-выбор shm/IB/TCP) [подтв.]. **Репозиторий архивирован 2023-10-31 [устарело]**. Cleanba показал, что кривые обучения moolib зависят от аппаратной конфигурации.

**Acme/Reverb/Launchpad**: **Launchpad архивирован 2024-10-25**, последний релиз 2022 [устарело]; Acme — последний релиз dm-acme 0.4.0 от 2022-02; Reverb 0.14.0 (2023-12). SRL исключил Acme из сравнений из-за низкой производительности. Брать как идеи (Reverb как replay-сервис), не как зависимость.

**SRL — «ReaLly Scalable RL»** (Mei et al., ICLR 2024, arXiv 2306.16688; код `openpsi-project/srl`, заморожен с 2024-02) [подтв. субагентом по статье]:
- Три типа воркеров: **actor workers** (CPU, симуляция, multi-agent: разные агенты — через разные streams, «environment rings»), **policy workers** (батчевый инференс на GPU/CPU, динамический batch по размеру/таймауту), **trainer workers** (DDP, prefetch).
- Streams: inference streams (obs↔actions) — shared memory на узле, сокеты между узлами; sample streams (траектории) — FIFO, zero-copy на GPU; сокеты поддерживают lz4. Веса: trainers пушат на parameter server на NFS, policy workers регулярно pull'ят; явных staleness-порогов нет.
- Fault tolerance: динамический батч переживает падение actor worker; controller следит за воркерами.
- Цифры: в 6.3–21.6x быстрее RLlib; Atari 644K FPS на 800 ядрах; DMLab 741K на 1600; gFootball 89K на 3200; SMAC 17K на 1280; **максимум 15 000+ CPU-ядер и 32 A100**; Hide-and-Seek в 3x быстрее «Rapid» OpenAI с CPU-инференсом и до 5x с GPU-инференсом (сравнение с опубликованными числами).
- **Ближайший по духу аналог Colosseum** с точки зрения ролей (actor / policy / trainer) и multi-agent/PBT; ключевой урок — отделение policy workers как опциональной роли.

**TLeague** (Tencent AI Lab, arXiv 2011.12895, 2020) [подтв. по PDF]: Actor–Learner–InferenceServer + **LeagueMgr/GameMgr** (payoff matrix, алгоритмы выбора оппонентов), **ModelPool** (хранит параметры пула оппонентов; до M_M реплик с балансировкой, в памяти), HyperMgr; **InfServer опционален** (батчевый инференс на GPU «может дать больший throughput, чем batch-1 инференс на каждом Actor»); RPC на **ZeroMQ** с собственным протоколом на Python; K8s-развёртывание (LeagueMgr, ModelPool, Learner, InfServer — k8s-ресурсы); лёрнеры синхронизируются Horovod (allreduce), 7 Learner + 1 InfServer должны быть на одной GPU-машине. Throughput-таблица для Dota/SC2/Quake/ViZDoom/Pommerman (rfps/cfps) есть в статье. Это самая близкая **по составу ролей** система к Colosseum (лига + пул моделей + опциональный inference server + k8s). Статус проекта сегодня не проверен [не подтв.].

**EnvPool** (sail-sg): C++ пул CPU-сред, sync/async. На DGX-A100 (256 ядер): Atari 1.07M FPS, MuJoCo 3.13M FPS; на 32 ядрах: 200K / 582K; Sample Factory на тех же 256 ядрах: 707K / 1.57M [подтв., README]. Репозиторий активен (push 2026-09). Оборачивает только C++-среды.

### 1.5 GPU-среды и когда они уместны

| Проект | Цифры | Статус |
|---|---|---|
| Brax | `brax/envs` deprecated, поддерживается `brax/training`; рекомендуют MuJoCo Playground (MJX / MuJoCo Warp) | частично [устарело] |
| Pgx, gymnax, PureJaxRL (README: до 1000x против PyTorch-RL при параллельном запуске многих агентов) | Pgx последний push 2025-03; PureJaxRL 2024-09 | жив/заморожен |
| JaxMARL (arXiv 2311.10090) | ~14x wall-clock против существующих реализаций, до 12 500x при векторизованных прогонах | активен |
| Mava (InstaDeep) | MARL на JAX, Anakin/Sebulba | активен (2026-09) |
| Madrona/GPUDrive (arXiv 2408.01584) | пик 2.3M agent-steps/s (ASPS), управляемые агенты ~200K; Nocturne 15K ASPS | 2025 |
| Jux (Lux S2 на JAX) | A100: 363K шагов/с против 0.31K на CPU-версии (1166x) | сообщество |

Для **соревнований** GPU-среды обычно не подходят [выводы субагента по write-up победителей]: правила и баги движка меняются организаторами по ходу соревнования; feature engineering на JAX неудобен (победитель Lux AI S3 переписал и движок, и признаки на Rust, чтобы ловить расхождения с настоящим); узким местом оказывается не симулятор, а GPU-инференс+backward (оценка победителя: симулятор 110K шагов/с на 16-ядерном CPU, а обучение модели 10M параметров — ~430 шагов/с, всего ~300M шагов за ~8 суток на одной станции [не подтв. независимо; со слов субагента по write-up Frog Parade]). Neural MMO 2023: все победители в рамках 8 часов на одной GPU (4090), 15–22M шагов; основными рычагами были reward shaping и разнообразие карт, а не масштаб инфраструктуры.

### 1.6 Historical reference для лиг: OpenAI Five (Rapid), AlphaStar

**OpenAI Five** (arXiv 1912.06680) [подтв. субагентом по PDF; частично — по поиску]:
- Rollout-машины на CPU гоняют игру и **не** считают политику; отдельный пул Forward-Pass GPU (батч ~60) считает действия; Optimizer GPU (до 1536) усредняют градиенты по NCCL; **Controller на Redis** хранит параметры и метаданные; оптимизаторы публикуют версию каждые 32 шага, Forward-Pass GPU **тянут (pull)** свежую версию.
- Rollout шлёт данные каждые 256 шагов (~34 с игры). Ранняя версия слала целые эпизоды: данные устаревали на часы и градиенты были «часто бесполезными или разрушительными».
- **Staleness M−N** (версия оптимизатора минус версия данных): целевое 0–1, при ~8 версиях заметное замедление; sample reuse 2–3x замедляет обучение вдвое, 8x может не дать компетентной политики; «качество данных важнее объёма вычислений».
- Self-play: 80% против последней версии, 20% против прошлых. Конфигурация: 128 000 CPU-ядер (preemptible) и 256 P100 в ранней версии [вторичный источник], 1536 GPU на пике.

**AlphaStar**: лига main agents / main exploiters / league exploiters; на каждого агента 32 TPU v3, 44 дня [цифры из сниппета, Nature недоступен — не подтв.].

### 1.7 LLM-RL: rollout и training на разном железе

Субагент прочитал arXiv-PDF AReaL, PipelineRL, INTELLECT-2/3, LlamaRL, Laminar, ROLL Flash, StreamRL, StaleFlow, Magistral, Kimi k1.5/K2, ScaleRL, A-3PO, VCPO, SparseRL-Sync и документацию verl/OpenRLHF/slime/prime-rl/checkpoint-engine.

**AReaL** (arXiv 2505.24298, NeurIPS 2025) [подтв.]:
- Роли: interruptible rollout worker (SGLang), reward service, trainer (Megatron), rollout controller. Генерация и обучение на **разных GPU**.
- **Staleness-контроль**: параметр η — максимальная staleness в батче; rate-limit на постановку новых generate-запросов: `⌊(Nr−1)/B⌋ ≤ i + η` (Nr — число сгенерированных траекторий, B — размер батча, i — текущая версия политики). η=0 → синхронный режим. Данные используются один раз. η=4 (код), η=8 (математика).
- **Decoupled PPO objective**: лосс использует три политики: behavior (кто сэмплировал, для IS-веса `π_prox/π_behav`), proximal (недавняя, якорь trust region: клиппинг `π_θ/π_prox`), target. Абляция на 1.5B (AIME24): без decoupled при η=4 → 23.3, с decoupled → 42.2 (oracle η=0 → 42.0); при η=8: 35.7 → 41.0; η=16: 35.8 → 38.7.
- Результаты: до **2.77x** к синхронной системе на тех же GPU, линейное масштабирование до 512 GPU (64 узла × 8 H800), interruptible generation +12–17% throughput. Малый сетап (8 GPU, 1.5B): throughput 27.1K (η=0) → 47.8K (η=1) → 49.0K (η=4) → 52.0K (η=16) — **основной выигрыш уже при η=1**, дальше убывающая отдача.
- **A-3PO** (arXiv 2512.06547): заменяет явный forward для π_prox log-линейной интерполяцией `log π_prox = α·log π_behav + (1−α)·log π_θ`, α=1/d (d — staleness в версиях), до 1.8x ускорения на 1.5B/8B; версия behavior-политики хранится в данных [подтв.].
- Fault tolerance и эластичность в статье не описаны [не подтв.].

**Другие системы (краткая сводка, подробности в таблице раздела 6):**
- **verl** (HybridFlow, arXiv 2409.19256): colocated hybrid engine; recipes `one_step_off` (NCCL sync «<300 мс», +23–40% на Qwen2.5-Math-7B) и `fully_async` (Rollouter / MessageQueue / Trainer / ParameterSynchronizer; `staleness_threshold`, `partial_rollout`; Qwen2.5-7B, 128 GPU: 2.35x) [док.]. Delta weight sync: рассылаются только изменившиеся байты BF16 (1–3% байтов за шаг у dense), ускорение 1.3–1.5x (7B) … до 21x (72B+) [док.].
- **OpenRLHF**: Ray + vLLM, NCCL/CUDA IPC, async через ограниченную очередь `async_queue_size` (по умолчанию 1); документация предупреждает, что async не подходит для **малых моделей и sparse rewards** [док.] — релевантно для малых игровых сетей.
- **slime** (THUDM): Megatron + SGLang + Data Buffer, colocated/disaggregated, NCCL/shared FS/delta [док., README].
- **Prime-RL / INTELLECT-3** (arXiv 2512.16144) [подтв.]: orchestrator (лёгкий CPU-процесс) + FSDP2-trainer + vLLM inference; **`max_off_policy_steps`=8 — rollouts старше отбрасываются**; in-flight weight updates; masked token-level IS с границами [0.5, 5] вместо клиппинга; 60 узлов × 8 H200 (16 training + 44 inference).
- **INTELLECT-2** (arXiv 2505.07291) [подтв.]: **децентрализованный async RL на permissionless гетерогенной сети**; **SHARDCAST** — HTTP-дерево relay (nginx), шардированная конвейерная раздача (62 ГБ на все узлы в среднем 14 минут, ~590 Мбит/с), relay хранит 5 последних версий, клиент выбирает relay по `success rate × bandwidth`; **TOPLOC** проверяет rollouts недоверенных воркеров; two-step async. Для вас интересен как прецедент «произвольные ноды присоединяются к обучению».
- **PipelineRL** (arXiv 2509.19128): in-flight weight updates (пауза только на приём весов), `max_lag` throttling, Redis streams между стадиями, ~2x быстрее conventional RL (128 H100).
- **ROLL Flash** (arXiv 2510.11345): **async ratio α на уровне sample** (каждый sample должен начаться не старше n−α версий; буфер ≤ (1+α)·batch); α=2 даёт почти максимальный throughput; до 2.24x (RLVR), 2.72x (agentic).
- **Magistral** (arXiv 2506.10910): генераторы работают непрерывно, веса NCCL <5 с, `n_async/n_batch ≤ 2`.
- **Kimi k1.5/K2**: в основном **синхронный colocated** RL; partial rollouts; K2 checkpoint-engine: обновление параметров 1T-модели <30 с; MoonshotAI/checkpoint-engine: Broadcast и **P2P-режим для динамически подключаемых инстансов** (mooncake-transfer-engine, RDMA).
- **LlamaRL** (arXiv 2505.24034): DDMA (GPU→GPU по NVLink/IB), sync 7B 0.04 с / 70B 1.15 с / 405B 2.31 с против OpenRLHF 4.32 / 111.65 с; **AIPO** — IS-отношение обрезается сверху константой ρ∈[2,10] (ссылается на IMPALA).
- **Laminar** (arXiv 2510.12633): trajectory-level асинхронность, слой **relay-воркеров в CPU-памяти как распределённый parameter service** (trainer пушит мастер-relay и продолжает; rollout тянет у своего relay между траекториями), heartbeat-failover, dynamic repack; актор стоит на sync 0.64–1.40 с.
- **StreamRL** (arXiv 2504.15930): disaggregated, профилировщик балансирует ресурсы стадий, cross-datacenter sync 72B <10 с по 80 Гбит/с; до 2.66x; 1.33x cost-effectiveness в гетерогенном (H800+H20) сетапе.
- **StaleFlow** (arXiv 2601.12784): версия на траектории, ограничение `V_traj + η ≥ V_buf`, виртуальный staleness buffer (Reserve/Occupy/Consume), накладные расходы протокола <1%, 1.42–2.68x. Классификация: AReaL/verl-async/ROLL Flash — строгий контроль через лимит in-flight; LlamaRL/Laminar — без явных гарантий.
- **ScaleRL** (arXiv 2510.13786): PipelineRL-k эффективнее PPO-off-policy-k; оптимум k=8.
- **VCPO** (arXiv 2602.17616): масштабирует LR по ESS, заявлена стабильность при очень большом числе off-policy шагов; **SparseRL-Sync** (arXiv 2605.07330): ~100x сжатие дельт при BF16 [не подтв. на FP32 малых сетях].
- **Tinker (Thinking Machines)**: управляемый сервис, внутренности не раскрыты [не подтв.].

**Переносимо на игровой RL (вывод, детали в разделе 7):**
1. Версия весов в каждом чанке; считать staleness как разницу версий (OpenAI Five, AReaL, StaleFlow).
2. Rate-limit на «новые чанки» по версии (AReaL) = естественный backpressure без барьеров.
3. Отбрасывание старше k версий (Prime-RL k=8, TorchRL `PolicyAgeFilter`, SF `max_policy_lag`).
4. Proximal policy/decoupled objective (AReaL, A-3PO) вместо одного π_old; верхняя обрезка IS-веса (AIPO, IMPALA ρ̄).
5. Hierarchical weight distribution (Laminar relay, SHARDCAST) — актуально, когда воркеров сотни, а сеть узкая.
6. Метрики staleness/ESS/idle time как первоклассные.
7. **Оговорка**: оптимальные η/k из LLM (4–8) нельзя переносить: в OpenAI Five ~8 версий уже заметно вредили, SF рекомендует lag < 20–30 SGD-шагов (это другая единица), а OpenRLHF предупреждает о малых моделях и sparse rewards. Стартовать с η=1–2 и измерять.

---

## 2. Оси дизайна и компромиссы

### 2.1 Где делать инференс

| Вариант | Где применяется | Плюсы | Минусы |
|---|---|---|---|
| **Локально на CPU воркера** | IMPALA, Sample Factory (inference workers на CPU/GPU того же узла), RLlib EnvRunner, Colosseum сейчас | нет сетевой латентности на шаг; воркеры независимы, элегантная эластичность; не нужна GPU на воркере | дорогой для больших сетей (SEED: «CPU неэффективны для NN; время инференса перевешивает env step»); актор чередует env и NN, утилизация неровная; нужно пушить веса на каждый воркер |
| **Центральный GPU-сервер инференса** | SEED RL, OpenAI Five (Forward-Pass GPU), SRL policy workers, TLeague InfServer (опционально), TorchRL inference server | батчинг по сотням env → высокая утилизация GPU; одна копия модели, нет pull весов и policy lag; актор «тупой» | per-step RTT; единственный узел → узкое место (SRL: 32 ядра перегружают один GPU-лёрнер SEED); SEED на P100 был **0.63x** от IMPALA; нужна быстрая сеть (SEED: unix sockets на одной машине) |
| **GPU-воркер батчами (inference в том же процессе, что среды)** | Sample Factory (inference workers), PufferLib 4.0 (async rollout на CUDA streams), Podracer Sebulba | нет сети; батчинг по многим env одного процесса; хорошо для GPU-нод, которые тоже гоняют среды | GPU делится между средой/инференсом/лёрнером; требует достаточного числа env на процесс |

Правила выбора [вывод на основе SEED/SRL/SF/TLeague/OpenAI Five]:
- Сеть маленькая (MLP/небольшой CNN/GRU, ≤ единицы миллионов параметров), среды медленные или умеренные → **локальный CPU-инференс** (ваш дефолт). Чем медленнее среда относительно forward, тем меньше выигрыш от любого иного варианта.
- Сеть тяжёлая (десятки–сотни миллионов параметров, большие CNN/трансформеры; OpenAI Five LSTM-4096 159M) или среда очень лёгкая → инференс на GPU: либо batched-в-процессе на GPU-нодах, либо отдельный inference server при LAN-RTT ≲ 1 мс.
- Через WAN/VPN (Tailscale 10–50 мс RTT) центральный инференс по шагу неприменим; только локальный [вывод].
- SEED/SRL прячут латентность несколькими env на актор (SEED: 12–16 env на 4-CPU машину; SRL: environment rings; SF: double-buffer). Любой выбор требует ≥ 2×N независимых env в процессе.
- Опираться нужно на **калибровку на старте**: микробенчмарк `env_step_ms` и `forward_ms(batch)` на каждой ноде (раздел 7, S3), а не на догму.

**Локальный микробенчмарк CPU-инференса [LOCAL]** (1 поток torch 2.14 CPU, `inference_mode`, WSL2 8 vCPU, машина параллельно загружена другими задачами — шум, порядок величины надёжен; скрипт `/tmp/r1_bench/infer.py`). Время одного forward / на сэмпл:

| Сеть | B=1 | B=16 | B=64 | B=256 |
|---|---|---|---|---|
| MLP 256-256-256-64 (0.15M параметров) | 0.22 мс (222 мкс) | 0.65 мс (40 мкс) | 1.6 мс (25 мкс) | 5.7 мс (22 мкс) |
| Linear + GRU256 (0.5M) | 0.33 мс (325 мкс) | 0.93 мс (58 мкс) | 2.2 мс (34 мкс) | 8.7 мс (34 мкс) |
| MLP 512-1024-1024-64 (1.6M) | 2.3 мс (2307 мкс) | 8.2 мс (515 мкс) | 15.6 мс (244 мкс) | 27.6 мс (108 мкс) |
| CNN 3×conv (8×16×16 вход, 4.3M) | 3.4 мс (3352 мкс) | 47 мс (2949 мкс) | 137 мс (2135 мкс) | не мерялось |

Вывод [вывод]: батч по 16–64 env внутри процесса снижает стоимость на сэмпл в 5–10 раз (поэтому векторные среды на процесс обязательны). Малые MLP/GRU (≤0.5M) стоят 25–60 мкс/сэмпл при батче ≥16, то есть ≈ 15–40K шагов/с/ядро только на инференс — это дешевле почти любой Python-среды, и CPU-инференс не узкое место. Для 1–5M параметров (в особенности CNN) инференс стоит 0.1–3 мс/сэмпл, то есть 300–10K шагов/с/ядро — сопоставимо или хуже, чем шаг лёгкой среды; именно здесь GPU-инференс (batched-in-process на GPU-ноде или inference server) начинает окупаться. Цифры SEED по латентности (DMLab end-to-end 17.97 мс IMPALA против 10.98 мс SEED; Atari 7.2 мс) относятся к ResNet из IMPALA на другом железе [подтв. субагентом по PDF].

### 2.2 Синхронизация весов

| Способ | Примеры | Цифры/замечания |
|---|---|---|
| **Pull с версией** | OpenAI Five (Forward GPU опрашивают Redis раз в ~минуту), RLlib (`pull_if_newer(weights_seq_no)`), SRL (NFS parameter server), Laminar, StaleFlow, SF (shared memory) | воркер сам решает, когда тянуть (на границе чанка/эпизода); лёрнер не блокируется на новых/медленных нодах — **лучше всего подходит для эластичного набора CPU-нод** |
| **Push (broadcast)** | TorchRL weight schemes, verl/OpenRLHF/PipelineRL/Magistral NCCL, LlamaRL DDMA | для фиксированного набора GPU; у динамических CPU-нод NCCL не годится (фиксированная группа) [вывод] |
| **Иерархия/relay** | SHARDCAST (HTTP-дерево, relay хранит 5 версий), Laminar (relay в CPU-памяти), TLeague ModelPool (реплики), checkpoint-engine P2P | окупается при сотнях нод или узкой сети; INTELLECT-2: 62 ГБ за ~14 мин (~590 Мбит/с) на глобальной сети |
| **Дельты** | verl delta sync: 1–3% изменившихся байтов за шаг у BF16 dense, 1.3–21x; SparseRL-Sync ~100x | работает из-за BF16-квантования при малом LR; для FP32 малых сетей **не подтверждено**. [LOCAL, синтетика] fp16-дельта 2.0x, int8-дельта 4.0x (4.85x с zstd), ошибка ~1e-5 при 25M параметров |
| **Версионирование** | AReaL (`i`), StaleFlow (`V_traj`), RLlib (`WEIGHTS_SEQ_NO`), Prime-RL (step n), A-3PO | версия должна лежать **в данных**, не только в сторе |

Размеры и частоты (для планирования) [вывод + LOCAL]:
- Сеть 10 MB, 100 процессов-воркеров, pull раз в 5 с, **каждый процесс тянет сам**: 200 MB/s исходящих с weight store (1.6 Gbit/s) — Python-gRPC-сервер на одном потоке даёт ~0.8 GB/s [LOCAL], то есть вы сжигаете около четверти ядра и сеть впустую. С **per-node weight cache** (один pull на машину, раздача по shared memory) при 10 машинах: 20 MB/s. При fp16-передаче вдвое меньше.
- Для сетей 5–50 MB на 10 Gbit передача занимает 4–40 мс — на порядки меньше времени чанка. Сжатие весов не нужно (lz4 на fp32-весах: ratio 1.00; zstd-1: 1.08; blosc2 shuffle+zstd: 1.18 на 51 MB/s [LOCAL]).
- Текущая реализация (`GRPCWeightSource.get_nowait`): на каждый опрос — `get_version` RPC и затем отдельный `get` RPC (гонка между ними, лишний round-trip) + `torch.save`+`lz4.frame` на лёрнере. Лучше: один `GetWeights(agent_id, if_newer_than=v)`.

### 2.3 Policy lag / staleness

Механизмы (от слабых к сильным):
1. **V-trace** (ρ̄, c̄ клиппинг) — в APPO есть. IMPALA/SF/SEED/RLlib.
2. **Truncated IS / masked IS**: AIPO (клип ρ∈[2,10]), INTELLECT-3 (маска [0.5,5], rollout маскируется при токен-отношении <1e-5), CISPO. Для PPO с критиком в играх не подтверждено [не подтв.].
3. **Decoupled PPO** (Hilton et al., arXiv 2110.00641; AReaL; A-3PO): behavior policy для IS, proximal для trust region.
4. **Hard gate по версии**: Prime-RL `max_off_policy_steps`=8, SF `max_policy_lag`, TorchRL `PolicyAgeFilter`.
5. **Rate-limit на источнике**: AReaL `⌊(Nr−1)/B⌋ ≤ i+η`; ROLL Flash α на sample (буфер ≤ (1+α)·batch); verl `staleness_threshold`; PipelineRL `max_lag`; StaleFlow `V_traj+η ≥ V_buf`.
6. **Мониторинг**: SF `version_diff_{min,avg,max}`, RLlib off-policy'ness metric (2.39), PipelineRL ESS.

Ориентиры допустимой staleness (единицы разные — внимание):
- OpenAI Five: цель 0–1 версия; ~8 — заметное замедление; reuse 2–3x вдвое замедляет [подтв.].
- Sample Factory: lag < 20–30 **SGD-шагов** (≠ версий весов; при `weight_push_interval=5` версия = 5 шагов) [док.].
- AReaL: η≤8 безвредно с decoupled objective; без него даже η=4 даёт −19 пунктов [подтв.]; ROLL: α=2 ≈ максимум throughput; INTELLECT-3: 8; ScaleRL: оптимум k=8.
- Cleanba: PPO теряет data efficiency при async, IMPALA нет [подтв.].
- OpenRLHF: для малых моделей/sparse rewards async опасен [док.].
→ Для APPO в играх стартовать с лимитом 1–2 версии на rate-limit, hard-drop на 4–8, измерять win-rate/ELO против synchronous baseline. [вывод]

### 2.4 Транспорт (реальная пропускная способность в Python)

Все цифры [LOCAL] — loopback, один поток, лучший из N прогонов; шум 20–30%; сеть 10/25 Gbit не измерялась.

| Транспорт | Что измерено / известно | Применимость |
|---|---|---|
| **gRPC (grpcio)** | sync unary: 226 MB/s (0.3 MB), 785 (4 MB), 804 (32 MB); client-stream, 1 сообщение на элемент: 725 / **1124** / 686 MB/s; кадры по 64 KB: 61–171 MB/s; aio gather 1033 MB/s (4 MB), аномалия 129 MB/s (32 MB). Принимает только `bytes` (memoryview/numpy → TypeError) → минимум одна копия. Лимит приёма 4 MiB по умолчанию (подтверждено: 4 194 314 B падает с RESOURCE_EXHAUSTED). Protobuf `bytes`-поле: serialize 12.8 GB/s (0.27 MB), 3.4 (4 MB), 0.97 GB/s (100 MB), parse 19 / 5.1 / 1.5 GB/s | control plane (всегда); чанки 0.3–4 MB при persistent stream и крупных сообщениях; потолок ≈ 6–9 Gbit/s на поток, для 25 Gbit упирается в CPU. Официальный совет grpc.io: streaming в Python медленнее unary (из-за потоков) — на крупных сообщениях мой замер этого не подтвердил |
| **ZeroMQ (pyzmq)** PUSH/PULL | tcp: 0.3 MB 978 (copy) / 720 (nocopy); 4 MB 622 / **1449**; 32 MB 451 / 1162 MB/s. ipc: 4 MB 1260 / 1401 MB/s. `copy=False` на крупных сообщениях ≈ вдвое быстрее; multipart: заголовок + буферы без склейки | TLeague использует ZeroMQ. Минусы: нет встроенных TLS/discovery/health; свой протокол |
| **NATS/JetStream** | `max_payload` по умолчанию 1 MB, «не рекомендуется >8 MB, максимум 64 MB» [док.] | события/сигналы; не тензоры (веса 100 MB придётся дробить) |
| **Redis** | значение ≤ 512 MB [док.]; однопоточные команды, две копии (клиент→сервер→клиент) [вывод]; **не измерено** | OpenAI Five использовал как Controller для параметров/метаданных; для данных не рекомендую |
| **Ray plasma** | zero-copy чтение numpy из shared memory на узле; ~11–13.6 GB/s single-client put на m4.16xl (Ray 0.7–0.8, 2019–2020) [устарело, не сеть] | не рассматриваем (No Ray) |
| **Arrow Flight** | до ~6 GB/s DoGet на быстрых интерконнектах (C++), ~2 GB/s на обычных сетях при 16 потоках [SEARCH, arXiv 2204.03032, не Python] | колоночные таблицы; для произвольных тензоров неудобно; под капотом тот же gRPC |
| **NCCL/UCX/RDMA/Gloo** | не измерено | GPU↔GPU весов и DDP-градиенты; фиксированная группа → плохо для эластичных CPU-нод. TorchRL предлагает Gloo/NCCL/MPI/UCC как backends коллекторов |
| **torch.distributed.rpc / TensorPipe** | **TensorPipe в maintenance mode (README 2025-12)**, RPC-API «stable and in maintenance mode»; CVE-2024-5480 (RCE через невалидированный вызов) [док./SEARCH] | **не использовать** для нового проекта |
| **Shared memory** | `multiprocessing.shared_memory` запись 7–36 GB/s, чтение 5–32 GB/s; `tensor.clone().share_memory_()` на каждый чанк — 1.3–3 мс на 4 MB (дорого); `mp.Queue` через pipe 0.79 мс (0.3 MB) против torch-shm 2.63 мс; для 4 MB shm выигрывает вдвое | single-machine: предвыделенный ring buffer + очередь индексов (как SF); `faster-fifo` вместо `mp.Queue` при >10^5 FPS |

Сводка: в Python-стеке потолок межмашинной передачи ≈ 1–1.5 GB/s на поток независимо от выбора (gRPC или ZMQ); масштабировать надо числом процессов/потоков получения и шардированием по агентам. Бюджет CLAUDE.md «100 воркеров × 10 чанков/с × 300 KB = 300 MB/s» укладывается в один поток лишь с запасом ~3x, **без учёта десериализации** (torch.load 425 MB/s на 0.27 MB [LOCAL] → 70% ядра на 300 MB/s): приём и декодирование нужно выносить из процесса лёрнера в отдельные потоки/процессы с записью в shared memory.

### 2.5 Сериализация и сжатие

[LOCAL], MB/s serialize / deserialize, траектория из шести тензоров (obs f32, actions i64, logp, values, rewards, dones):

| Метод | 0.27 MB | 4.2 MB | 100 MB (веса) |
|---|---|---|---|
| `torch.save/load` (BytesIO, `weights_only=True`) | 715 / 425 | 1348 / 3119 | 1815 / 4499 |
| pickle5 in-band | 8257 / 11531 | 3067 / 11431 | 4237 / 7849 |
| raw `tobytes` + msgpack-заголовок | 6193 / 4963 | 4564 / view | 2067 / view |
| safetensors (в `bytes`) | 1132 / 2689 | 3492 / 7934 | 2417 / 8216 |
| Arrow IPC (`write_tensor`) | 6354 / 13309 | 8408 / view | 635 / view |
| memcpy (baseline) | 38771 | 10735 | 4841 |

- «view» и «сотни GB/s» для out-of-band/zero-copy — это время создания view без копирования, не пропускная способность; реальная цена появляется при склейке в `bytes` (для grpcio неизбежна, 2–4 GB/s) или при отправке multipart.
- `torch.save` на мелких чанках в 5–10x медленнее остальных (ZIP+pickle, ~0.4 мс на 0.27 MB). safetensors безопасен (нет исполнения кода). `torch.load(weights_only=True)` сохранять.
- **Сжатие, [LOCAL], синтетические данные — пересчитывать на реальных наблюдениях**: плотные fp32 (веса, непрерывные obs) — ratio 1.00–1.18 любым кодеком; разреженные fp32 (~10% ненулевых): lz4 10x при ~1.8 GB/s, zstd-1 22x при ~600 MB/s; one-hot uint8: lz4 5.2x, zstd-1 12.5x; int64 actions: lz4 3.0x, zstd-1 11.6x (но int64→int8 без сжатия даёт 8x бесплатно).
- Окупаемость (по моим скоростям): 1 Gbit — сжатие разреженных obs окупается всегда; 10 Gbit — только lz4 и только если сжатие/отправка/распаковка конвейеризованы на разных ядрах; 25 Gbit — не окупается (c, d < L). Приоритет: (1) сузить dtype, (2) lz4 для разреженных/дискретных obs при ≤10 Gbit, (3) zstd при 1 Gbit/WAN. Включать **адаптивно**: если ratio первых чанков < 1.3 → слать без сжатия.
- Веса: fp16/bf16 → 2x (ошибка ~1e-4 для весов порядка 0.05 [LOCAL]), delta-int8 → 4x, с периодическим full snapshot для новых нод. Влияние на качество политики не проверялось.
- Литература: ActorQ/QuaRL (arXiv 1910.01055) — 8-бит квантованные акторы дают end-to-end ускорение 1.5–5.4x на непрерывном управлении [SEARCH].

### 2.6 Backpressure, fault tolerance, эластичность, discovery, SPOF

| Аспект | Практики из систем | Что взять |
|---|---|---|
| **Backpressure** | AReaL/ROLL: отказ в новых запросах по версии; PipelineRL: throttling по `max_lag`; RLlib: `max_requests_in_flight`; verl: staleness threshold + очередь; OpenRLHF: bounded queue (1); SF: размеры слотов shared memory | bounded queue у лёрнера **с drop-oldest** + счётчики; в ответе на приём стрима — «кредиты»/сигнал saturation; version rate-limit на воркере |
| **Fault tolerance** | RLlib: `restart_failed_env_runners`, probe actors; Laminar: heartbeat, переназначение траекторий, relay failover; SRL: dynamic batch переживает падение actor; INTELLECT-3: маскирование rollout при сбое sandbox; OpenAI Five: состояние в Redis Controller | чанк самодостаточен (obs, behaviour logp, версия, hidden state на старте) → потеря воркера = потеря ≤ 1 чанка; reconnect с экспоненциальным backoff и **повторным запросом адреса лёрнера у координатора**; чекпоинт лёрнера = возобновление |
| **Эластичность** | Ray autoscaler, moolib (лидер раздаёт state новым пирам), checkpoint-engine P2P, StreamRL (перераспределение GPU), INTELLECT-2 permissionless | воркеры stateless → free join/leave; лёрнер эластичность не нужна (1 процесс = 1 агент); dynamic агенты — через координатор |
| **Service discovery** | фиксированный адрес координатора (Ray head, k3s URL, Nomad servers); etcd lease/keepalive; Consul; k8s headless Service; mDNS (zeroconf — только L2) | фиксированный `COORDINATOR` + регистрация; адреса лёрнеров/стора раздаёт координатор; на k8s — headless Service |
| **SPOF** | TLeague: реплики ModelPool с балансировкой; Laminar: relay, replicated; RLlib: единый EnvRunnerStateServer (alpha) | coordinator = soft state + persist (SQLite/etcd); воркеры продолжают работать по последнему назначению при недоступности координатора; weight store: реплики/relay при росте |

### 2.7 Как делить один GPU между лёрнерами (агенты лиги)

- **Time-slicing по умолчанию** (несколько CUDA-контекстов от разных процессов): ядра разных процессов не выполняются одновременно, переключение контекстов стоит; в работе arXiv 2110.00459 (RTX 3090, training + 5000 inference-запросов) добавка к времени обучения: **time-slicing до 50 с, priority streams 30–40 с, MPS 20–30 с**; MPS «лучше всего подходит, когда ядра используют меньше всех ресурсов GPU» [подтв., прочитано в PDF]. Прочие числа (например «потеря 38%» из сниппета CERN-блога) не проверены.
- **NVIDIA MPS** [док.: docs.nvidia.com/deploy/mps]: полезен, когда каждый процесс не насыщает GPU (наш случай — маленькие сети); общий серверный CUDA-контекст, одновременное выполнение ядер из разных процессов; только Linux; **ограниченный error containment** (фатальный GPU-fault одного клиента может затронуть других на этом устройстве); по документации суммарно до 60 клиентских контекстов на устройство при `CUDA_DEVICE_MAX_CONNECTIONS=2` по умолчанию (цифра из пересказа, перепроверить); рекомендуют `active thread percentage` = 100%/(0.5·n).
- **Батчинг нескольких моделей в одном процессе** (`torch.func.stack_module_state` + `vmap`): ~2x для 10 малых одинаковых сетей (1.30 мс → 0.62 мс в туториале PyTorch), но **требует идентичной архитектуры** — конфликтует с вашим «агенты полностью независимы». Подходит для лиги однотипных агентов.
- **Один процесс-лёрнер на несколько агентов** с разными CUDA streams — реализуемо, но ломает «один процесс = один агент» и изоляцию сбоев.
- **K8s**: NVIDIA device plugin поддерживает time-slicing/MIG/MPS-шаринг (в современных версиях GPU Operator) [не проверено мной в доках]; MIG — только для A100/H100-класса.
- Практика [вывод]: для 2–8 лёрнеров малых сетей на одной GPU — MPS включить по умолчанию в deployment-рецепте (docker/k8s), в статусе «опционально», и иметь fallback на time-slicing; CPU-лёрнеры для самых маленьких агентов (по вашему CLAUDE.md лёрнеру GPU не обязательна).

---

## 3. Когда мультимашинность окупается против одной мощной машины

### 3.1 Что даёт одна машина (цифры)

| Система | Цифра | Железо | Условия |
|---|---|---|---|
| Sample Factory | до 130K FPS (abstract); 146K (VizDoom), 136K (Atari), 42K (DMLab) | 36 ядер + RTX 2080 Ti | pixel-среды, [подтв., 2020] |
| IMPALA (single machine) | 24K FPS | 48 ядер | для сравнения; кластер 500 CPU — 250K FPS |
| EnvPool | 1.07M FPS Atari, 3.13M MuJoCo | DGX-A100, 256 ядер | чистые C++-среды |
| PufferLib 2.0 / 3.0 / 4.0 | 300K–1.2M / 4M / 15–20M sps | RTX 4090 / 5090 | C-среды (Ocean), малые сети |
| Podracer Anakin | >5M шагов/с | 8-ядерный TPU | JAX-среды |
| Lux AI S3, 1-е место | ~110K шагов/с симулятор (16 ядер), обучение ~430 шагов/с | Ryzen 9950X + RTX 3090 | ~300M шагов за ~8 суток [не подтв. независимо] |
| Neural MMO 2023 | 15–22M шагов за 8 ч | 1× RTX 4090 | победители, PuffeRL |

### 3.2 Узкие места по порядку [вывод по источникам]

1. **Среда** (Python-реализации Kaggle-симуляций, Lux AI/NMMO на CPU): десятки–тысячи шагов/с/ядро → нужно много ядер → multi-machine окупается рано.
2. **Инференс**: пока сеть малая и CPU-инференс дешевле env step — не узкое место (SEED: дорог для больших CNN; Lux S3: backward+inference на GPU ограничивали 430 шагов/с для 10M-параметрической модели).
3. **Обучение**: один GPU-лёрнер для малой сети потребляет десятки–сотни тысяч сэмплов/с [не измерено]; после этого CPU-воркеров нужно столько, чтобы его накормить — а при лиге из A агентов потребление умножается на A.
4. **Сеть/Python**: приём чанков (1–1.5 GB/s на поток, десериализация), раздача весов.
5. **Staleness**: рост числа воркеров → рост лага; приходится увеличивать batch (SF: lag растёт с числом воркеров и envs).

### 3.3 Правило безубыточности [вывод]

```
cores_needed = (consume_rate_per_agent × n_trainable_agents) / (steps_per_sec_per_core × inference_overhead_factor)
```

- Если `cores_needed` ≲ числа ядер одной доступной машины (≤ 32–64) — **одна машина**: меньше проблем со сетью, нулевая staleness от сети, проще отладка.
- Если среда на Python 100–1000 шагов/с/ядро, лига из 4–8 агентов с потреблением по 20–50K сэмплов/с на агента → `cores_needed` ≈ сотни–тысячи → **multi-machine обязателен**.
- Если среда на C/Rust ≥ 50K шагов/с/ядро и один агент — мультимашинность не нужна; лучше оптимизировать single-node (PufferLib-стиль).
- Лига с несколькими GPU-лёрнерами (по одному на агента, разные архитектуры) — отдельная причина для multi-machine: ресурсы, а не throughput одной среды.

Для **соревнований** (Lux AI, Neural MMO, CodeCraft, Kaggle): факты показывают, что победы достигались на 1 станции за часы–сутки. Мультимашинность — это скорее гибкость и возможность для лиги/самоигры расти, чем необходимость; поэтому **простота добавления машины и надёжный single-machine режим важнее максимального scale-out**. [вывод]

---

## 4. Оценка текущей схемы Colosseum и рекомендуемая топология

### 4.1 Что в текущей схеме оптимально

(Оценка по CLAUDE.md и чтению `src/colosseum/{distributed.py,transport/grpc_transport.py,transport/serialization.py,weight_store/*,worker/rollout_worker.py,learner/learner.py,core/config.py}`.)

- **Локальный CPU-инференс на воркерах + асинхронные лёрнеры + V-trace** — подтверждается SF/IMPALA/RLlib/SRL как лучший дефолт для малых сетей и гетерогенных/эластичных кластеров. Не менять на SEED-схему.
- **Чанки самодостаточны**: `TrajectoryChunk` содержит `behavior_policy_version`, LSTM hidden (stored-state), action masks. Это ровно то, что нужно для staleness-контроля и безопасной потери чанков.
- **Queue-like адаптеры** (`GRPCTrajectorySink`, `GRPCWeightSource`, `GRPCWeightSink`): `learner_process` и `rollout_worker_process` не знают о транспорте — отличная база для смены backend'а (gRPC ↔ ZMQ ↔ shm) без переписывания логики.
- **Pull весов с периодом** (`weight_sync_interval_sec=5`, `weight_push_interval=5`): совпадает с OpenAI Five/RLlib/SRL.
- **gRPC как control plane** и protobuf для метаданных — нормальный выбор (Python gRPC тут не узкое место).
- **Single-machine fast path** (mp.Queue + shared memory) оставить как есть — SF/PufferLib показывают, что любая сериализация на одной машине вредна.
- Принципы CLAUDE.md «`weights_only=True`», lz4-компрессия — сохранить (с поправками ниже).

### 4.2 Что не оптимально (с привязкой к коду)

| # | Наблюдение | Evidence | Последствие |
|---|---|---|---|
| 1 | `GRPCTransport.send_chunk` открывает stream на каждый чанк (`self._stub.SendChunks(iter([proto]))`); `send_chunks_batch` есть, но `GRPCTrajectorySink` его не использует | `transport/grpc_transport.py`, `distributed.py` | per-call overhead ≈ 1.3 мс на 0.3 MB [LOCAL] → ≈ 226 MB/s вместо ~1 GB/s; ваш бюджет 300 MB/s не выдерживает |
| 2 | Серверный хендлер: десериализация `torch.load` в gRPC-потоке (4 потока), `queue.put(timeout=5)` и drop при заполнении; клиент при `RpcError` молча дропает чанк | `TrajectoryServicer.SendChunks`, `GRPCTrajectorySink._send` | десериализация съедает ~70% ядра на 300 MB/s [LOCAL]; backpressure нет (только drop), потери невидимы без метрик |
| 3 | Нет контроля staleness: `behavior_policy_version` не используется ни при приёме, ни при обучении; нет `max_policy_lag`; нет метрики лага | grep `learner.py`, `appo.py`, `rollout_worker.py` | при росте воркеров/задержек сети — тихая деградация APPO (Cleanba, OpenAI Five) |
| 4 | Веса: два RPC (`get_version` + `get`) на опрос, каждый процесс-воркер тянет сам, `torch.save`+lz4 (ratio 1.0 на fp32) | `GRPCWeightSource`, `weight_store/grpc_store.py`, `transport/serialization.py` | трафик ∝ числу процессов, лишнее CPU; гонка версий |
| 5 | `torch.save` для чанков | `serialize_chunk` | в 5–10x медленнее raw-буферов на 0.3 MB |
| 6 | Нет распределённого координатора: PFSP/исторические оппоненты/арена только на одной машине; адреса лёрнеров статичны (`learner_addresses`); нет регистрации/heartbeat/lease | docstring `distributed.py` | ключевая функциональность лиги недоступна кластеру; добавление машины — ручная правка CLI |
| 7 | Нет проверки совместимости кода/версий воркер↔лёрнер; пользовательский код среды раздаётся «как-то» (образ) | — | при расхождении (изменённая среда/encoder) — молчаливо неверные траектории |
| 8 | `insecure_channel`, `insecure_port`, без токена | `grpc_transport.py` | недопустимо при подключении машин через интернет/VPN |
| 9 | Weight Store = единственный gRPC-сервис; нет хранения нескольких версий/чекпоинтов-оппонентов как «ModelPool» | `grpc_store.py` | исторические оппоненты в распределённом режиме негде брать |
| 10 | Нет калибровки/выбора роли по возможностям ноды | — | «GPU-ноды тоже могут гонять среды» придётся делать вручную |

### 4.3 Рекомендуемая топология для гетерогенного кластера

**Принцип**: одна «нода» = один демон `colosseum node`, который при старте измеряет возможности, регистрируется у координатора, получает **роли** (процессы) и сам их запускает/останавливает. Роли — наши существующие компоненты, ничего принципиально нового на data plane.

```
                          ┌──────────────────────────────────────┐
                          │ Coordinator (control plane, 1 шт.)   │
                          │ registry нод · пул агентов · матчмейк│
                          │ ELO/WR · назначение ролей · version  │
                          │ gate (code_hash) · метрики · persist │
                          └─────▲──────────▲──────────────▲──────┘
          register/heartbeat/   │          │ match tasks  │ results
          assignments (pull, outbound only)
   ┌─────────────┐   ┌──────────┴───┐  ┌───┴──────────┐  ┌─────────────┐
   │ CPU-нода    │   │ GPU-нода     │  │ GPU-нода     │  │ CPU-нода    │
   │ node-agent  │   │ node-agent   │  │ node-agent   │  │ node-agent  │
   │  weight     │   │  Learner A,B │  │  Learner C   │  │  weight     │
   │  cache(shm) │   │  (MPS)       │  │  + Workers   │  │  cache(shm) │
   │  Workers×N  │   │  + Workers   │  │  (GPU infer.)│  │  Workers×N  │
   └──────┬──────┘   └───▲──────────┘  └──▲───────────┘  └──────┬──────┘
          │ chunks (persistent gRPC stream, batched, version-tagged)
          └──────────────┴────────────────┴───────────────────────┘
          weights: pull if_newer (1 pull/нода) ← Model/Weight Store (versions + frozen ckpts)
```

Роли:
1. **Coordinator**: gRPC-сервис; хранит registry нод (capabilities, lease), пул агентов, ELO/WR, матчмейкер (self-play/PFSP/arena), version-gate `code_hash`. Состояние персистентное (SQLite) — рестарт не убивает обучение; воркеры продолжают по последнему назначению.
2. **Model/Weight Store** (версионированный, N последних версий на агента + frozen checkpoints; ключ `agent_id@version`): аналог TLeague ModelPool. Сначала совмещён с координатором; при росте — реплики/relay.
3. **Learner**: один процесс = один обучаемый агент (как сейчас); на GPU-ноде или CPU. Принимает чанки на **persistent stream**, декодирует в отдельном потоке/процессе в shared memory; гейтит по версии; публикует веса.
4. **Worker**: процессы с векторными средами; **инференс локально** на CPU по умолчанию, на GPU-нодах — опция `inference_device=cuda` (батч по всем env процесса). Отправляет чанки лёрнерам по адресам от координатора.
5. **Node agent** + **per-node weight cache**: один pull весов на машину, раздача воркерам через shared memory (как в SF/RLlib-аналоге).
6. **(Опционально, позже) InferenceServer**: батчевый сервис для больших сетей/GPU-богатых LAN-кластеров (SRL policy worker / TLeague InfServer).

Размещение по возможностям [вывод]:
- GPU-нода: learner(ы) + воркеры на оставшихся ядрах (GPU-нода «тоже гоняет среды»); если сеть тяжёлая — воркеры используют ту же GPU для батч-инференса (MPS при нескольких лёрнерах).
- CPU-нода: только воркеры (число процессов = ядра − reserved), локальный CPU-инференс.
- Слабые/случайные ноды (ноутбук владельца, spot): воркеры с низким приоритетом, drain по SIGTERM.
- Калибровка на старте: `env_step_ms`, `forward_ms(B=1,16,64)`, `net_MB_s` до лёрнера → координатор выбирает число процессов/envs на процесс и режим инференса.

### 4.4 Что поменять, чтобы схема стала оптимальной (кратко; детали в разделе 7)

1. Persistent-stream + батч чанков + decode вне процесса лёрнера; backpressure/метрики потерь.
2. Version-aware pipeline: `weight_version` в чанке (уже есть), gate на лёрнере, rate-limit на воркере, метрики лага; опционально decoupled objective.
3. `GetWeights(if_newer_than)`, per-node cache, fp16-транспорт, без lz4 для весов; версионированный store с историей.
4. Node agent + registration + capabilities + lease + drain; coordinator как сервис; `code_hash` gate.
5. Собственная компактная сериализация (header + flat buffers; safetensors для весов), адаптивное сжатие.
6. Токен-аутентификация (+ optional TLS) на всех gRPC-портах.

---

## 5. UX деплоя: «добавить машину» и раздача кода

### 5.1 Как это делают популярные системы

| Система | «Добавить машину» | Гетерогенность | Минусы |
|---|---|---|---|
| **Ray** (VM) | `pip install "ray[default]"` + `ray start --address=<head>:6379 [--num-gpus=..]` (2 шага) | custom `--resources`, `--labels` (beta, Ray 2.49+; селекторы `in()`, `!in()`) | жёсткое совпадение версий Ray/Python; адрес head недоступен из другой подсети/NAT; token auth: с 2.52, обязателен для всех кластеров с 2.61 [док.] |
| **KubeRay** | один `RayCluster` с `headGroupSpec` и несколькими `workerGroupSpecs` (replicas/min/max, `nodeSelector`, `tolerations`, `rayStartParams`) | CPU- и GPU-группы с разными шаблонами; autoscaler по логическим запросам ресурсов | только внутри k8s; autoscaler смотрит на запросы, не на утилизацию |
| **Kueue / Volcano / JobSet / Kubeflow Trainer v2** | очереди, квоты, gang/all-or-nothing admission, JobSet — группы Job с разными pod template | да (flavors) | избыточно для async RL: gang нужен только DDP-лёрнеру |
| **Slurm** | `#SBATCH hetjob` + `srun --het-group` — «learner-компонент + worker-компонент» | да (hetjob) | статичная аллокация, динамически присоединиться нельзя |
| **SkyPilot** | `sky launch task.yaml` (resources/accelerators, `workdir` как git url+ref, `file_mounts`, `setup`/`run`), **SSH Node Pools** — свои IP+SSH в `~/.sky/ssh_node_pools.yaml` + `sky ssh up` | да | ориентирован на задачи, не на долгоживущий кластер с динамическим join |
| **submitit / Dask SSHCluster / Prefect worker** | submitit: pickle через shared FS; Dask: `SSHCluster`; Prefect: `prefect worker start -p <pool>` опрашивает work pool каждые 15 с | частично | submitit — не пул воркеров |
| **k3s / kubeadm** | `curl -sfL https://get.k3s.io \| K3S_URL=https://server:6443 K3S_TOKEN=<token> sh -`; kubeadm: bootstrap token (TTL) | `--node-labels`, taints | k3s через NAT: wireguard-native нужен достижимый внешний IP; Tailscale-интеграция experimental |
| **Nomad client** | `servers`/`retry_join` в конфиге; авто-fingerprint CPU/RAM/disk | `node_class`, `node_pool`, `meta`, `reserved` | нужен Nomad |
| **GitHub Actions runner / Buildkite agent** | `./config.sh --url --token` (токен регистрации живёт 1 час) / `buildkite-agent start --token --tags "queue=..."` | теги/labels | — |
| **Celery** | `celery -A proj worker -Q q --concurrency N`; warm shutdown по SIGTERM | очереди | брокер — внешний сервис |
| **torchrun elastic** | `--nnodes min:max --rdzv-backend c10d` | нет | любое изменение состава рестартует воркеры — для DDP, не для async RL |
| **INTELLECT-2 / PRIME-RL** | permissionless swarm, TOPLOC (верификация rollouts), SHARDCAST (раздача весов) | да (произвольные узлы) | сложная верификация/доверие |

### 5.2 Общий протокол «агент-нода сама регистрируется» (выведен из систем выше)

1. **Join token**: короткоживущий (GitHub 1 ч, kubeadm TTL) или reusable; обменивается на постоянную идентичность (`node_id` + ключ). Tailscale-стиль: ephemeral/pre-approved/tagged ключи.
2. **Capabilities при регистрации**: авто-fingerprint (Nomad) — ядра, RAM, GPU (модель, VRAM), arch/OS + пользовательские labels (`--label region=home`) и `reserved` ядра хосту.
3. **Только исходящие соединения** и pull работы (GitHub long poll 50 с, Buildkite, Prefect polling 15 с) → работает за NAT, нужен один порт на координаторе.
4. **Heartbeat + lease** (kubelet Lease, etcd TTL keepalive): потеря lease → слоты ноды освобождаются.
5. **Graceful drain** (Nomad `drain -deadline`, Celery warm shutdown, kubelet graceful shutdown): SIGTERM → закончить чанк → `Deregister`.
6. **Проверка совместимости при join** (Ray: mismatch версий): `protocol_version`, `framework_version`, `code_hash`.

### 5.3 Раздача пользовательского кода (кастомная среда/encoder/policy)

| Способ | Плюсы | Минусы | Примеры |
|---|---|---|---|
| Docker-образ с кодом | воспроизводимо, тег/digest | пересборка/пуш на каждое изменение, нужен registry | k8s, Ray `container` |
| Образ + динамическая накатка кода | тяжёлые слои в кэше, код меняется быстро | две версии для согласования | Modal (файлы добавляются при старте контейнера) |
| Архив `working_dir` по content-hash | мгновенно, без registry | лимит 500 MiB (Ray), нужен общий Python/ABI | Ray runtime_env (кэш на узле, `excludes` как .gitignore) |
| Wheel / `pyproject`+lock (`uv sync --locked`) | зависимости+версии | нужен индекс/артефакт | uv docker-guide: зависимости отдельным слоем, `UV_COMPILE_BYTECODE=1` |
| Git-ref pinning | просто, аудируемо | git-доступ на каждой ноде | SkyPilot `workdir: {url, ref}` |

**Рекомендация [вывод]**: пользовательский пакет = `pyproject.toml` + `uv.lock` + код; клиент собирает архив с content-hash (`code_hash`) и кладёт в store координатора (или в S3/registry); node agent скачивает по хэшу, кэширует, создаёт venv через `uv sync --locked`, проверяет хэш; **version gate**: координатор выдаёт задания только нодам с совпадающим `code_hash` в рамках «поколения» обучения; при смене кода — новое поколение (drain → скачать → перезапуск). Для k8s — тот же пакет внутри базового образа (`colosseum-node:<fw_version>`), код докачивается при старте. Без Ray runtime_env, но с той же идеей.

### 5.4 Связность и discovery

- **Tailscale/WireGuard** — самый простой способ связать машины за NAT без публичных IP (auth-key reusable/ephemeral/tagged, MagicDNS). В k3s — wireguard-native или `--vpn-auth` (experimental).
- **Docker Compose** сам по себе не раскатывает на несколько машин (работает через один Docker-endpoint) [вывод, прямого подтверждения нет]; `docker context` + SSH применим для разовых запусков; **Docker Swarm** — лишняя зависимость (порты 2377, 7946, 4789/udp).
- **SSH-лаунчеры** (Ansible, Fabric, pssh, SkyPilot SSH Node Pools) как *опциональный* bootstrap: «поставить node agent на N машин». Fabric/pssh не проверялись [не подтв.].
- **Discovery**: фиксированный `COORDINATOR=host:port` (или DNS-имя Tailscale); k8s — headless Service/DNS. mDNS только для LAN.
- **K8s**: HPA/KEDA по custom metric (queue depth = число чанков в очереди лёрнера, или `idle_ratio` лёрнера); nodeSelector/taints для GPU vs CPU; PriorityClass для лёрнеров выше воркеров; spot/preemptible для воркеров (stateless).

### 5.5 Минимально-простой дизайн «одна команда на новой машине»

```
docker run -d --restart=unless-stopped --gpus all \
  -e COLOSSEUM_COORDINATOR=coord.tailnet:7000 \
  -e COLOSSEUM_JOIN_TOKEN=<token> \
  -v colosseum-cache:/cache \
  ghcr.io/<org>/colosseum-node:<fw_version>
# без Docker: curl -sfL https://<coord>/install | COLOSSEUM_JOIN_TOKEN=... sh -
```

Что делает агент: fingerprint → `Join(token, caps, framework_version, protocol_version, code_hash)` → получает `node_id`, lease TTL, адрес стора → pull назначений (роль, агенты, число процессов) → скачивает код по хэшу → запускает роли → heartbeat; SIGTERM → drain. Все соединения исходящие. Для GPU-нод флаг `--gpus all`, без него — CPU-only.

Не брать: Swarm, torchrun elastic, gang scheduling, Volcano/Kueue для первой версии. Брать: идея `--address`/join-token (Ray, k3s), fingerprint+reserved (Nomad), pull+long poll (GitHub/Buildkite), lease (kubelet/etcd), drain (Nomad/Celery), content-hash архив (Ray runtime_env/Modal).

---

## 6. Сводная таблица систем по осям дизайна

| Система | Инференс | Синхронизация весов | Контроль staleness | Транспорт данных | Multi-agent / league | Multi-node / эластичность | Fault tolerance | Цифры / статус |
|---|---|---|---|---|---|---|---|---|
| **IMPALA** (2018) | локально на акторе (CPU) | pull перед каждым unroll | V-trace | gRPC/TF queues | нет | кластер 500 CPU | н/д | 250K FPS @500 CPU; single 24K |
| **SEED RL** (2020) | центральный на TPU/GPU | не нужна (1 копия) | модель одна; V-trace/R2D2 | gRPC streaming (C++/TF), unix sockets локально | нет | до TPU pod 2048 ядер | перезапуск акторов | 2.4M FPS @4160 CPU + 64 TPU; на P100 0.63x IMPALA; архив [устарело] |
| **Sample Factory 2.x** | inference workers (CPU/GPU) на том же узле, double-buffering | shared memory, <1 мс | V-trace + PPO clip, `max_policy_lag`, lag 5–10 шагов | shared-memory слоты, `faster-fifo` | multi-agent, multi-policy, self-play, PBT | в первую очередь single-node (multi-node — [не подтв.]) | н/д | 130–154K FPS/узел; PyPI 2.1.1 (2023), репо активно |
| **PufferLib 3.0 / 4.0** | в процессе: GPU (CUDA-C), env на OpenMP-потоках | не нужна (один процесс) | почти on-policy (sync / 1-epoch async как Cleanba) | shared memory, pinned buffers | native multiagent; league/PFSP — [не подтв.] | нет | н/д | 4M (3.0) / 15–20M sps (4.0) @RTX 5090; 5.0 («60M») [не подтв.] |
| **Cleanba / Podracer** | Sebulba: TPU/GPU-ядра actor-группы; Anakin: внутри jit | collective ops (pmap) на устройствах | детерминированный 1-step async | NVLink/ICI | нет (Mava поверх) | TPU pods | н/д | Anakin >5M/с @8 TPU ядер; Sebulba 43M FPS @2048 ядер |
| **RLlib new stack 2.5x** | локально на EnvRunner (Ray actors) | pull `pull_if_newer(seq_no)` через `EnvRunnerStateServer` (alpha) | V-trace/APPO circular buffer; `WEIGHTS_SEQ_NO` в данных | Ray object store | `MultiRLModule`, league-пример на 1 драйвере | Ray cluster, aggregator actors | restart EnvRunner, spot tolerance | 100K ts/s @400 EnvRunner + 16 GPU; активно |
| **TorchRL 0.14** | в коллекторе; inference server (0.12+) | push, weight sync schemes | `max_weight_update_interval`, `PolicyAgeFilter` | gloo/NCCL/RPC/Ray | блоки, нет league | да (distributed/Ray) | в доках нет | активно (0.14.0 2026-09) |
| **moolib** | пиры | all-reduce + раздача state лидером | V-trace | TensorPipe (shm/IB/TCP) | нет | elastic пиры | перезапуск пира | архив 2023 [устарело] |
| **Acme/Reverb/Launchpad** | на акторе | параметр-сервис | replay-семплинг | gRPC (Courier) | нет | Launchpad | н/д | Launchpad архив 2024, Acme релиз 2022 [устарело] |
| **SRL** (2023) | policy workers (батчевый, GPU/CPU) | trainers → NFS parameter server → pull | нет явных порогов | shm / сокеты, lz4 | multi-agent, PBT-ориентирован | 15 000+ ядер, 32 A100 | динамический batch, controller | 6.3–21.6x RLlib; код заморожен (2024-02) |
| **TLeague** (2020) | локально или опциональный InfServer | pull из ModelPool (реплики) | V-trace/PPO (в алгоритмах) | ZeroMQ + Horovod | **LeagueMgr/GameMgr, CSP-MARL** | k8s, гибрид CPU/GPU | н/д | статус неизвестен [не подтв.] |
| **OpenAI Five Rapid** | отдельные Forward-Pass GPU | pull из Redis Controller | цель staleness 0–1 | собственный (закрыт) | self-play 80/20 | 1536 GPU + ~128K CPU | состояние в Redis | закрыт |
| **AReaL / verl / Prime-RL / ROLL / PipelineRL** | отдельный rollout-кластер (vLLM/SGLang) | NCCL/ФС/HTTP, версия на sample | rate-limit η/α/`max_lag`, drop >k, decoupled PPO | NCCL, Redis/MQ, HTTP | нет (LLM) | да (до 512 GPU) | Laminar подробно; остальные [не подтв.] | 1.5–2.8x vs sync |
| **Colosseum (сейчас)** | локально на CPU воркера | pull раз в 5 с, 2 RPC, `torch.save`+lz4 | V-trace; версия в чанке есть, **gate нет** | mp.Queue+shm локально; gRPC stream-per-chunk | свой матчмейкинг (PFSP, arena), ELO — **только single-machine** | статические адреса, нет регистрации | drop чанков при ошибке | 148 тестов; distributed — только self-play latest |

---

## 7. Рекомендации для Colosseum (приоритизированные)

Обозначения: **Усилие** S (≤ 2–3 дня), M (≈ неделя), L (> недели). **Риск** — что может пойти не так.

### 7.1 Must-have

**M1. Контроль staleness по версиям (ядро надёжности APPO при распределении).** Усилие S–M.
- Чанк уже несёт `behavior_policy_version`; добавить в него версию на момент **начала** чанка и (если воркер обновляет веса внутри чанка) `version_end`/массив. Обновлять веса воркера **только на границе чанка** (проще для LSTM stored-state: у AReaL/PipelineRL пересчёт KV-cache не критичен, у вас нет эквивалента для hidden state, подтверждений безопасности нет [не подтв.]).
- Лёрнер: считать `lag = learner_version − chunk_version` для каждого чанка; метрики min/avg/max/p95 (как SF `version_diff_*`); **soft gate** (предупреждение/снижение веса) и **hard drop** при `lag > max_policy_lag` (config); счётчик отброшенных.
- Воркер: rate-limit в стиле AReaL — не начинать новые чанки, если `in_flight_chunks_for_agent > (η+1)·batch_chunks`, где `in_flight` оценивается по ack/кредитам от лёрнера. Это backpressure без барьера.
- Стартовые значения: η=1–2, hard drop 4–8 версий; подобрать A/B-тестом против синхронного baseline (SF-режим `async_rl=False`) на tic-tac-toe/ChaseEnv и одной тяжёлой среде. Единица версии = push весов (по умолчанию каждые 5 шагов), согласовать с `weight_push_interval`.
- Evidence: OpenAI Five (staleness ~8 вредит), Cleanba (PPO теряет data efficiency), AReaL (rate-limit), Prime-RL (drop >8), SF (`max_policy_lag`).
- Риск: слишком жёсткий η сводит async к sync; нужен график throughput vs η (как Table 7 AReaL).

**M2. Persistent client-stream + батч чанков + декодирование вне лёрнера.** Усилие M.
- Один долгоживущий `SendChunks` stream на пару (процесс-воркер → агент), в одном сообщении 1–N чанков (цель 1–4 MB); не дробить на кадры <1 MB. Проверить `send_chunks_batch`, подключить его в `GRPCTrajectorySink` с локальным буфером (flush по размеру/таймауту).
- Приёмник: отдельные потоки/процессы-ресиверы, `np.frombuffer` в предвыделенные слоты shared memory; процесс-лёрнер читает индексы (как SF). Не `torch.load` в gRPC-потоке.
- Bounded queue с **drop-oldest** (свежие данные ценнее: OpenAI Five), с метриками `chunks_dropped_{queue_full,rpc_error,stale}` и «кредитами»/ответом `queue_depth` в gRPC-ответе для rate-limit на воркере.
- Поднять `max_message` согласованно на обоих концах; ключевой потолок [LOCAL] ≈ 0.8–1.1 GB/s на поток; шардировать несколькими receiver-процессами при >500 MB/s.
- Риск: backpressure без блокировки воркера → решать drop-политикой, а не `put(timeout=5)` в gRPC-потоке.

**M3. Weight distribution: conditional pull, per-node cache, версионированный store.** Усилие M.
- `GetWeights(agent_id, if_newer_than=v) → payload | NOT_MODIFIED` (один RPC вместо двух).
- Node agent держит **один** pull на машину и публикует веса в shared memory; воркеры локальные процессы читают оттуда (трафик ∝ числу машин, не процессов; 10 MB × 10 машин / 5 с = 20 MB/s вместо 200 MB/s).
- Формат весов: safetensors/raw-буфер, **без lz4** (ratio 1.0 [LOCAL]); опция `weights_dtype: fp16|bf16` (2x; проверить влияние на win-rate), позже int8-дельты.
- Store: хранить N последних версий на агента + frozen checkpoints по ключу `agent_id@version` (ModelPool из TLeague) — необходимо для PFSP/исторических оппонентов в кластере.
- Риск: fp16-передача меняет поведение политики воркера относительно лёрнера (V-trace скомпенсирует часть) — A/B на задаче.

**M4. Node agent + регистрация + capabilities + lease + drain + version gate.** Усилие L (но это и есть ваш главный приоритет).
- `colosseum node --coordinator HOST:PORT --token ...`: fingerprint (ядра, RAM, GPU/VRAM, arch), labels, `reserved`; `Join(...)` → `node_id`, lease TTL, адреса; heartbeat; pull назначений (роль, агенты, число процессов); drain по SIGTERM и по команде.
- **Version gate**: `protocol_version` + `framework_version` + `code_hash` пользовательского пакета; несовпадение → нода не получает работу, а получает инструкцию синхронизироваться.
- Все соединения исходящие; один порт у координатора; токен (см. S6). Роль и число процессов выбирает координатор по capabilities и калибровке (см. S3).
- Evidence: Ray `ray start --address`, k3s join, Nomad fingerprint, GitHub runner/Buildkite pull, kubelet lease.
- Риск: самая большая по объёму работа; сделать минимальный вертикальный срез: Join + heartbeat + запуск воркера с назначенным агентом на 2 машинах.

**M5. Координатор как сервис (распределённая лига).** Усилие L.
- Вынести `Coordinator`/`AgentPool`/`Matchmaker`/ratings за gRPC-API: `RequestMatch`, `ReportResult`, `ListLearners`, `Subscribe(assignments)`. Состояние в SQLite (рестарт без потери ELO/WR/пула).
- Воркер запрашивает матч: coordinator возвращает слоты (агент → версия → learner address / collect flag). Исторические оппоненты — по `agent_id@version` из store.
- Без этого PFSP/арена в кластере недоступны (текущий docstring `distributed.py`).
- Риск: латентность запроса матча на каждый эпизод — батчить (получать пачку заданий) и кэшировать.

**M6. Сохранить и отполировать single-machine fast path.** Усилие S.
- Не регрессировать: shared-memory ring buffer, no-serialization (SF/PufferLib); тот же код ролей запускается как «локальный кластер» (coordinator + node agent на одной машине) — один и тот же путь для single и multi.

### 7.2 Should

**S1. Собственный compact-формат сериализации и адаптивное сжатие.** Усилие S–M. Заголовок (msgpack/JSON: dtype, shape, offset) + плоские буферы; `torch.save` убрать с горячего пути (5–10x на 0.3 MB). Сузить dtype (actions int64→int8/uint8, bool-маски упакованные, obs uint8 где можно). lz4/zstd включать по измеренному ratio первых чанков (>1.3), канал ≤10 Gbit — lz4 с конвейеризацией; на 1 Gbit/WAN — zstd-1. Веса — safetensors.

**S2. Decoupled/proximal PPO-objective в APPO как опция** (`appo.proximal: "behavior" | "learner_prev" | "interp"`). Усилие M. π_prox = параметры перед апдейтом (как AReaL) либо A-3PO-интерполяция (`α=1/d` по логитам, без лишнего forward). Верхняя обрезка IS-веса (AIPO ρ, у вас V-trace ρ̄/c̄). Оценить на A/B с M1.

**S3. Калибровка ноды и auto-placement; GPU-ноды как воркеры; опциональный InferenceServer.** Усилие M–L. При регистрации микробенч: `env_step_ms` (пользовательская среда), `forward_ms(B)` на CPU и GPU, пропускная способность до лёрнера. Правило: если `forward_cpu_per_sample > ~0.5·env_step` и есть GPU-нода в LAN (RTT ≲ 1 мс) — `inference_device=cuda` внутри процесса воркера (первый шаг) → опциональный сервис батчевого инференса (SRL/TLeague). По моему микробенчу [LOCAL] порог пересекается примерно при 1–5M параметров/CNN.

**S4. Шаринг GPU между лёрнерами.** Усилие S–M. Рецепт запуска MPS (docker/compose/k8s) + документация; режим «несколько агентов в одном процессе-лёрнере» отложить; `vmap`-батчинг однотипных агентов — исследовать для лиг одинаковой архитектуры (2x на 10 малых сетях в туториале PyTorch). CPU-лёрнер для самых малых агентов уже поддерживается (`learner.device`).

**S5. Раздача кода: content-hash пакет + uv.** Усилие M. Клиент: `colosseum pack` → архив + `code_hash`; store координатора (или S3/registry); node agent: скачать по хэшу, `uv sync --locked`, проверить. Базовый образ `colosseum-node:<fw_version>`; на k8s код докачивается при старте.

**S6. Безопасность control/data plane.** Усилие S–M. Join-токены с TTL → постоянный node key; bearer-token (или mTLS) на всех gRPC-портах (Ray 2.61 делает token auth обязательным — прецедент); `torch.load(weights_only=True)` сохранить; не использовать pickle для сообщений между нодами (CVE-2024-5480 в torch.distributed.rpc).

**S7. Наблюдаемость.** Усилие S. Метрики: staleness-гистограмма, ESS, доля отброшенных чанков, queue depth/idle ratio лёрнера, chunks/s и MB/s по нодам, время цикла pull весов, step-time env по нодам. Это вход для HPA/KEDA и для калибровки η.

### 7.3 Nice-to-have

- **N1.** Альтернативные backend'ы `BaseTransport`: ZeroMQ (multipart, `copy=False`; 1.0–1.5 GB/s [LOCAL]) для LAN-кластеров, как у TLeague; Arrow Flight — маловероятно нужен.
- **N2.** Иерархическая раздача весов (relay на узле/регионе, SHARDCAST/Laminar-стиль) при >50–100 нод или WAN.
- **N3.** Дельты весов (fp16/int8, 2–4x [LOCAL, синтетика]) + периодические full snapshots; проверить разреженность на реальных обучаемых весах.
- **N4.** Рецепты запуска: Tailscale (ephemeral tagged keys), SkyPilot SSH Node Pools / Slurm hetjob / Ansible для bootstrap node agent; k8s: KEDA по queue depth, nodeSelector/taints GPU vs CPU, PriorityClass лёрнеров > воркеров, spot для воркеров.
- **N5.** Spot-aware: обработка preemption-уведомлений облака → drain.
- **N6.** Replication Weight Store/coordinator (active-passive) при росте масштаба.
- **N7.** GPU-среды (JAX/Warp) как опциональный backend `BaseEnv` — только для сред, где правила зафиксированы (Pgx/gymnax-типа); для Kaggle/Lux — нет.
- **N8.** Fault injection тесты (убийство воркера/лёрнера/сети), chaos-режим в CI.

### 7.4 Чего избегать

Ray (решение «No Ray» подтверждается: RLlib-лига — один драйвер, aggregator actors и state server alpha); TensorPipe/`torch.distributed.rpc`; NATS/Redis как data plane для тензоров; torchrun elastic; Docker Swarm; gang scheduling/Volcano/Kueue в v1; центральный инференс как дефолт; перенос LLM-значений η/k без A/B; полагаться на «60M sps» PufferLib 5.0 до независимого подтверждения.

### 7.5 Порядок работ (предложение)

1. M1 + S7 (staleness + метрики) — быстрый эффект на качество, не зависит от инфраструктуры.
2. M2 + S1 + M3 (data/weight plane) — снимает потери производительности.
3. M5 + M4 (coordinator service + node agent) — основной приоритет владельца; вертикальный срез на 2 машинах, потом capabilities/калибровка (S3), код-раздача (S5), безопасность (S6).
4. Остальное по потребности.

---

## 8. Что не подтверждено, устарело или требует измерения на вашем железе

- **Сеть**: все числа транспорта — loopback WSL2; не измерены 10/25 Gbit, RTT, MTU, TLS, RDMA/UCX, Redis, NATS, реальный Python-vs-C++ gRPC. Перед выбором архитектуры приёма чанков измерить на целевой сети (скрипты `/tmp/bench_scripts/{grpc_b,zmq_b,ser_b,comp,shm_b,delta_b}.py` — одноразовые, не часть репозитория).
- **Сжатие**: данные синтетические; на реальных наблюдениях Lux AI/NMMO/Kaggle ratio будет другим.
- **PufferLib**: 5.0 и «60M sps» — только docs-страница через summarizer, независимого подтверждения нет; self-play/league и multi-node — не нашёл; точное содержимое 4.0/5.0 кроме блога 4.0 не проверял.
- **Sample Factory**: multi-node и дефолт/семантика `max_policy_lag` не проверены; число из «PyPI 2.1.1 (2023)» и «репо активно (коммит 2026-10-01)» — данные субагента.
- **RLlib**: бенчмарки Anyscale не найдены; расхождение по multi-agent vectorization (release notes vs docs); issue #50221 не проверен; даты удаления старого стека нет.
- **TLeague**: текущий статус репозитория неизвестен. **TorchRL**: fault tolerance и throughput не найдены.
- **LLM-RL**: числа verl/OpenRLHF/slime/checkpoint-engine/prime-rl взяты из документации через summarizer; fault tolerance у AReaL/verl/Prime-RL в доступных источниках не описана; Tinker, Seed/ByteDance one-step — не подтверждено; AsyncFlow/StaleFlow/Laminar — таблицы в PDF плохо читались. Значения η/k/α получены на LLM с GRPO-лоссами (без критика) и **не переносимы** на PPO в играх без A/B.
- **Lux AI S3 / Neural MMO**: цифры победителей (8 суток на станции, 430 шагов/с, 8 ч на 4090) — со слов субагента по write-up/статьям, независимо не перепроверены мной.
- **AlphaStar**: числа лиги (32 TPU v3 на агента, 44 дня) из сниппета, Nature недоступен.
- **MPS**: лимит «60 клиентских контекстов» и отсутствие полного error containment — из пересказа документации; нужна проверка на вашей версии драйвера. Данные «потеря 38% при time-slicing» не проверены.
- **WebFetch-ограничение**: инструмент пересказывает страницы маленькой моделью; два пересказа (AgentJet arXiv 2606.04484 и «Expected Comparison Framework» для 2110.00459) оказались недостоверными и были исключены; PDF по ключевым цифрам (SF, SEED, Podracer, TLeague, arXiv 2110.00459, PufferLib 2.0 paper) перечитаны локально через `pdftotext`.
- **Не исследовались**: Hugging Face-системы для игрового RL, Kaggle-специфичные инфраструктуры помимо write-up победителя Lux S3, fabric/pssh/clusterssh, Dagster code locations.

---

## 9. Источники

### Игровой/классический distributed RL
- IMPALA — https://arxiv.org/abs/1802.01561
- SEED RL — https://arxiv.org/abs/1910.06591 ; репо (архив) https://github.com/google-research/seed_rl
- Podracer architectures — https://arxiv.org/abs/2104.06272
- Cleanba — https://arxiv.org/abs/2310.00036 ; https://github.com/vwxyzjn/cleanba
- Sample Factory (ICML 2020) — https://arxiv.org/abs/2006.11751 ; доки https://www.samplefactory.dev/ ; policy lag https://www.samplefactory.dev/07-advanced-topics/policy-lag/ ; multi-agent https://www.samplefactory.dev/03-customization/custom-multi-agent-environments/ ; репо https://github.com/alex-petrenko/sample-factory
- PufferLib — https://puffer.ai/ ; блоги https://puffer.ai/blog/2.0/ , https://puffer.ai/blog/3.0/ , https://puffer.ai/blog/4.0/ , https://puffer.ai/blog/engineering-4.0/ , https://puffer.ai/blog/game-rl/ ; docs https://puffer.ai/docs.html ; releases https://github.com/PufferAI/PufferLib/releases ; статья https://arxiv.org/abs/2406.12905 ; RLC 2025 https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_151.pdf
- SRL — https://arxiv.org/abs/2306.16688 ; https://github.com/openpsi-project/srl
- TLeague — https://arxiv.org/abs/2011.12895 ; https://github.com/tencent-ailab/tleague_projpage
- OpenAI Five — https://arxiv.org/abs/1912.06680
- moolib — https://github.com/facebookresearch/moolib (архив)
- Acme / Reverb / Launchpad — https://github.com/google-deepmind/acme , https://github.com/google-deepmind/reverb , https://github.com/google-deepmind/launchpad (архив)
- EnvPool — https://github.com/sail-sg/envpool
- RLlib: https://docs.ray.io/en/latest/rllib/scaling-guide.html , https://docs.ray.io/en/latest/rllib/key-concepts.html , https://docs.ray.io/en/latest/rllib/multi-agent-envs.html , https://github.com/ray-project/ray/blob/ray-2.59.0/rllib/env/env_runner_state_server.py , https://github.com/ray-project/ray/blob/ray-2.59.0/rllib/algorithms/impala/impala.py , релизы https://github.com/ray-project/ray/releases/tag/ray-2.43.0 , https://github.com/ray-project/ray/releases/tag/ray-2.40.0
- TorchRL: https://docs.pytorch.org/rl/main/reference/collectors.html , https://docs.pytorch.org/rl/main/reference/collectors_distributed.html , https://docs.pytorch.org/rl/main/reference/collectors_weightsync.html , https://docs.pytorch.org/rl/main/reference/generated/torchrl.envs.transforms.PolicyAgeFilter.html , https://github.com/pytorch/rl/releases
- GPU-среды: JaxMARL https://arxiv.org/abs/2311.10090 ; GPUDrive https://arxiv.org/abs/2408.01584 ; Brax https://github.com/google/brax ; MuJoCo Playground https://github.com/google-deepmind/mujoco_playground ; Pgx https://github.com/sotetsuk/pgx ; PureJaxRL https://github.com/luchris429/purejaxrl ; Mava https://github.com/instadeepai/Mava ; Jux https://github.com/sgoodfriend/jux
- Соревнования: Lux AI S3 write-up (1-е место) https://github.com/IsaiahPressman/kaggle-lux-2024/blob/main/write-up.md ; Lux S3 https://github.com/Lux-AI-Challenge/Lux-Design-S3 ; Neural MMO 2023 итоги https://arxiv.org/abs/2508.12524

### LLM-RL
- AReaL — https://arxiv.org/abs/2505.24298 ; https://github.com/inclusionAI/AReaL ; A-3PO https://arxiv.org/abs/2512.06547 ; Decoupled PPO (Hilton et al.) https://arxiv.org/abs/2110.00641
- verl / HybridFlow — https://arxiv.org/abs/2409.19256 ; https://verl.readthedocs.io/en/latest/advance/fully_async.html ; https://verl.readthedocs.io/en/latest/advance/one_step_off.html ; https://verl.readthedocs.io/en/latest/advance/delta_weight_sync.html
- OpenRLHF — https://openrlhf.readthedocs.io/en/latest/async_training.html ; https://openrlhf.readthedocs.io/en/latest/architecture.html
- slime — https://github.com/THUDM/slime
- INTELLECT-2 — https://arxiv.org/abs/2505.07291 ; INTELLECT-3 — https://arxiv.org/abs/2512.16144 ; prime-rl docs https://docs.primeintellect.ai/prime-rl/overview
- PipelineRL — https://arxiv.org/abs/2509.19128 ; https://github.com/ServiceNow/PipelineRL
- ROLL Flash — https://arxiv.org/abs/2510.11345
- Magistral — https://arxiv.org/abs/2506.10910
- Kimi k1.5 — https://arxiv.org/abs/2501.12599 ; Kimi K2 — https://arxiv.org/abs/2507.20534 ; checkpoint-engine https://github.com/MoonshotAI/checkpoint-engine
- LlamaRL — https://arxiv.org/abs/2505.24034 ; Laminar https://arxiv.org/abs/2510.12633 ; StreamRL https://arxiv.org/abs/2504.15930 ; StaleFlow https://arxiv.org/abs/2601.12784 ; AsyncFlow https://arxiv.org/abs/2507.01663 ; ScaleRL https://arxiv.org/abs/2510.13786 ; VCPO https://arxiv.org/abs/2602.17616 ; SparseRL-Sync https://arxiv.org/abs/2605.07330 ; Noukhovitch et al. (async RLHF) https://arxiv.org/abs/2410.18252

### Транспорт, сериализация, GPU sharing
- gRPC performance https://grpc.io/docs/guides/performance/ ; benchmarking https://grpc.io/docs/guides/benchmarking/
- NATS config https://docs.nats.io/running-a-nats-service/configuration ; Redis strings https://redis.io/docs/latest/develop/data-types/strings/
- TensorPipe https://github.com/pytorch/tensorpipe ; torch RPC https://docs.pytorch.org/docs/2.11/rpc.html ; torchcomms https://github.com/meta-pytorch/torchcomms ; CVE-2024-5480 https://asec.ahnlab.com/en/79456/
- Arrow Flight benchmark https://arxiv.org/pdf/2204.03032 ; PEP 574 https://peps.python.org/pep-0574/ ; safetensors https://github.com/huggingface/safetensors ; zstd https://github.com/facebook/zstd
- ActorQ/QuaRL https://arxiv.org/pdf/1910.01055
- NVIDIA MPS https://docs.nvidia.com/deploy/mps/index.html , https://docs.nvidia.com/deploy/mps/when-to-use-mps.html ; характеристика конкуренции GPU (streams/time-slicing/MPS) https://arxiv.org/pdf/2110.00459 ; PyTorch ensembling (stack_module_state + vmap) https://docs.pytorch.org/tutorials/intermediate/ensembling.html

### Деплой
- Ray on-prem https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/on-premises.html ; resources https://docs.ray.io/en/latest/ray-core/scheduling/resources.html ; labels https://docs.ray.io/en/latest/ray-core/scheduling/labels.html ; runtime_env https://docs.ray.io/en/latest/ray-core/handling-dependencies.html ; token auth https://docs.ray.io/en/latest/ray-core/internals/token-authentication.html ; KubeRay https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/config.html , https://docs.ray.io/en/latest/cluster/kubernetes/user-guides/configuring-autoscaling.html
- Kueue https://kueue.sigs.k8s.io/docs/overview/ ; Volcano https://volcano.sh/en/docs/ ; JobSet https://jobset.sigs.k8s.io/docs/overview/ ; Kubeflow Trainer https://www.kubeflow.org/docs/components/trainer/overview/
- Slurm hetjob https://slurm.schedmd.com/heterogeneous_jobs.html ; submitit https://github.com/facebookincubator/submitit ; SkyPilot https://docs.skypilot.ai/en/latest/getting-started/quickstart.html , https://docs.skypilot.ai/en/latest/reservations/existing-machines.html ; Dask SSH https://docs.dask.org/en/stable/deploying-ssh.html ; Prefect workers https://docs.prefect.io/v3/concepts/workers
- kubeadm bootstrap tokens https://kubernetes.io/docs/reference/access-authn-authz/bootstrap-tokens/ ; Node leases https://kubernetes.io/docs/concepts/architecture/leases/ ; k3s https://docs.k3s.io/quick-start , https://docs.k3s.io/networking/distributed-multicloud ; Nomad client https://developer.hashicorp.com/nomad/docs/configuration/client , drain https://developer.hashicorp.com/nomad/docs/commands/node/drain
- GitHub runners https://docs.github.com/en/actions/hosting-your-own-runners/managing-self-hosted-runners/adding-self-hosted-runners ; Buildkite https://buildkite.com/docs/agent/v3 ; Celery workers https://docs.celeryq.dev/en/stable/userguide/workers.html ; torchrun elastic https://docs.pytorch.org/docs/stable/elastic/run.html
- Tailscale auth keys https://tailscale.com/kb/1085/auth-keys ; Modal images https://modal.com/docs/guide/images ; uv docker https://docs.astral.sh/uv/guides/integration/docker/ ; HPA https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/ ; KEDA https://keda.sh/docs/latest/concepts/scaling-deployments/ ; GPU scheduling https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/

### Локальные артефакты
- Исходники Colosseum, прочитанные для оценки: `/home/viv/dev/repos/Colosseum/src/colosseum/{distributed.py,transport/grpc_transport.py,transport/serialization.py,weight_store/grpc_store.py,worker/rollout_worker.py,learner/learner.py,core/config.py}`, `/home/viv/dev/repos/Colosseum/proto/colosseum.proto`.
- Микробенчмарки (одноразовые, вне репозитория): `/tmp/bench_scripts/`, `/tmp/r1_bench/infer.py`.

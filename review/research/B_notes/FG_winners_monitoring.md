# Заметки F+G: что делали победители соревнований-ботов и как мониторить лигу

Дата сбора: 2026-10-07. Автор: research-агент для проекта Colosseum (APPO IMPALA-style, BC -> self-play -> PFSP/league, команда 1-3 человека).
Язык: русский, термины английские. Пометка "[не подтверждено]" = источник не найден/не открылся/это моя интерпретация.

## 0. Методология и ограничения источников

- WebSearch оказался слабым (мусорная выдача), поэтому основные факты взяты напрямую из первоисточников:
  - Kaggle discussion-посты. Страницы JS-рендерные и через WebFetch не открываются, но работает внутренний JSON-эндпоинт
    (проверено через curl, без авторизации):
    `POST https://www.kaggle.com/api/i/discussions.DiscussionsService/GetForumTopicById` с телом `{"forumTopicId":<ID>,"includeComments":true}`
    -> `forumTopic.firstMessage.rawMarkdown` + `comments[].rawMarkdown`.
    Список тем: `POST .../discussions.DiscussionsService/GetTopicListByForumId` с `{"forumId":..,"sortBy":"TOPIC_LIST_SORT_BY_HOT","group":"TOPIC_LIST_GROUP_ALL","page":N,"pageSize":20}`;
    forumId берётся из `POST .../competitions.CompetitionService/GetCompetition {"competitionName":"lux-ai-2021"}`.
    Ответы авторов внутри комментариев (nested replies) этим способом не возвращаются, поэтому часть вопросов "сколько GPU/дней" осталась без ответа.
  - GitHub raw (README победителей), arXiv/PDF через curl+pdftotext, Context7 (документация W&B/MLflow/Aim/ClearML), docs-сайты (samplefactory.dev, docs.ray.io).
- Не открылось/удалено: Hungry Geese 1st place writeup (Kaggle topic 263279 помечен "[Deleted Topic]"); Nature-версия AlphaStar (редирект на логин; использован PDF DeepMind).
- Для CodeCraft, Battlecode, Pommerman данные поверхностные (см. разделы).
- Важное наблюдение про рынок инструментов: Neptune.ai закрыт (SaaS остановлен 2026-03-05, см. раздел G.5), W&B docs переехали на docs.coreweave.com.

---------------------------------------------------------------------------------------------------

# ЧАСТЬ F. Что реально делали победители

## F.0 Сводная таблица (кто выиграл и чем)

| Соревнование | 1-е место | Подход победителя | Роль BC/IL | Роль RL |
|---|---|---|---|---|
| Lux AI S1 (2021, Kaggle) | Toad Brigade (IsaiahP + Liam + Rob) | Deep RL: IMPALA + UPGO + TD(lambda) + KL к frozen teacher, ResNet 24 блока 128ch (~20M params) | нет (RL с нуля, потом teacher-chain) | основной |
| Lux AI S1, топ-10 | топ-4..12 в основном IL | Imitation learning с реплеев лидеров (4, 5, 6, 11, 12 места) | основной у большинства | у RLIAYN (2-е) - PPO + PFSP |
| Lux AI S2 (2023, Kaggle/NeurIPS) | ry_andy_ (Ryan Anderson) | Rule-based + forward simulation 2.9 c/ход, роли/цели | нет | нет |
| Lux AI S2, лучший RL | FLG (4-е), Deimos (10-е) | IMPALA/V-trace + teacher KL (FLG); PPO RLlib + JAX env (Deimos) | IL использовался для поиска архитектуры (FLG) | да |
| Lux AI S3 (2024-25, NeurIPS) | Flat Neurons (TonyK, kat_ies и др.) | Multi-agent RL: IMPALA/V-trace + UPGO + teacher KL + frozen opponent pool | BC-режим был реализован, но НЕ использован в финале | основной, ~20B env steps суммарно |
| Lux AI S3, 2-е | Frog Parade (IsaiahP + Garrett) | PPO, Rust-симулятор и фичи, ResNet-SE 8 блоков 256ch (10M) | нет | основной |
| Lux AI S3, 3-е | aDg4b | Imitation Learning (2 UNet) на реплеях Frog Parade и Flat Neurons | основной (100%) | нет |
| Halite IV (2020) | Tom Van de Wiele (tvdwiele/ttvand) | Rule-based, 11k+ строк; Deep RL пробовал месяц, провал | нет | нет (отвергнут) |
| Hungry Geese (2021) | HandyRL (kyazuki, YuriCat; DeNA) | RL (HandyRL: IMPALA-like, UPGO/V-trace/TD) + поиск (по косвенным данным); writeup удалён | [не подтверждено] | основной |
| Hungry Geese 2-е | Goosebumps (IsaiahP, lpkirwin) | BC + MCTS (PUCT) | основной | не вошёл в финал |
| Hungry Geese 3-е | YuyaYamamoto, maxwell110, niwatori | BC pretraining + RL (HandyRL) + MCTS; local TrueSkill "Coliseum" | да | да |
| Hungry Geese 5-е | takedarts | AlphaZero self-play (без внешних данных) | нет | self-play AZ |
| Google Research Football (2020) | WeKick (Tencent: Ziyang Li, fenngming и др.) | RL+self-play: PPO+LSTM, multi-head value, league, GAIL | GAIL для разнообразия оппонентов | основной |
| GRF 2-е/3-е/5-е | SaltyFish / Raw Beast (T. Van de Wiele) / TamakEri (HandyRL) | RL: распределённый IMPALA-подобный; Raw Beast = clone + RL | да (Raw Beast, SaltyFish пул) | основной |
| Kore 2022 | Harm Buisman | Rule-based (порт Kore Beta 1st, +5923 строк) | нет | нет |
| Neural MMO 2022 (NeurIPS) | realikun | Standard RL + доменная инженерия; первое во всех треках | в IL-треке отдельно | основной |
| Neural MMO 2023 (NeurIPS) | Takeru и Yao Feng (co-winners) | PPO через CleanPuffeRL (PufferLib), лимит 8 A100-часов | нет | основной |
| MineRL BASALT 2022 | GoUp | Скрипты (human knowledge) + ML | BC (VPT fine-tune) в базе | частично |
| Pommerman NeurIPS 2018 | HakozakiJunctions (1-е), dypm-final (3-е) | Tree search with pessimistic scenarios; Navocado = лучший learning-агент (A2C) | - | RL уступил поиску |
| Battlecode | 4 Musketeers (2023), Cout for Clout (2024, 2-е в общем) | Hand-coded rule-based | нет | нет (в постмортемах не упомянут) |
| ConnectX | см. F.9 | Решатели/minimax; RL лишь временно лидировал | - | - |

Главный вывод таблицы: в соревнованиях с большим числом юнитов/длинным горизонтом (Halite, Kore, Lux S2, Battlecode) побеждали
правила + симуляция; RL выигрывал там, где (а) есть быстрый симулятор, (б) один "центральный" контроллер (карта -> действия на все клетки,
Lux S1/S3), (в) мало дискретного состояния и много данных (GRF, Hungry Geese, NMMO). IL с реплеев лидеров - самый дешёвый путь в top-5
для команды без кластера (Lux S1: 5 из 14 перечисленных топ-решений - чистый IL; Lux S3: 3-е место).

## F.1 Lux AI Season 1 (2021)

Источники: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021 (README = полный writeup, hall_of_fame/ с сабмишнами),
https://www.kaggle.com/c/lux-ai-2021/discussion/294993 (1st), https://www.kaggle.com/c/lux-ai-2021/discussion/300844 (RLIAYN, 2nd),
https://www.kaggle.com/c/lux-ai-2021/discussion/294459 (таблица подходов топов), https://www.kaggle.com/c/lux-ai-2021/discussion/296519 (итоги).
Масштаб: 1449 участников / 1178 команд, 22331 сабмишн (296519).

### Toad Brigade (1st)
- Команда: Liam (rules-агент), Isaiah (RL), Rob (мета-анализ игр и слабых мест). Изначально ставка на rules (по опыту Halite),
  но в течение первого месяца RL обогнал rules-агента и улучшался монотонно; rules-агент забросили с August sprint.
- Алгоритм: IMPALA (FAIR TorchBeast/monobeast, `run_monobeast.py`) + UPGO + TD(lambda) loss terms; **frozen teacher model** + KL(policy || teacher)
  "для стабилизации и против стратегических циклов, которые мучают чистый self-play".
- Action/obs: один fully-convolutional ResNet-SE (24 блока, 128ch, 5x5, без нормализации, ~20M params) выдаёт действия сразу для ВСЕХ клеток
  (workers/carts/city tiles); учитываются только клетки с юнитами. Лог-вероятности суммируются по юнитам. Padding до 32x32 с маскированием после каждого conv.
  Illegal actions маскируются logit=-inf. Game phase embedding (turn//40) - "crucial part of success".
- Reward: reward shaping только первые 20M шагов (за города/юниты/research/fuel), затем sparse +-1 (win/loss).
- Progressive scaling: 8-block (shaped) -> 16-block -> 24-block, каждый следующий на sparse reward с предыдущей сетью как teacher.
- Железо: "All training was done on my personal PC - an 8-core/16-thread dual-GPU system". Обучение "overnight most nights" весь соревновательный период
  (~4 месяца). RLIAYN в своём постмортеме оценил: "training time (4 days vs 20 days)" - т.е. Toad Brigade ~20 дней суммарно [оценка конкурента, не самого автора].
- Inference: усреднение с поворотом на 180 градусов, greedy actions (не sampling), ручные правила разрешения конфликтов (порядок по вероятности модели),
  модель ~2-2.5 c/ход на Kaggle при batch 2 (лимит времени был главным ограничителем размера).
- Слабости (из их же разборов): "agent gives up" после шага 200 в проигрышных партиях; авторы сами пишут, что лёгкий reward shaping или лига
  разнообразных оппонентов могли бы помочь.
- Выбор сабмита: [не подтверждено]; в репо `internal_testing/hall_of_fame` хранится история сабмитов (по сути ручной архив чемпионов).
  Kaggle-ladder = TrueSkill-like score; после 2 недель финализации LB (причём nosound жаловался "LB match making, scoring and final submissions policy are BAD").

### RLIAYN (Theo Cachet; 2nd)
- PPO "с нуля" (без RL-библиотек), PFSP-самоигра по прошлым чекпоинтам, KL-дистилляция предыдущих моделей, сеть ResNet 17 блоков x 32ch, ~300k params (!).
- Reward: hand-crafted dense (score diff: farming + Voronoi-control + citytiles + win/loss +-50). Чистый sparse reward с нуля дал максимум Elo ~1200 "после
  нескольких дней" - т.е. dense был необходим, в отличие от Toad Brigade, который сначала шейпил, потом снял.
- Железо: RTX 2080Ti + i9-9900k, ~300 env steps/s (~1M steps/час), 80% времени в env.step (Kaggle JS-движок). ~4 дня обучения, плато не достигнуто.
- BC использован: (1) для выбора фичей и архитектуры (BC на city tiles как быстрый прокси качества архитектуры), (2) отдельная BC-модель для city tiles из данных top-5.
- Ключевой вывод авторов: "the biggest difference between the two approaches is scale" - время обучения и размер сети.

### Imitation Learning топ-решения
- 4th Team Durrett (https://www.kaggle.com/c/lux-ai-2021/discussion/296938): IL с 7 лучших агентов, shared CNN + отдельный FC-head на каждого агента-донора,
  при инференсе используется head лучшего; ~1 неделя обучения финальной модели, много эпох критично.
- 5th ironbar (https://www.kaggle.com/c/lux-ai-2021/discussion/293911, https://github.com/ironbar/luxai): Conditioned UNet (24M params), условие = one-hot "кого имитировать";
  данные: матчи всех агентов с LB-score > 1700 на 01.12.2021 (82 агента, ~16k матчей); 1 GPU 3090, <1 суток обучения (ПК с 2x RTX 3090). Автор признаёт: хорош потому,
  что Toad Brigade был на ~300 LB-очков выше, т.е. IL копирует лидера, но не превосходит.
- 6th nosound (https://www.kaggle.com/c/lux-ai-2021/discussion/293776): чистый IL; fine-tune последних 3 слоёв только на реплеях Toad Brigade; "multiple attempts at RL did not succeed".
  Уроки RL от него: IL-прототип позволяет проверить архитектуру дёшево; быстрый симулятор критичен; value-функция должна быть точной (учить её отдельно на реплеях);
  глобальный critic в multi-unit среде даёт шумный advantage.
- 8th A.Saito (https://www.kaggle.com/c/lux-ai-2021/discussion/294603): IL policy + value net + MCTS на C++ (policy-net-only ~1530-1550 LB, с MCTS и value ~1660-1700, финал 1743).
- Таблица chimuichimu (294459): топ по способу - Toad Brigade (RL); 4,5,6,11,12,20 - IL; 8 - IL+MCTS; 15 - rules; 16 - IL+rules.
- Data selection у IL-команд: только матчи, где ВСЕ участники выше порога LB (Hungry Geese 2nd делали то же), порог растёт по ходу соревнования.

### Инструменты ladder-мониторинга Lux S1
- https://www.kaggle.com/c/lux-ai-2021/discussion/281823: Streamlit+Plotly+Selenium веб-приложение lux-ai-stats (score growth, win-rate EMA, W/L по submission ID).
- Episode scraper notebook: https://www.kaggle.com/robga/simulations-episode-scraper-match-downloader (используется Hungry Geese 2nd/3rd и многими IL-командами).
- "Tricks List" про PPO-реализацию (https://www.kaggle.com/c/lux-ai-2021/discussion/283883): 25 имплементационных деталей PPO (advantage norm per minibatch, reward scaling/clipping,
  orthogonal init, grad clip 0.5, Adam eps 1e-5, value clipping, obs normalization) - чек-лист для сверки нашего APPO.

## F.2 Lux AI Season 2 (2023)

Источники: https://www.kaggle.com/competitions/lux-ai-season-2/discussion/407982 (1st), /406702 (FLG, 4th), /409394 (Harm Buisman, 5th), /411725 (Deimos, 10th),
/405476 (Tigga), /404842 (IL notes), /408186 (recap). Масштаб: 757 участников / 651 команда, 1481 сабмишн (recap).

- 1st: Ryan Anderson (ry_andy_) - rules + forward simulation (~2.9 с на ход, планирование на 5-50+ шагов), роли/цели (antagonizer, miner, pillager и т.д.), приоритетный порядок
  фиксации действий; ставка на lichen как источник энергии. Топ-3/4 (ry_andy_, ttigga, danmctree, FLG) шли очень близко и менялись в последний день. Порядок 2/3 [не подтверждено].
- Tigga: stateful rule-based на TypeScript; self-play партии >20 минут, нужно 10+ часов на 8 ядрах для разумных данных (показатель, что тяжёлая среда убивает RL).
- 4th FLG (RL, https://www.kaggle.com/competitions/lux-ai-season-2/discussion/406702) - самое информативное RL-решение сезона:
  - Single-learner-multi-actor на Python multiprocessing + queues, PyTorch, без RL-фреймворка (близко к нашей архитектуре); V-trace (PPO давал похожее на 16x16/24x24).
  - Losses: TD(lambda) value (lambda 0.95, gamma 0.9995), V-trace policy gradient (7 actor heads), entropy 1e-5..1e-4, teacher KL ~5e-3; teacher держат ~30M шагов позади агента.
  - План: упростить action/obs -> IL+RL на маленьких картах для выбора архитектуры (датасет ~2000 матчей топ-агентов через MetaKaggle, HDF5 ~300GB) -> малая модель RL ->
    большая с teacher -> оптимизация под CPU inference. Сетевая архитектура "DoubleCone" (часть вычислений на 12x12 вместо 48x48) выбрана по accuracy на IL-датасете.
  - Реварды: ~65M шагов "selfish" dense shaping, затем zero-sum (-1..1 за результат, lichen advantage и др.), "agent was very robust to reward changes".
  - Боль: на Kaggle только CPU (часто 1 ядро), нельзя поставить onnxruntime/OpenVINO; не удалось запустить большую модель -> отмена обучения DoubleCone(6,8,6) на ~100M шагов.
  - Сэмплирование редких событий (bidding/spawn ~1% шагов) решено отдельным "spawn actor" с апсемплингом: полезный приём для редких фаз.
  - Железо/дни: [не указано].
- 5th Harm Buisman: rules + route search (позже выиграл Kore 2022).
- 10th Deimos (RL): PPO по реализации RLlib, JAX-версия среды, 48x48x30 карта + action queue; команда "top RL contestant" в середине сезона (по словам Harm).
- IL-notes (404842): supervised imitation агентов Deimos - работает, т.к. у них простое action space.
- Итог: на длинном горизонте (1000 шагов), сложной энергетике и action queue победили rules с симуляцией; RL остался в топ-10, но проиграл. 

## F.3 Lux AI Season 3 (NeurIPS 2024, финал 2025-03)

Источники: https://www.kaggle.com/competitions/lux-ai-season-3/discussion/569919 (итоги), /569562 (1st), /568621 (2nd), /568494 (3rd IL), /569928 (4th IL),
/570673 (8th IL), /571111 (5th xLSTM RL), /568789 (9th), /570196 (10th), /568721 (EcoBangBang RL), /567961 (3Comets 14th). 704 команды, 800+ участников.
Решения 2025 УЖЕ известны (опубликованы 2025-03/04).

### 1st Flat Neurons (https://www.kaggle.com/competitions/lux-ai-season-3/discussion/569562; код https://github.com/tonykozlovsky/lux-ai3-pub)
- Multi-agent RL, IMPALA + V-trace + UPGO + TD loss + entropy; **dynamic reward scaling** (скользящее среднее returns по ~5000 батчам -> множитель, чтобы returns в [-5,5])
  и **dynamic entropy** (target entropy на голову линейно падает 0.9->0 (move) и 3.9->0 (sap) за 100M шагов; коэффициент подстраивается автоматически; после 100M - перезапуск с меньшими целями).
- Teacher KL + teacher baseline loss (frozen лучшая модель) против forgetting; frozen opponent pool (часть игр против старых моделей), BC-смешивание реплеев было реализовано,
  но отключено ("already seeing significant improvements without it").
- Сеть: ~1000+ бинарных/дискретных признаков на клетку 24x24 -> 128ch, 24 residual blocks, ConvLSTM, 4 Transformer-блока, патч 15x15 на юнит; доп. supervised head предсказания позиций врагов.
- Reward итог: sparse +-1 за матч + +-2.5 суммарно за очки реликвий (чтобы не "застаивался").
- Масштаб: финальная модель ~3-4 дня на ~1.5B env steps (итерации по 200M); за соревнование >20B шагов во всех экспериментах; bfloat16, torch.compile (~1.5x). Тип/число GPU и CPU [не указаны].
- Оценка: тысячи self-play матчей против старых версий и teacher; "real-time win rates" против старых/teacher во время обучения.
- **Тестирование на LB без утечки IL-подражателям**: два агента в одном сабмите; слабый играет 85% партий, сильный 15%; логируется, кто играл; по `main_submission_id/enemy_submission_id/is_strong`
  считается winrate. Позже - шум на логиты сильной модели или использование сильной в отдельных раундах матча. Причина: IL-команды копируют сильнейших по реплеям.
- Inference на Kaggle: greedy, флип-аугментация (отключалась при >30 с overtime).
- Выбор финала: Kaggle берёт 2 последних сабмита (см. ниже); они держали до конца "проверенный" top-3 и сильную модель в одном сабмите.

### 2nd Frog Parade (IsaiahP + Garrett; https://www.kaggle.com/competitions/lux-ai-season-3/discussion/568621; https://github.com/IsaiahPressman/kaggle-lux-2024)
- Симулятор правил и feature engineering переписаны на Rust (PyO3/Maturin), TDD (интеграционные тесты против настоящего движка), ~10.8k строк Rust + ~6.5k Python.
- PPO (clip, GAE, gamma 0.9999-1.0, masking, entropy, teacher KL), ResNet-SE 8 блоков d_model=256 (~10M params, 62MB запаса до лимита 100MB), value-голова "видит обе команды" (softmax двух значений).
- Reward: sparse win/loss +-1 весь основной период.
- Железо: Ryzen 9950X (16c/32t), 64GB RAM, RTX 3090 + RTX 2070 Super. Симулятор 110k steps/s без модели; финальная модель 430 steps/s (GPU - bottleneck), маленькая 420k params - 2800 steps/s.
  ~300M игровых шагов (600M per-player observations), ~8 дней непрерывного обучения, плато к ~200M.
- Метрики логировали в W&B: loss-термы, средние очки, частоты действий, winrate против предыдущей лучшей модели.
- Ретроспектива автора: слишком узкий action masking (blind sap), следовало сильнее масштабировать модель.

### 3rd aDg4b: чистый IL (https://www.kaggle.com/competitions/lux-ai-season-3/discussion/568494; https://github.com/w9PcJLyb/lux3-bot)
- Rule-based упёрся -> IL: два UNet (Unit-UNet и SAP-UNet), "несколько дней на скачивание/подготовку данных, пара часов обучения".
- Данные: реплеи Frog Parade и Flat Neurons (!); при проигрыше берутся только выигранные матчи; фильтр "решённых" матчей; 95% кадров "все Center" отбрасываются.
- Критичный шаг - препроцессинг: пересчёт скрытых состояний (reward-позиции, скрытые константы) своим кодом по наблюдениям донора (аналог space.update в публичном Relicbound bot).
- Урок для нас: IL-конкуренты используют ваши ladder-реплеи; публичная лига "протекает". Flat Neurons это знали и скрывали силу.

### Прочие RL и IL решения S3
- 4th YumeNeko (IL), 8th Gregor Lied (IL) - также IL от лидеров.
- 5th Kiwis (OneUpKiwi): xLSTM + Transformer encoder, recurrent PPO на JAX; self-play: 25% против последних 128 чекпоинтов, 75% против последнего.
- 10th Boey (https://www.kaggle.com/competitions/lux-ai-season-3/discussion/570196): PureJaxRL, end-to-end JAX, 80k steps/s, модель 1.8M (+1.4M critic);
  PFSP по образцу AlphaStar: 75% self-play, 25% замороженные прошлые версии, PFSP-приоритет по win rate; RTX 4090 + Ryzen 7950x; сабмиты: 58B шагов/8 дней (LB 1771.5) и 23B/7 дней с центральным critic (LB 1884.5);
  вывод: больше параметров (10M+) стоило бы пропускной способности; центральный critic ~2x sample efficiency, но 40k vs 80k SPS.
- EcoBangBang (Fei Wang; https://www.kaggle.com/competitions/lux-ai-season-3/discussion/568721): RL по коду Toad Brigade; ключевые находки: 50+ дней RL не работал,
  заработал после перехода на zero-sum baseline + удаления UPGO-loss (его величина на порядок превышала остальные в дашборде) -> рост до 1900+.
  Т.е. дашборд "величина компонентов loss" напрямую спас решение.
- 3Comets 14th (https://www.kaggle.com/competitions/lux-ai-season-3/discussion/567961): MAPPO, RNN-память, 15 суток на одной RTX 4090 (vast.ai), >1B env steps, ~500 USD на все эксперименты.

### Правила финального выбора на Kaggle (Lux S3)
- Играют только последние 2 сабмита (https://www.kaggle.com/competitions/lux-ai-season-3/discussion/554992, /557243, /567001): "we only evaluate your last 2 submissions".
  Совет SiestaGuru: заливать пробные, а лучшее перезалить ближе к концу; мета смещается, поэтому периодически перезаливать лучшее.
- Ladder = TrueSkill-подобный рейтинг, вариативность велика (см. Raw Beast в GRF: одинаковые агенты расходились на ~90 очков за неделю).

## F.4 Halite IV (2020)

Источники: https://www.kaggle.com/c/halite/discussion/183543 (1st), https://github.com/ttvand/Halite, /183312 (8th IL), /186032 (2nd), /169623 (RL from random play), /164644.
- 1st Tom Van de Wiele (tvdwiele): rule-based, "scores -> plans -> actions", >11k строк. Подал 22 сабмита, 16 из них заняли бы 1-е место.
  Deep RL пробовал ~месяц и бросил: "credit assignment hard with arbitrary number of units, long episode, dynamic opponent pool; not having infinite compute is also annoying".
  Модели оппонентов (risk model, conversion threshold) учатся на лету по наблюдаемым действиям соперников. В репозитории остались папки Deep Learning Agents, Rule agents (evolution I..X),
  Stable opponents pool, Leaderboard simulation/replays - то есть он строил локальную симуляцию против замороженных агентов с LB.
- 2nd Raine Force (dereview): 100% rule-based; считал таблицы win/loss против топ-5 агентов других команд (локальный head-to-head) - выбирал сабмит по матрице попарных результатов.
- 8th Kha Vo с командой: imitation learning через semantic segmentation (UNet по клеткам) + эвристические override (spawn, convert, defense); код https://github.com/digitalspecialists/halite4.
- 5th Panpan Zhou: Shipwise ML agent. 4th: rules.
- Вывод для нас: для задач "много юнитов + длинный эпизод + быстро меняющиеся правила" RL-инфраструктура без крутого симулятора не окупилась; IL как "мост" дал топ-8.

## F.5 Hungry Geese (2021)

Источники: https://www.kaggle.com/c/hungry-geese/discussion/264053 (итоги), /263686 (2nd), /263735 (3rd), /263702 (5th), /218190 (HandyRL), https://github.com/DeNA/HandyRL,
https://speakerdeck.com/hoxomaxwell/kaggle-hungry-geese (3rd deck). 1039 участников / 875 команд, 33296 сабмишнов.
- 1st: HandyRL team (kyazuki, YuriCat; DeNA). Writeup на Kaggle удалён ([Deleted Topic] 263279) [детали не подтверждены]. Известно:
  - README HandyRL: "The 1st place solution in Hungry Geese", а также 5th place в GRF; IMPALA-подобный learner-worker, off-policy corrections: Monte Carlo, TD(lambda), V-Trace, UPGO.
  - Публичная модель (обучение 1 день, 1 GPU + 64 CPU, ~Rating 1100): gamma 0.8, forward_steps 32, lambda 0.7, batch 400, maximum_episodes 500000 (RAM 64GB), update_episodes 500 (/218190) - это НЕ финальный агент.
  - Из текста 2nd-place: HandyRL "в конце комбинировал сильную policy self-play-сети с быстрым скомпилированным поиском" с отличным результатом [косвенное свидетельство от соперника].
  - Лидировали "большую часть соревнования", финальный отрыв от Goosebumps - ~1 очко LB (/264053).
- 2nd Goosebumps (IsaiahP + lpkirwin + vishyvishal): BC на реплеях топ-агентов (data selection: только эпизоды, где худший участник выше порога LB, порог растёт) + PUCT MCTS;
  скриптовая сторона на Numba/Rust. 8C/16T dual-GPU PC, "ограничивает CPU-intensive RL". Пробовали A2C/IMPALA self-play на GPU-среде (сотни-тысячи env на 8GB GPU, 5-6 игр/с,
  компетентный агент за час), но policies "too close to deterministic to combine with MCTS". Также онлайн-обучаемый вес "непредсказуемости оппонента" (KL-минимизация к реальным действиям).
- 3rd (YuyaYamamoto + maxwell110 + niwatori): цикл BC (pretraining на скрейпленных эпизодах с LB>1200 + эпизоды собственных MCTS-агентов с GPU на vast.ai, 12x1080Ti) ->
  RL на HandyRL (UPGO, в конце V-trace; forward_steps 12->72->12; gamma 0.8/0.97) -> MCTS; ResNet 8 слоёв 46ch.
  **Локальная лига "Coliseum"**: стадия 1 - 224 матча против стандартного агента (~LB1200), агенты с winrate <0.55 отсеиваются; стадия 2 - пул прошедших, TrueSkill-рейтинг (оценка устойчивости),
  результаты выводились в Slack как "LB"; GCP-инстансы на 224 и 96 CPU; финальный ансамбль LB 1239.1. Идея: "RL via self-play = hating its own past; RL by imitating opponents adapts to meta-game".
- 5th takedarts GeeseZero: AlphaZero без внешних данных и RL-фреймворков, DUCT; ~350k игр на Threadripper 3970X.
- Урок: комбинация BC + RL + поиск, плюс локальный турнир-пул; чистый RL-агент без поиска слабее.

## F.6 Google Research Football (Kaggle 2020)

Источники: https://www.kaggle.com/c/google-football/discussion/202232 (WeKick), /202977 (SaltyFish 2nd), /200709 (Raw Beast 3rd), /203412 (TamakEri 5th, HandyRL), /204645 (итоги),
TiKick https://arxiv.org/abs/2110.04507.
- WeKick (1st; Tencent AI Lab: Ziyang Li, Kaiwen Zhu и др.): асинхронная архитектура как у Honor of Kings (Ye et al.), PPO (openai baselines) + LSTM (32 шага, 256 hidden),
  multi-head value (reward-компоненты с разными gamma), доп. фичи (relative pose, offside flag), zero-sum shaped reward ("making it zero-sum is very important for self-play"),
  **GAIL по реплеям других команд** (kangaroo) -> такие модели используются как фиксированные оппоненты для робастности; **league training** по AlphaStar со стилями (counter-attack, short pass, holding ball).
  Оценка: из-за редкого LB (2-3 дня на оценку сабмита) держали **локальный LB** (model pool + ELO), ранг сильно коррелировал с публичным LB. Заметка: "internal LB: final model against all styles ~+100 Elo vs silver candidate".
  Железо: [не указано; вопрос про CPU/GPU в комментариях без ответа в доступных данных].
- 2nd SaltyFish: 80 CPU-ядер до соревнования (~190k эпизодов/нед, 2 недели -> ~1400 LB) -> на соревнование 1000 CPU-ядер на эксперимент (20k+ fps), до 2500 ядер в последние 2 недели; GPU не нужны (векторный вход);
  ELO-выбор оппонента ~ равен "новейший оппонент с вероятностью 70%"; curriculum + imitation WeKick.
- 3rd Raw Beast (Tom Van de Wiele): clone-политики (cross-entropy на действиях целевых сабмитов + entropy) как старт, потом IMPALA-подобный RL на PyTorch, replay buffer 3000-30000;
  **внутренний LB из разнообразных агентов (rules, публичные, клоны, RL)** потому что публичный LB имеет слишком высокую дисперсию (два одинаковых агента через неделю различались на ~90 очков). Всё в одном контейнере.
- 5th TamakEri (HandyRL): off-policy distributed RL, UPGO(lambda), 1-step Retrace; воркеры 96-core CPU, learner 24-core + 1xT4 (small) / 8x96-core + 4xT4 (large);
  в последние 3 дня ~4000 CPU и ~20 GPU параллельно; финальный агент сыграл ~700k игр; дистилляция прошлой модели как teacher; не смогли залить последнюю модель (нельзя сабмитить во время скоринга предыдущей).
- TiKick (WeKick как эксперт -> multi-agent offline RL): подтверждает статус WeKick "first place... imitation learning, multi-head value trick, distributed league training" (arXiv 2110.04507).
- Уроки: zero-sum shaping, internal ELO-лига, разнообразие стилей через reward/GAIL; масштаб CPU ~1000+ ядер у топов.

## F.7 Kore 2022 / Kaggle Simulations: RL против поиска/правил

- Kore 2022 (https://www.kaggle.com/competitions/kore-2022/discussion/340035): 1st Harm Buisman, rule-based; top-5 по заголовкам постов в основном rules
  (2nd basilisk1337 rules, 4th qihuaz rules, 5th rules, 3rd attack second shipyard); 13th - IL с языковой моделью (Fumihiro Kaneko, /337476); Kore Beta 1st (w9PcJLyb) - rules.
  Harm: "version control, список issues, итерации"; 26 апдейтов за две недели, чтобы обойти собственный предыдущий агент; ранг "was a bit unstable".
- ConnectX: RL временно лидировал по LB (Tom V.d.W. об этом: https://www.kaggle.com/competitions/connectx/discussion/129145), в итоге выигрывают решатели/поиск (solved game) [детали не подтверждены].
- Lux S1->S3 показывает смещение: от rules/IL к RL по мере появления быстрых симуляторов (JAX у S3) и open-source baselines.

## F.8 Neural MMO, MineRL/BASALT, Pommerman, Battlecode, CodeCraft

### Neural MMO (NeurIPS)
- 2022 (https://proceedings.mlr.press/v220/liu23a.html): 500 участников, >1600 сабмишнов; realikun - 1st во всех треках; "top submissions: mostly standard RL + domain-specific engineering";
  scripted-агенты быстро стартовали, но к концу уступали RL и переобучались под PvE-оппонентов; один IL-сабмит занял 5-е место (данные - от RL-политики); PvP-оценка через TrueSkill (учитывает ранг всех политик в матче);
  daily (100 матчей) и weekly (1000 матчей) evaluation на k8s; бейзлайн достигает 0.5 Top-1 ratio на PC+GPU за сутки.
- 2023 (https://arxiv.org/abs/2508.12524, "Results of the NeurIPS 2023 Neural MMO Competition on Multi-task RL"): >200 участников; лимит обучения **8 A100-часов и 12 CPU**; топ ~4x выше baseline за 8 часов на одной 4090;
  Co-winners Takeru (Jianming Gao, Yunkun Li) и Yao Feng: PvP 25.21% / 24.88% против baseline 6.39%; baseline - Clean PuffeRL (PufferLib), 3.9M params;
  ключевая правка PufferLib: убрана zero-padding наблюдений (при >75% паддинга эффективный batch size падает и дестабилизирует PPO). Оценка победителей: PvE по 32 эпизодам x 4 seeds, PvP 9 политик, 9 раундов, 1800 эпизодов;
  победители переобучены организаторами с нуля (8 ч, 15-22M шагов) для проверки честности.
- Уроки: на стандартных RL + аккуратной реварде/фичах выигрывают; организаторы использовали PufferLib/CleanRL, т.е. мы можем перенять их паттерн "лимитированный compute + воспроизводимость".

### MineRL BASALT 2022 (https://arxiv.org/abs/2303.13512)
- 1st GoUp (2.09 score): задача разделена на скриптуемую часть (human knowledge) и ML-часть; 2nd UniTeam, 3rd voggite; KAIROS - research prize за естественное сочетание RL+IL. База - fine-tuned VPT с BC.
  Оценка людьми-судьями, не ladder [детали подтверждены только по аннотации/поиску].

### Pommerman NeurIPS 2018 (https://arxiv.org/abs/1902.10870)
- Tree search with pessimistic scenarios занял 1-е (HakozakiJunctions) и 3-е (dypm-final) места; Navocado (A2C) - лучший learning-агент. То есть поиск > RL в основной номинации.

### Battlecode
- Постмортемы победителей (https://battlecode.org/assets/files/postmortem-2023-4-musketeers.pdf, .../postmortem-2024-cout-for-clout.pdf): hand-coded стратегия, коммуникация, pathfinding, bytecode-оптимизация; RL в постмортемах не фигурирует [выводы по поиску, PDF полностью не читал].

### CodeCraft
- "CodeCraft" как исследовательская RTX-среда: Clemens Winter, "Mastering RTS games with deep RL: Mere Mortal Edition" (https://clemenswinter.com/2021/03/24/mastering-real-time-strategy-games-with-deep-reinforcement-learning-mere-mortal-edition/):
  4x RTX 2080 Ti (по 2 запуска на карту), 32-core Threadripper 2990WX, 8 параллельных процессов, 125M samples "чуть больше 2 суток" (менее суток без evals), self-play + PPO/GAE, 64 env x 128 шагов rollout,
  Automatic Domain Randomization и curriculum по размеру карты, omniscient value function. Win-rate vs scripted Replicator 90-98%, Destroyer ~70%. Оценка: каждые 5M samples ~500 игр, финал ~3000 игр.
- Отдельно существовал турнир AI Cup 2020 "CodeCraft" (Mail.Ru) - победители и роль RL [не подтверждено].

## F.9 Общие уроки (сжатая выжимка)

1. Железо (реальные цифры из постов):
   - Одиночная персональная машина (1-2 GPU, 8-16 ядер) хватало для топ-1: Toad Brigade S1 (~20 дней по оценке RLIAYN), Frog Parade S3 (RTX3090+2070S, 8 дней, 300M steps),
     Boey S3 (4090, ~8 дней, 58B steps в JAX), 3Comets (1x4090, 15 дней, ~500 USD), RLIAYN (2080Ti, 4 дня, 300 steps/s).
   - Большие CPU-кластеры нужны там, где среда медленная и неГПУ: GRF (SaltyFish 1000-2500 ядер; TamakEri ~4000 CPU + ~20 T4 в последние 3 дня), Hungry Geese (HandyRL 64 CPU+1 GPU на день для публичной модели; 3rd - vast.ai 12x1080Ti).
   - Скорость симулятора = главный рычаг: Lux S1 JS-движок 300 steps/s (80% времени в env.step); Rust 110k steps/s (Frog Parade), JAX 80k SPS (Boey). Все победители S3 писали собственный быстрый симулятор/JAX-движок.
2. Что дало больше всего:
   - Масштаб обучения (время и размер сети): RLIAYN считает это главным отличием от Toad Brigade; Boey: "should have taken 10M+ model".
   - Teacher/Distillation (KL к замороженной лучшей модели): Toad Brigade, RLIAYN, FLG, Flat Neurons, Frog Parade, EcoBangBang, GRF TamakEri - у ВСЕХ RL-победителей. Это самый воспроизводимый приём.
   - Reward: sparse win/loss в конце + короткий dense shaping на старте (Toad Brigade: 20M шагов; FLG: 65M шагов; Boey: +0.002/очко на раннем этапе). Zero-sum rewards критичны в self-play (WeKick, EcoBangBang, Frog Parade).
     RLIAYN: sparse с нуля = Elo 1200 за дни -> dense. Нет универсального правила, но "swap shaped -> sparse" поздно дал выигрыш (Boey: последние 10% шагов дали "significant gains").
   - BC/IL: (а) дешёвый способ выбрать архитектуру до дорогого RL (RLIAYN, FLG, nosound), (б) стартовая политика (Raw Beast, Hungry Geese 3rd), (в) самостоятельное топ-5 решение (Lux S1 4-6, S3 3rd), но потолок = качество донора.
   - Скриптовые боты: стартовый baseline и sparring partner (Toad Brigade сначала rules; FLG "schedule"), плюс ручные правила на инференсе для разрешения конфликтов (Toad Brigade, Durrett, nosound).
     В rule-heavy играх (Halite, Kore, Lux S2, Battlecode) скрипты победили.
   - Self-play схема: чистый self-play нестабилен (RLIAYN: RSP-циклы), рабочее: latest ~70-85% + прошлые чекпоинты 15-30% (OpenAI Five 80/20; Boey 75/25; Kiwis 75/25; SaltyFish 70% newest;
     AlphaStar main agent 35% SP / 50% PFSP / 15% forgotten+exploiters), PFSP у RLIAYN и Boey; league со стилями у WeKick.
   - Инференс-ограничения диктуют архитектуру: Lux S1 ~2 с/ход, S2 только CPU (FLG не смог поставить onnxruntime), S3 лимит 100MB файла. Закладывать бюджет inference сразу (Colosseum: eval должен мерить latency).
3. Как выбирали финальный сабмит:
   - Kaggle-ladder даёт шум: GRF Raw Beast - расхождение 90 очков у одинаковых агентов; Halite Tom - из 22 сабмитов 16 были бы 1-ми; Kore - ранг "unstable"; Lux S3 - только последние 2 сабмита играют.
   - Практика топов: локальные турниры/пулы - WeKick local ELO LB; Raw Beast internal LB из разнообразных агентов; Hungry Geese 3rd "Coliseum" (2 стадии, TrueSkill); Raine Force (Halite) таблицы head-to-head vs top-5;
     Flat Neurons - A/B внутри сабмита на LB (85/15) с логированием is_strong; Toad Brigade - hall_of_fame.
   - Решение: финал = лучший по локальной матрице против разнообразного пула (+ ladder как sanity), с учётом нетранзитивности (WeKick: "ELO not fully transitive").
4. Операционка для команды 1-3 человека: делайте только одну пару "быстрый симулятор + один центральный контроллер"; тесты против настоящего движка (Frog Parade TDD); автоматический архив чемпионов;
   не публиковать сильнейшего агента раньше времени, если конкуренты используют IL (Flat Neurons).

## F.10 Импликации для Colosseum (кратко)

- Подтверждает выбор APPO/IMPALA + teacher KL: добавить в roadmap `teacher_policy` (frozen best) KL-loss как стандартную опцию (у нас есть kickstart.py - расширить на "teacher=предыдущий чекпоинт" и teacher baseline loss как у Flat Neurons).
- Нужен dynamic reward scaling + target-entropy controller (Flat Neurons) - дёшево, сильный эффект на стабильность [проверить на нашем APPO].
- Пул оппонентов: старт 75/25 (latest/historical), затем PFSP; сделать "league styles" через разные reward-веса (WeKick).
- Eval/Local ladder: встроить "internal LB" с ELO/TrueSkill и матрицей head-to-head (см. G); выбор сабмита - по матрице против разнообразного пула.
- Редкие события (spawn/bid) апсемплить отдельным актором (FLG) - полезно для multi-phase игр.
- В BC-пайплайне поддержать "conditioned imitation" (one-hot на донора, ironbar/Durrett) и data selection по порогу рейтинга (Goosebumps, ironbar).
- IL/BC против "утечки": в Colosseum, если нужно скрывать силу, предусмотреть "mixed submission" (Flat Neurons).

---------------------------------------------------------------------------------------------------

# ЧАСТЬ G. Мониторинг лиги

## G.1 Каталог метрик (что логируют зрелые фреймворки)

### G.1.1 CleanRL (PPO) - минимальный эталонный набор
Из исходников `cleanrl/ppo.py`, `ppo_atari_envpool.py` (https://github.com/vwxyzjn/cleanrl, ключи writer.add_scalar):
- `charts/episodic_return`, `charts/episodic_length`, `charts/avg_episodic_return` (envpool-версия), `charts/learning_rate`, `charts/SPS`
- `losses/value_loss`, `losses/policy_loss`, `losses/entropy`, `losses/old_approx_kl`, `losses/approx_kl`, `losses/clipfrac`, `losses/explained_variance`.

### G.1.2 PufferLib 3.0 (pufferl.py, ветка `3.0`; https://github.com/PufferAI/PufferLib)
- Верхний уровень: `SPS`, `agent_steps`, `uptime`, `epoch`, `learning_rate`.
- `losses/*`: policy_loss, value_loss, entropy, old_approx_kl, approx_kl, clipfrac, importance (ratio mean), explained_variance.
- `performance/*`: elapsed по профилировщику (eval, env, eval_misc, train, train_forward, train_copy, train_misc).
- `environment/*`: любые stats из `info` среды (усредняются np.mean) - сюда естественно класть win-rate/score по играм.
- Логгеры: wandb и neptune (Neptune SaaS теперь недоступен - см. G.5); есть встроенный консольный dashboard (`print_dashboard`); sweeps требуют wandb/neptune, ранняя остановка по NaN-loss (`is_loss_nan`).

### G.1.3 Sample Factory (https://www.samplefactory.dev/05-monitoring/metrics-reference/) - самый полный набор для async/IMPALA-подобной системы
- `train/`: loss, policy_loss, value_loss, exploration_loss, kl_loss, entropy, grad_norm, kl_divergence, kl_divergence_max, fraction_clipped, ratio_min/mean/max,
  adv_min/max/std (и mean в коде), value, value_delta(+max), max_abs_logprob, valids_fraction, same_policy_fraction (доля данных от "своей" политики в мульти-политике), num_sgd_steps,
  adam_max_second_moment, act_min/act_max, returns_running_mean/std, obs_running_mean/std, lr/actual_lr, `train/version_diff_min|avg|max` (**policy lag** в версиях политики).
- `perf/`: `_fps` (learner, после frameskip), `_sample_throughput` (sampling).
- `stats/`: avg_request_count, step_policy, wait_policy, GPU/RAM usage (в т.ч. master_process_memory_mb), gpu_cache_*.
- `reward/`: reward, reward_min, reward_max; `policy_stats/avg_true_objective(+min/max)`; `len/`: len, len_min, len_max.
- Policy lag: в консоли "Policy #0 lag: (min, avg, max)"; рекомендованный диапазон "keep below 20-30 SGD steps"; оценка lag = (num_epochs * num_workers * envs_per_worker * agents_per_env * rollout) / batch_size
  (https://www.samplefactory.dev/07-advanced-topics/policy-lag/). В runner логируются fps окнами и throughput на политику; `max_policy_lag` делает `valids=0` для устаревших сэмплов.
- PBT/мульти-политика: `--with_pbt`, `--num_policies`; PBT ранжирует по "true objective" (разреженная метрика, например победа) - аналог того, что нам нужно в лиге
  (https://www.samplefactory.dev/07-advanced-topics/pbt/). Для самоигры/популяции Sample Factory пишет метрики по каждой политике отдельно (policy_id) [формат ключей - в runner.py `policy_avg_stats`].

### G.1.4 RLlib (https://docs.ray.io/en/latest/rllib/metrics-logger.html)
- `env_runners`: episode_return_mean, episode_length_mean, num_env_steps_sampled; `learners`: policy_loss, vf_loss, entropy, curr_kl_coeff, mean_kl_loss; timing по фазам;
  MetricsLogger: reduce = mean/sum/max/min/ema/lifetime_sum, window, `with_throughput=True`.
- RLlib имеет league-пример с OpenSpiel (self_play_league_based_with_open_spiel.py) [файл не открылся - детали не подтверждены].

### G.1.5 Что реально использовали победители/лаборатории
- Frog Parade (W&B): "various loss terms, average points scored, action frequencies, winrate against the previous best model".
- EcoBangBang: дашборд величин loss-компонентов (UPGO loss на порядок больше остальных -> удалён -> рост 1400 -> 1900+).
- Flat Neurons: win-rate против старых версий и teacher в реальном времени + графики training progress (3 графика в посте).
- Hungry Geese 3rd: win-rate против эталонного агента (~LB1200), TrueSkill в локальной лиге, вывод "LB" в Slack.
- WeKick: local ELO leaderboard по пулу моделей.

### G.1.6 Рекомендуемый список для Colosseum (чек-лист; комбинация источников выше)
Learner (на агента, префикс `agent/<id>/`):
1. Оптимизация: policy_loss, value_loss, entropy (и target entropy, если контроллер), approx_kl (+ max), clip_fraction, ratio min/mean/max, explained_variance,
   grad_norm (до/после clip), lr, adam_max_second_moment, возможно value_delta.
2. V-trace / off-policy: `rho` mean/min/max и доля клипнутых rho и c (clip fraction для importance weights), `log_rho` mean/abs-max, effective sample size (ESS) = (sum rho)^2 / sum rho^2 на chunk,
   версия политики chunk vs learner (policy_lag: min/avg/max; Sample Factory `version_diff_*`), доля chunk, отброшенных по max_policy_lag.
3. Advantages: mean/std/min/max до нормализации; returns mean/std; value mean; доля valid-сэмплов (valids_fraction).
4. Kickstart/teacher: KL(student||teacher), lambda kick, teacher baseline loss; вклад каждого loss-слагаемого в total (дашборд "loss composition" - спас EcoBangBang).
5. Для LSTM: нормы hidden, gradient norm по времени (BPTT), доля эпизодов с обрезанной историей.
6. Для composite actions: entropy и KL на каждую голову отдельно (Flat Neurons - 2 головы с разной целевой энтропией).
Throughput/system:
7. Learner SPS/FPS (samples/s), workers env-steps/s, inference latency, queue depth (trajectory queue per agent, weight queue), chunk age (ms от генерации до использования),
   weight push/pull lag (в версиях и секундах), GPU util/memory, CPU util, число живых workers/learners, drop-count chunks, время в env.step vs forward vs transport (профиль как у PufferLib `performance/*`).
Env/league:
8. Эпизоды: return mean/std (по слотам), length, win/draw/loss на агента, доля self-play vs arena vs historical матчей, частота выбора оппонента (sampling distribution).
9. Rating: Elo/TrueSkill (mu, sigma) каждого агента во времени; payoff matrix + доверительные интервалы; min win-rate vs past (forgetting proxy); Nash/entropy распределение пула.
10. Алармы: NaN loss, entropy collapse (< порога), approx_kl > 0.1 стабильно, clip_fraction > 0.5, explained_variance < 0 долго, policy_lag > 20-30 (Sample Factory), queue depth растёт монотонно (learner не успевает) или равен 0 (workers не успевают), win-rate vs frozen baseline падает.

## G.2 Диагностики лиг: AlphaStar, OpenAI Five, FTW (по первоисточникам)

### AlphaStar (PDF DeepMind: https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf ; Nature 2019, doi 10.1038/s41586-019-1724-z)
- Лига: Main agents (35% SP / 50% PFSP по всем прошлым / 15% PFSP против "forgotten" main и прошлых main exploiters; снапшот каждые 2e9 шагов; никогда не сбрасываются),
  Main exploiters (против main агентов; PFSP f_var; сброс на supervised после добавления), League exploiters (PFSP по всей лиге; добавляются при >70% побед против всей лиги или timeout).
  PFSP weighting: f_hard(x)=(1-x)^p ("hardest opponents"), f_var(x)=x(1-x) ("around own level"). 12 копий actor-learner (по 1 main, 1 main exploiter, 2 league exploiter на 3 расы), центральный coordinator
  "maintains an estimate of the payoff matrix", evaluator-воркеры на CPU "supplement the payoff estimates". Каждый агент: 32 TPU v3, 44 дня; ~900 distinct players за лигу.
- Диагностические графики (Fig.3 + Extended Data):
  - Training Elo всех игроков лиги по времени (относительно Elite built-in bot = 0).
  - Доля Validation Agents, побеждающих main agents в >80 из 160 игр - робастность к невиданным стратегиям (растёт со временем).
  - **Nash distribution** игроков лиги во времени - показывает, что вес у недавних игроков, т.е. нет cycling/forgetting; пример player 40 в Nash 5 дней.
  - Сравнение композиций лиги (+main exploiters, +league exploiters) по test Elo и "Relative Population Performance"; "min win-rate vs past versions" как прокси forgetting; naive self-play - высокий Elo, но forgetful.
  - Состав юнитов Protoss (exploiters быстро меняют состав, main - стабильно).
- Идея для нас: payoff matrix + Nash-веса + min win-rate vs past - три "must have" графика лиги.

### OpenAI Five (https://arxiv.org/abs/1912.06680)
- TrueSkill во времени (0 = random; ~8.3 TrueSkill разности ~ 80% winrate); 80% игр против latest, 20% против прошлых. Opponent sampling: q_i score на прошлого оппонента, p_i ∝ exp(q_i);
  обновление q_i <- q_i - eta/(N p_i) при победе над ним; "Fig.27" - распределение оппонентов в нескольких точках обучения: узкое = агент быстро растёт, широкое = прогресс медленнее
  (т.е. **график opponent sampling distribution** - диагностика темпа прогресса).
- Staleness = M - N (версия оптимизации минус версия, сгенерировавшая sample): замедление при +8 версий; целевой staleness 0-1; "high quality data matters more than compute" (Fig.5).
- Surgery и Rerun: валидация изменений через ререн (~20% ресурсов), batch-size/speedup по TrueSkill thresholds.
- Это прямое подтверждение важности policy lag/staleness как первоклассной метрики.

### FTW / Capture the Flag (https://arxiv.org/abs/1807.01281)
- Популяция 30 агентов; online Elo по результатам тренировочных игр; Elo FTW-популяции по времени vs baselines и людям; эволюция гиперпараметров (lr, KL weight, internal time scale tau) как mean/std по популяции;
  вероятность победы агентов против людей в турнире; PBT мета-оптимизирует внутренние награды -> логируйте веса internal reward во времени.

### PSRO/Open-ended learning (теория для Nash-диагностики)
- Balduzzi et al. "Open-ended learning in symmetric zero-sum games" (arXiv 1901.08106) - Nash clustering/payoff-анализ [URL не перепроверен в этой сессии].
- Для практики: `nashpy` и `open_spiel.python.egt`/`algorithms/nash` для расчёта Nash-смеси по payoff matrix [не проверено в этой сессии].

## G.3 Как выглядит хороший дашборд лиги (проектное предложение)

Панели, сгруппированные по вопросам "что сломалось?":
A. Health of training (по агенту): строки G.1.6 п.1-5; sparklines с алармами; loss composition (stacked area - вклад каждого слагаемого).
B. Throughput/pipeline (кластер): SPS learner vs sample throughput workers, queue depth по агенту, chunk age, weight lag; число живых workers/learners; время в env/forward/transport.
C. Rating-over-time: Elo/TrueSkill всех агентов (линия на агента, tooltip на чекпоинт, закреплённая "ось" - frozen baseline = 0, как Elite bot в AlphaStar и random=0 в OpenAI Five).
D. Payoff matrix heatmap: строки/столбцы = агенты/чекпоинты, цвет = win-rate с поправкой на N игр (Wilson CI в tooltip; серым - мало данных); отдельная heatmap "число матчей".
   Нетранзитивность: строка/столбец отсортированы по Elo; выделять ячейки, где слабый бьёт сильного (Raw Beast/WeKick: ELO не транзитивен).
E. Per-opponent win-rate: для выбранного агента - bar/линии win-rate против каждого оппонента (или кластера) во времени; "min win-rate vs past" (forgetting).
F. Opponent sampling distribution: stacked area, какая доля матчей шла против какого оппонента (self-play/historical/arena) в окне (OpenAI Five Fig.27, PFSP weights).
G. Nash/exploitability: веса Nash-смеси пула во времени (AlphaStar Fig.3C), число "активных" агентов в Nash, лучший ответ exploiter.
H. Agent tree/lineage: граф "кто от кого форкнут/дистиллирован" (teacher chain Toad Brigade: 8->16->24 блока), узел = чекпоинт, цвет = Elo, размер = steps.
I. Evaluator vs frozen references: win-rate vs scripted bot/baseline/best-so-far/teacher (Flat Neurons real-time winrates) - дешёвый "pulse check" каждые K шагов.
J. Behavior metrics: частоты действий (Frog Parade logged action frequencies), доля no-op, "agent gives up" индикатор (Toad Brigade), возможно replays/видео по лучшим/худшим матчам.
K. Ladder panel (если Kaggle): наш LB score по каждому сабмиту, win-rate по enemy_submission_id (Flat Neurons подход), число сыгранных эпизодов.

## G.4 Структура логирования для N агентов

Варианты:
1. **Единый run + префиксы** (`agent/A/losses/policy_loss`, `league/elo/A`, `cluster/queue_depth/A`): проще сравнивать в одном UI и держать одно время; но метрик N*K -> упирание в лимиты cardinality (W&B: metric cardinality 100,000/проект, steps/run 500,000, 1,000 log calls/min, 100k values/min - https://docs.coreweave.com/models/track/limits ).
2. **Run на агента** + отдельный run "league"/"cluster" (группировка `group=experiment_id`, `job_type=learner|coordinator|worker`): чище для динамического add/remove агентов, дешёвый compare между агентами и экспериментами; payoff/Elo лиги живут в coordinator-run.
3. Рекомендация для Colosseum: **гибрид** - coordinator-run (league/*: Elo, payoff, sampling, cluster) + run на каждого trainable агента (group = league_id). Динамическое добавление агента = новый run без пересоздания основного.
4. Матрицы: НЕ логировать N^2 скаляров (`wr/A_vs_B`), а раз в T минут одну таблицу/картинку (`wandb.Table`/Plotly/PNG) + скаляры только для "ключевых" пар (vs baseline, vs best, vs teacher).
5. Шаг по x: единый `global_step` per run + `define_metric(step_metric)` для метрик с другим масштабом (league metrics по `league_step`/wall time).

## G.5 Инструменты трекинга: пригодность для лиги/N-agent

Сводка (подтверждено документацией через Context7/WebFetch, если не помечено):

| Инструмент | Матрицы/таблицы/heatmap | Много агентов | Self-hosted | Ограничения/заметки |
|---|---|---|---|---|
| Weights & Biases | `wandb.Table` (до 200 000 строк на ключ; каждый log создаёт новую версию таблицы), Custom Charts на Vega-Lite (`wandb.plot_table`, wandb/line/v0 и др.; композитные гистограммы), `wandb.plot.confusion_matrix`; для heatmap payoff проще логировать Plotly/matplotlib как image/`wandb.Plotly` или Vega-спеку над таблицей [нативной payoff-heatmap-панели в документации не нашёл] | runs + `group`/`job_type`; run comparison; `define_metric` для нескольких x-осей | облако (docs теперь на docs.coreweave.com); self-managed - платный [не проверено] | Лимиты: 10,000 runs/project, 500,000 steps/run, metric cardinality 100,000/project, 1,000 log rows/min, 100k values/min, один log() < 25MB, значение < 1MB; "performance issues are often caused by too many distinct metrics, not steps" |
| TensorBoard | scalars, images (add_image/add_figure), text, histograms, hparams, Custom Scalars (multi-line, margin charts) - heatmap только как image; нет таблиц payoff как первоклассных [общеизвестное, в этой сессии не перепроверял] | по папкам runs; много runs тормозит | полностью локально | Нет групп/алертов/сравнения в стиле W&B; хорош как fallback, поддерживается CleanRL и SF |
| Aim | metrics с `context`, figures (Plotly/matplotlib `aim.Figure`), images, distributions, text; запросы по runs (`metric.name == ...`) | отдельные runs, контексты (`context={'agent':..}`) | self-hosted, remote tracking server `aim://host:port` | Нет hosted; комьюнити меньше; хорош для "локального W&B" |
| MLflow | metrics (log_batch), `log_figure`, `log_artifact` (JSON/PNG матрицы), `log_table`; nested runs (parent/child) - естественно для лиги: parent = league, child = agent | parent/child runs | self-hosted сервер (основной сценарий) | Нет real-time rich dashboards для таблиц/матриц; метрики одного run - скаляры с step; хорош для реестра чекпоинтов/артефактов |
| ClearML | `Logger.report_scalar/histogram/table/matrix/surface`, `report_matrix` для heatmap-подобных матриц; агенты для оркестрации | tasks/projects | открытый server (docker) + agent | Полезен, если нужна оркестрация задач; вес платформы выше для 1-3 человек |
| Neptune | был удобен для N метрик (PufferLib logger поддерживал), но **SaaS остановлен 2026-03-05, данные удалены; self-hosted Helm-репозитории удалены 2026-03-08** (https://docs.neptune.ai/transition_hub; OpenAI acquisition) | - | недоступен | НЕ использовать; PufferLib `--neptune` опция неактуальна. Рекомендуемые миграции от Neptune: Comet, W&B, MLflow, Lightning AI и др. (по тому же документу) |

Выводы: для 1-3 человек - W&B (удобный UI, группы, таблицы) + локальный fallback (TensorBoard/SQLite/CSV) на случай лимитов; для полностью приватного кластера - Aim или MLflow (+ Grafana, см. G.6).
Особенность проекта: coordinator уже хранит Elo/win-rate (`coordinator/ratings.py`) - писать их в JSON/SQLite как источник истины, а трекер использовать только как "viewer" (устойчиво к смене инструмента, как показал случай Neptune).

## G.6 Простая self-hosted страница для real-time мониторинга кластера

Предлагаемая схема (проектные рекомендации, не подтверждены тестами в этой сессии):
- Источник правды: coordinator публикует `/metrics` (Prometheus text format, библиотека `prometheus_client`): gauges `queue_depth{agent}`, `learner_sps{agent}`, `worker_env_steps_per_s{worker}`,
  `policy_lag_{min,avg,max}{agent}`, `chunk_age_seconds{agent}`, `weight_version{agent}`, `elo{agent}`, `winrate{a,b}` (только топ-N пар), `workers_alive`, `learners_alive`.
  Workers/learners пушат на coordinator по gRPC (уже есть control plane) -> coordinator агрегирует.
- Prometheus (scrape 5-15 с) + Grafana: панели rate/timeseries для throughput/queue; state-timeline для статуса воркеров; heatmap-панель (Grafana "Heatmap"/"Status history") для payoff, если матрицу отдавать как labels `a,b` (малые N);
  alert rules по G.1.6 п.10 (queue depth, policy_lag, NaN, workers_alive).
- Альтернатива попроще: **Streamlit** (как lux-ai-stats; https://www.kaggle.com/c/lux-ai-2021/discussion/281823: Streamlit+Plotly) читает SQLite/Parquet, обновляется `st.fragment`/autorefresh; payoff heatmap через Plotly `imshow` с CI в hover.
  Подходит для 1-3 человек: один процесс, без Prometheus.
- Минимальный вариант для PufferLib-стиля: консольный TUI (как `print_dashboard` в PufferLib) в `colosseum monitor` через `rich`.
- Хранение: SQLite таблицы `matches(ts, agent_ids, slot_agent_map, result, env_seed)`, `ratings(ts, agent, elo, mu, sigma)`, `checkpoints(agent, step, path, parent)`, `chunks(ts, agent, policy_version, age)`; из matches строится любая матрица (в т.ч. по окну).

## G.7 Kaggle ladder tools (для соревновательного режима)

- Kaggle CLI (проверено в `kaggle-api/src/kaggle/api/kaggle_api_extended.py`): `kaggle competitions episodes <submission_id>` (список эпизодов), `kaggle competitions replay <episode_id>` (скачать replay),
  `kaggle competitions logs <episode_id> <agent_index>` (логи агента) - https://github.com/Kaggle/kaggle-api . Метаданные эпизодов: id, createTime, endTime, state, type; агенты: submissionId, index, reward, state, teamName, teamId.
- Episode scraper / downloader: https://www.kaggle.com/robga/simulations-episode-scraper-match-downloader ; Lux S3: https://www.kaggle.com/code/kuto0633/lux-ai-s3-download-episodes-from-meta-kaggle (Meta Kaggle: таблицы Episodes/EpisodeAgents),
  использовался IL-командами и FLG (2000 матчей топ-агентов, HDF5 300GB).
- Стат-трекеры: lux-ai-stats (Streamlit+Plotly+Selenium; https://www.kaggle.com/c/lux-ai-2021/discussion/281823; score growth, win-rate EMA, W/L); ноутбук-прототип https://www.kaggle.com/yalikesifulei/bot-statistics-with-selenium-beautiful-soup;
  Lux AI Episode Helper (Chrome-расширение; https://github.com/paradite/kaggle-lux-episode-helper; в списке эпизодов показывает 5/0 vs 3/2 и открывает визуализатор в один клик);
  Halite: ноутбук "count win or lose for each team" (https://www.kaggle.com/viewlagoon/halite-iv-count-win-or-lose-for-each-team) - матрица win/loss по командам.
- Viewer: `kaggle-environments` (`env.render(mode="ipython"|"html")`; для Lux - Lux Eye https://github.com/Lux-AI-Challenge/LuxEye3 [URL не перепроверен]); Hungry Geese/Halite - встроенный renderer; JMerle visualizer для Kore (упомянут Harm Buisman).
- Практика: сохранять `submission_id -> git commit -> checkpoint`, считать win-rate по `enemy_submission_id` (Flat Neurons) и держать локальную лигу как главный сигнал, ladder - как внешняя проверка.
- Kaggle ladder-механика: каждый новый сабмит играет чаще, рейтинг TrueSkill-подобный со sigma; финальное оценивание у Lux S3 - последние 2 сабмита; шум рейтинга велик (GRF: ~90 очков) - не принимать решения по <~100 эпизодам на сабмит [порог - моё эвристическое правило, не из источников].

## G.8 План внедрения в Colosseum (приоритеты)

1. P0 (дёшево, очень полезно): расширить APPO-логи (rho stats, ESS, policy_lag min/avg/max, clip fraction, approx_kl, explained_variance, grad_norm, advantage stats), loss composition, SPS/queue depth/chunk age.
2. P0: сохранять все матчи в SQLite (agent_ids, slots, outcome, seeds) - основа payoff matrix, Elo, PFSP; экспорт heatmap (Plotly/PNG) раз в T минут в трекер.
3. P1: eval-режим как "league evaluator" - периодическая оценка последнего чекпоинта против frozen set (baseline, best, teacher, scripted) -> win-rate + Wilson CI; графики E/I из G.3.
4. P1: Streamlit-страница `colosseum monitor` (Plotly: Elo-over-time, payoff heatmap, sampling distribution, queue/throughput); позже Prometheus/Grafana при переходе на k8s (уже есть HPA по queue depth).
5. P2: Nash-вес пула (nashpy/OpenSpiel) и "min win-rate vs past" как индикатор forgetting; lineage-граф чекпоинтов (teacher chain).
6. Структура W&B: coordinator-run `league/*` + run на агента, group = league_id (см. G.4); матрицы - только как таблицы/картинки.

---------------------------------------------------------------------------------------------------

# Приложение. Основные URL по темам

F (решения):
- Lux S1: https://www.kaggle.com/c/lux-ai-2021/discussion/294993 ; https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021 ; /300844 ; /296938 ; /293911 ; /293776 ; /294603 ; /294459 ; /296519 ; /283883 ; /281823
- Lux S2: https://www.kaggle.com/competitions/lux-ai-season-2/discussion/407982 ; /406702 ; /409394 ; /411725 ; /405476 ; /404842 ; /408186
- Lux S3: https://www.kaggle.com/competitions/lux-ai-season-3/discussion/569562 ; /568621 ; /568494 ; /568721 ; /570196 ; /571111 ; /567961 ; /569919 ; /554992 ; /567001 ; https://github.com/tonykozlovsky/lux-ai3-pub ; https://github.com/IsaiahPressman/kaggle-lux-2024
- Halite IV: https://www.kaggle.com/c/halite/discussion/183543 ; https://github.com/ttvand/Halite ; /183312 ; /186032
- Hungry Geese: https://www.kaggle.com/c/hungry-geese/discussion/264053 ; /263686 ; /263735 ; /263702 ; /218190 ; https://github.com/DeNA/HandyRL ; https://speakerdeck.com/hoxomaxwell/kaggle-hungry-geese
- GRF: https://www.kaggle.com/c/google-football/discussion/202232 ; /202977 ; /200709 ; /203412 ; https://arxiv.org/abs/2110.04507
- Kore 2022: https://www.kaggle.com/competitions/kore-2022/discussion/340035 ; /337476 ; /320833
- Neural MMO: https://proceedings.mlr.press/v220/liu23a.html ; https://arxiv.org/abs/2508.12524
- MineRL BASALT 2022: https://arxiv.org/abs/2303.13512 ; Pommerman: https://arxiv.org/abs/1902.10870 ; Battlecode postmortems: https://battlecode.org/assets/files/postmortem-2023-4-musketeers.pdf
- CodeCraft (Winter): https://clemenswinter.com/2021/03/24/mastering-real-time-strategy-games-with-deep-reinforcement-learning-mere-mortal-edition/

G (мониторинг):
- CleanRL: https://github.com/vwxyzjn/cleanrl ; PufferLib: https://github.com/PufferAI/PufferLib (pufferl.py, ветка 3.0) ; Sample Factory: https://www.samplefactory.dev/05-monitoring/metrics-reference/ , /07-advanced-topics/policy-lag/ , /07-advanced-topics/pbt/
- RLlib: https://docs.ray.io/en/latest/rllib/metrics-logger.html
- AlphaStar: https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf ; OpenAI Five: https://arxiv.org/abs/1912.06680 ; FTW: https://arxiv.org/abs/1807.01281
- W&B: https://docs.wandb.ai/models/track/log/log-tables ; https://docs.wandb.ai/models/app/features/custom-charts/walkthrough ; https://docs.coreweave.com/models/track/limits
- Aim: https://github.com/aimhubio/aim ; MLflow: https://github.com/mlflow/mlflow ; ClearML: https://github.com/clearml/clearml ; Neptune transition: https://docs.neptune.ai/transition_hub
- Kaggle CLI: https://github.com/Kaggle/kaggle-api ; Episode scraper: https://www.kaggle.com/robga/simulations-episode-scraper-match-downloader ; Lux episode helper: https://github.com/paradite/kaggle-lux-episode-helper

# SP4 «Selection and observability»: практика отбора, турниров, логов матчей и дашбордов

Дата исследования: 2026-10-16. Язык: русский (названия, URL, API в оригинале).

Условные обозначения:
- **[F]** — факт, прочитанный в первоисточнике (URL указан).
- **[S]** — вторичный источник (пересказ, агрегатор, сообщество). Проверяйте перед тем, как опираться.
- **[И]** — мой вывод или собственный расчёт/измерение (не из источника).

## 0. Что удалось и что не удалось прочитать

- Страницы Kaggle (discussion, writeups) рендерятся JS-ом. Обычный WebFetch возвращает только заголовок. Часть страниц читалась через прокси `r.jina.ai/<url>`, но не все. Удалось прочитать:
  - страницы Evaluation Hungry Geese и ConnectX;
  - Rules Lux S3 и Hungry Geese;
  - writeup Lux S3 9th place;
  - обсуждение GRF 5th place.
- Не удалось прочитать: основной пост Hungry Geese 1st place (на странице «This topic has been deleted»), Halite IV 1st place (Tom Van de Wiele), Lux S1 1st place (только README репозитория), Lux S2 1st place (только README репозитория), Kore 2022 (ни одного write-up).
- Поэтому в блоке «как топы выбирают сабмишен» много данных из 2–4 мест и честно помечено, где данных нет. Для Halite, Lux S1/S2 и Kore **нельзя утверждать**, сколько игр и против кого играли победители.
- Поисковик давал слабую выдачу. Контекст Kaggle-сообщества лучше всего покрывает агрегирующий writeup zoli800 [S], который я использую осторожно. Его числа про Wilson я пересчитал сам, они совпали (см. §1.4).

---

## 1. Как топ-участники выбирают, что сабмитить

### 1.1. Что известно по конкретным соревнованиям

| Соревнование / источник | Что делали | Числа | Тип |
|---|---|---|---|
| **Hungry Geese, 3rd place** (Maxwell, Speaker Deck: https://speakerdeck.com/hoxomaxwell/kaggle-hungry-geese) | Двухступенчатый фильтр. Ступень 1: кандидат играет 224 матча против «стандартных агентов» команды, отсекается при winrate ≤ 0.55 (224 CPU на GCP). Ступень 2: прошедшие играют в локальном «Coliseum» (96 CPU), сабмитят тех, у кого хороший TrueSkill. Обучающие эпизоды скрапили у агентов с публичным LB > 1200. | 224 матча; порог 0.55; итоговый public LB 1239.1 | [F] |
| **Google Research Football, 5th place** (DeNA и Manchester City; HandyRL; https://www.kaggle.com/c/google-football/discussion/203412) | Часть воркеров постоянно оценивает чекпойнты (кривая на каждом сохранении). Главная метрика — winrate против сильного встроенного AI (full 3000-step игры). Кандидаты на сабмишен: 1000–2000 игр каждый. Авторы пишут, что рейтинг LB «сильно коррелировал» с этим winrate. Пул оппонентов: idle/random/run-right, 5 публичных rule-based ядер Kaggle, hard built-in AI, прошлые обученные агенты, смеси. Веса оппонентов подстраивались динамически так, чтобы winrate обучаемого был около 50%. Доля self-play с latest росла с 10% до ~80%. Проблема: нельзя засабмитить следующую модель, пока оценивается предыдущая, поэтому финальный сабмишен — не самая свежая модель. | 1000–2000 игр на кандидата; ~700k игр за обучение; в последние 3 дня ~4000 CPU и 20 GPU | [F] |
| **Lux AI S3, 9th place** (team-k; https://www.kaggle.com/competitions/lux-ai-season-3/writeups/team-k-9th-place-solution) | Модифицировали `luxai_runner`, чтобы он показывал winrate и p-value. Правило: принимать изменение, только если winrate значимо выше лучшей модели. Streamlit-визуализатор для отладки скрытого состояния. | ~100 эпизодов на сравнение (пример из write-up) | [F] |
| **Lux AI S1, 1st place** (Toad Brigade, IsaiahPressman; https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021) | Формальной процедуры отбора в README нет. Прогресс отслеживали по сабмитам на Kaggle и просмотру отдельных эпизодов; сабмиты архивировали в `internal_testing/hall_of_fame`. Тренировка: замороженный предыдущий агент как «учитель» (KL). Сами авторы пишут, что более разнообразная лига могла бы дать устойчивость, но они её не делали. Вывод: топ-1 использовал ладдер Kaggle как оценщик. | — | [F] |
| **Halite IV, 1st place** (ttvand) | По пересказу поисковика: 22 сабмита, 16 из них заняли бы 1-е место; репо помечен «provisional first place». Оригинал не прочитан. | 22 / 16 | [S] |
| **Halite IV, 4th place** (0Zeta; https://github.com/0Zeta/HaliteIV-Bot) | Эволюционная настройка параметров (>100 параметров) против собственных ботов. Проблемы self-play: симметрия, переобучение, низкий шум, из-за чего один бот может «снежным комом» выиграть случайно. Оптимизацию бросили к концу. Баг, внесённый за день до дедлайна, испортил финальных ботов. Вывод авторов: надо было сделать replay viewer со статистикой. Топ-3 на LB с Elo 1505 на последней неделе. | — | [F] |
| **Orbit Wars** (не победитель; архив агента jacklvd; https://github.com/jacklvd/orbit_war_2026) | «Local win-rate is a regression gate, **not** proof of a leaderboard gain.» «Real validation is an A/B submission.» Финальную пару выбирали по стабильности, а не по пику. Лучший стабильный ~1239 Elo, пик 1352 (разница ~110 пунктов). Новые сабмиты стартуют с ~600 и сходятся за ~сутки. Winrate «сильно зависит от пачки сидов», одна хорошая пачка «раз за разом оказывалась шумом». Локальные оппоненты упираются в потолок ~1000–1100. | см. таблицу | [F] (самоотчёт не-победителя) |
| **Hungry Geese, 1st place** (HandyRL, DeNA и QUANTUM) | Основной пост удалён. По вторичному агрегатору: ~16M игр чистого self-play, затем ~3M игр против разнообразной популяции, потому что чистый self-play выучил негласную кооперацию, которую чужие оппоненты не соблюдали; вылечили разными оппонентами и поиском. Победитель — HandyRL (DeNA/QUANTUM), 875 команд. | 16M / 3M | [S] (zoli800) |
| **Hungry Geese, 2nd place** | Политика из отфильтрованных реплеев с высоким рейтингом + PUCT; фильтр по порогу LB-score. | — | [S] |
| **Kore 2022** | Для топ-3 по агрегатору упор на форвард-симуляцию и учёт угроз. Деталей оценки нет. | — | [S] |
| **Lux S2 1st** (ry_andy_; https://github.com/ryandy/Lux-S2-public) | README описывает только запуск одного локального матча (`luxai-s2 ./main.py ./main.py -s 0 -l 1000`, до часа). Про отбор ничего. | 646 команд | [F] |

**Что из этого видно ([И]):**
1. Ни в одном прочитанном источнике нет «чистого» протокола с фиксированным якорным набором, парными сидами и CI. Ближе всего: Hungry Geese 3rd (порог на 224 матчах, затем лига), Lux S3 9th (100 эпизодов + p-value), GRF 5th (1000–2000 игр на кандидата + динамический пул).
2. Топы с большим вычислением (GRF 5th: 4000 CPU) оценивают кандидатов **тысячами** игр, топы без кластера — сотнями, и полагаются на ладдер.
3. Локальный пул практически всегда включает: публичные/скриптовые боты (якоря), встроенный AI, прошлые собственные версии. Это совпадает с моделью `anchors` + `snapshots` в SP3.
4. Разрыв «локально ↔ LB» упомянут явно только в Orbit Wars. GRF 5th, наоборот, сообщает высокую корреляцию с LB — но только когда локальная метрика была winrate против того же типа соперников, что и на ладдере.

### 1.2. Правила Kaggle по финальному выбору
- Lux S3: «You may select up to three (3) Final Submissions for judging.» Если не выбрать, Kaggle выберет автоматически. Лимит 5 сабмитов в день. https://www.kaggle.com/competitions/lux-ai-season-3/rules [F]
- Lux S1: лимит 5 в день; на LB показывается только лучший по рейтингу бот. «Final evaluation window» — игры продолжаются с 7 по 20 декабря. https://www.for9a.com/en/opportunity/lux-ai-competition [S, агрегатор]
- Hungry Geese: 5 сабмитов в день. https://www.kaggle.com/competitions/hungry-geese/rules [F]

### 1.3. Что делали с шумом ладдера (по данным)
- GRF 5th: шум явно не обсуждался, только 1000–2000 игр на кандидата. [F]
- Orbit Wars (самоотчёт): финальную пару выбирали по стабильности; одна хорошая пачка сидов — шум. [F]
- Агрегатор zoli800 [S]: https://www.kaggle.com/writeups/zoli800/winning-kaggles-reinforcement-learning-competitio. Рекомендации: dev-арена (частые шумные проверки), promotion-арена (номинированные кандидаты), quarantine-арена (один раз для финалистов); фиксировать правило остановки до просмотра результатов; хешировать артефакт перед оценкой; не использовать ладдер как цикл настройки гиперпараметров; считать последовательные изменения рейтинга зависимыми («Do not infer too much from a short-term rating movement»); при двух разрешённых сабмитах предпочитать агентов с **дополняющими строками payoff-матрицы**, а не два соседних чекпойнта.

### 1.4. Сколько игр нужно (расчёт [И], совпадает с zoli800 [S])

Wilson 95%, p≈0.5:

| Полуширина CI | Игр |
|---|---|
| ±10 п.п. | ~97 |
| ±5 п.п. | ~385 |
| ±3 п.п. | ~1070 |
| ±2 п.п. | ~2400 |

Сколько игр нужно, чтобы нижняя граница Wilson превысила 50% при наблюдаемом winrate:

| Наблюдаемый winrate | Игр |
|---|---|
| 65% | ~41 |
| 60% | ~91 |
| 55% | ~370 |
| 52% | ~2380 |

Соответствие Elo: +10 → 51.4%, +25 → 53.6%, +50 → 57.1%, +100 → 64%, +200 → 76% [И, формула Elo]. Значит, «отличить улучшение на 25 Elo» требует примерно 1000+ независимых игр. Шкала OpenAI Five: разница TrueSkill ≈ 8.3 соответствует ~80% winrate [F, https://arxiv.org/pdf/1912.06680].

### 1.5. Проклятие победителя при выборе максимума ([И], моя симуляция, 500 прогонов)
Все K кандидатов **одинаковы** (истинный winrate 0.5), n игр на каждого, выбираем максимум наблюдаемого:

| K | n | Среднее наблюдаемое значение выбранного |
|---|---|---|
| 10 | 100 | 0.576 |
| 20 | 100 | 0.593 |
| 50 | 200 | 0.580 |
| 20 | 400 | 0.547 |
| 20 | 1000 | 0.529 |

Если у кандидатов реальный разброс (sd 0.03), K=20: при n=100 наблюдаемый максимум 0.606, а у выбранного истинное значение 0.529 (лучший истинный 0.556). При n=1000: наблюдаемый 0.564, выбранный истинный 0.550. Вывод: выбор максимума по шумной оценке завышает результат на 3–9 п.п. при n≈100; **повторная оценка выбранного на свежих сидах** (отдельная арена) обязательна. Это же объясняет разрыв 1352 пик → 1239 стабильно в Orbit Wars [F + И].

---

## 2. Рейтинговая система Kaggle simulation competitions и её уроки

**Первоисточник (страница Evaluation, Hungry Geese):** https://www.kaggle.com/competitions/hungry-geese/overview/evaluation [F]
- Рейтинг каждого сабмита моделируется гауссианой: μ — оценка навыка, σ — неопределённость.
- При загрузке сначала играется validation episode: сабмит играет сам с собой; если упал — статус «Error». Прошедшие стартуют с μ = 600.
- «We usually try to have each Submission participate in an episode a day but the frequency may vary.» Матчи подбираются из пула, пары с близкими рейтингами.
- Победа увеличивает μ победителя и уменьшает μ проигравшего; при ничьей μ сближаются; σ уменьшается пропорционально информации.
- «The Leaderboard will show the rating of each team's best Submission.»
- ConnectX: «all valid Submissions will continually play more matches and have dynamically changing scores as the pool increases»; пары «with similar ratings»; рейтинг — гауссиана. https://www.kaggle.com/competitions/connectx/overview/evaluation [F]
- Lux S1 (агрегатор): новые боты играют заметно чаще старых. [S]
- Страница Evaluation для Lux S1/S2/S3 и Halite не доступна через прокси (показывает правила игры); предполагаю ту же схему, но **не проверено**.

**Подтверждения из практики:**
- Orbit Wars: рейтинг — «continuously-updated Elo», агенты продолжают играть, рейтинг дрейфует, сходимость ~сутки [F, самоотчёт].
- Рейтинг не равен winrate: «the displayed rating is an estimate of skill, not a raw field-wide win rate» (по цитате zoli800 со страницы Connect Four) [S].
- Сжатие структуры: рейтинг скрывает циклы «камень-ножницы-бумага». В AlphaStar payoff-матрица Protoss: ~3 000 000 RPS-циклов с участием хотя бы одного exploiter и ~200 только среди main-агентов. https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf, Extended Data Fig. 8 [F].

**Уроки ([И], основаны на фактах выше):**
1. Ладдер — это шумный, медленный (≈1 игра/день/бот), зависимый от очереди оценщик с отбором максимума по сабмитам. Локальная арена нужна, чтобы уйти от этих трёх проблем.
2. Для «ладдер-подобного» режима Colosseum стоит иметь: validation-эпизод «сам с собой» (аналог `validate`), стартовый μ/σ как у нового участника, матчмейкинг по близости рейтингов — и **отдельный режим «калибровки»**, который играет много игр сразу, а не по одной в сутки.
3. «Показывать лучший сабмит» усиливает проклятие победителя; в нашем лидерборде надо показывать CI и число игр, а не только точку.

### 2.1. Урок CodinGame о TrueSkill на ботах [F]
- Штат (CodinGame65, 2014): первые 100 матчей играются последовательно против оппонентов в окне rank−10…rank+5; потом матчи только против тех, кто пересабмитил. Каждый матч с игроком из топ-15 переигрывается 5 раз с новыми сидами, и только объединённый результат идёт в TrueSkill. В конце топ-100 получает ещё 200 матчей. https://forum.codingame.com/t/multiplayer-contests-ranking-system/381
- CodeBusters: после закрытия — ещё 200 матчей для Legend-лиги (топ-150), итоговый балл = **среднее TrueSkill по этим 200 матчам**, а не последнее значение; у топ-15 в сумме ≥ 1000 матчей; «с 200 матчами стартовые позиции не важны». https://forum.codingame.com/t/codebusters-final-ranking/1730
- Smash The Code: TrueSkill «gives more importance to the last games», так как создан для людей, которые растут, а не для ботов. https://forum.codingame.com/t/smash-the-code-question-about-the-final-ranking/1515
- Platinum Rift 1: 300 дополнительных матчей не стабилизировали топ из-за случайности игры [S, пересказ поиска].
- Вывод [И]: для фиксированных ботов нужна модель без дрейфа (`tau`→0 в TrueSkill; psyleague тоже советует снижать `tau` только при 10 000+ игр на бота) или пакетная оценка (BT/Plackett-Luce MLE) по всему логу.

---

## 3. Форматы турниров для ботов

### 3.1. Сводка

| Формат | Где встречается | Плюсы | Минусы | Для Colosseum |
|---|---|---|---|---|
| **Round-robin (все пары)** | AlphaStar payoff-матрица; Chatbot Arena (win matrix); cg-brutaltester (матрица winrate) | Полная матрица, ловит нетранзитивность | O(N²): 36 чекпойнтов = 630 пар, ~322 560 игр при 512 играх на пару (по zoli800 [S]) | Для финалистов и малых N |
| **Gauntlet (новый бот против фиксированного набора)** | OpenAI Five: 83 референсных агента, μ референсов — константа; HG 3rd: 224 матча против «стандартных агентов» | Линейно по числу новых ботов; рейтинг на фиксированной шкале | Зависит от набора якорей | **Основной режим для снапшотов** |
| **Рейтинговый ладдер / матчмейкинг по близости** | Kaggle, CodinGame, psyleague, OpenAI Five (играют только против референсов ближе 10 TrueSkill) | Эффективен по информации | Дрейф и зависимые обновления | Для «живой» таблицы |
| **Swiss** | В прочитанных источниках для ботов не встречался | — | — | Не нужен на первом этапе [И] |
| **Single/double elimination по seed из ладдера** | Battlecode (по вторичным данным: скримиджи задают посев) [S, https://liuxue.xdf.cn/shijiazhuang/mglh_bklh_sqzn/2958353.shtml] | Зрелищность | Высокая дисперсия, плохо для выбора лучшего | Пропустить |
| **King-of-the-hill** | Не найден как практика для ботов | — | — | Пропустить [И] |

### 3.2. Инструменты (что делают, что стоит перенять)

- **psyleague** (https://github.com/FakePsyho/psyleague) [F]:
  - TrueSkill по умолчанию; `psyleague run` крутит матчи, пока не остановишь; `psyleague bot add NAME -s SRC`, `psyleague show`;
  - матчмейкинг двумя правилами: если у какого-то бота < `mm_min_matches` игр и случайное число < `mm_min_matches_preference`, играется недоигравший против случайного; иначе — два случайных;
  - N игроков: `n_players`; скрипт `play_game` возвращает `ranks`, `errors`, `player_data`;
  - данные в `psyleague.db` и `psyleague.games` (человекочитаемый формат), формат не обратно совместим;
  - метаданные ботов через stderr: `[TDATA] key = value` (уровень теста), `[PDATA] key = value` (уровень игрока), `[PDATA+]` (сумма за игру);
  - рекомендации: `n_workers=1` + `--verbose` для отладки; не плодить воркеров, иначе таймауты выглядят как падения ботов; `tau` не понижать без 10 000+ игр на бота; пинить версию.
- **cg-brutaltester** (https://github.com/dreignier/cg-brutaltester) [F]:
  - `-s` — swap: повтор каждой игры на той же карте со сдвинутыми позициями; при >2 игроках — простая ротация, а не все перестановки (пример: 4 матча на карту вместо 24 для 4 игроков); реферee должен принимать сид;
  - `-i` задаёт начальный сид (отдельные сиды выбрать нельзя), `-t` потоки (оставлять свободное ядро), `-n` число игр, `-l` — логи stdout/stderr рефери и игроков в каталог;
  - итог: таблица winrate каждого игрока против каждого.
- **CG Arena** (форум CodinGame; https://forum.codingame.com/t/cg-arena/207039; https://lib.rs/crates/cgarena) [F]: вдохновлён psyleague и cgstats; веб-UI, рейтинги OpenSkill/TrueSkill/Elo на выбор, лидерборд в реальном времени, кастомные лидерборды с фильтром по критериям матча (например малые карты), графики параметров по ходам (например деньги на ход), «fully local», но веб-сервер можно открыть, чтобы смотреть лидерборд с телефона.
- **luxai-s3 `--tournament`** (https://raw.githubusercontent.com/Lux-AI-Challenge/Lux-Design-S3/main/src/luxai_runner/README.md) [F]: список путей к ботам или каталог с подкаталогами `main.py`; Elo с K=32, исходы W/T/L; матчмейкинг случайный; `--tournament_cfg.concurrent=N`; живой лидерборд (Player, Rating, Episodes); реплеи `replays/replay_<episode_id>.json`, смотрятся в s3vis.lux-ai.org. Документация про сиды в турнире отсутствует. Для S2: `luxai-s2 a/main.py b/main.py -o replay.json|html -s <seed>` и сторонняя статистика luxaistats.crabdance.com [F/S].
- **kaggle-environments** (https://github.com/Kaggle/kaggle-environments) [F]: `evaluate` прогоняет N эпизодов и возвращает награды; о ранжировании и расписании в README ничего нет. Kaggle CLI имеет команды для симуляций: `kaggle competitions episodes <SUBMISSION_ID>`, `replay <EPISODE_ID> -p`, `logs <EPISODE_ID> <AGENT_INDEX> -p` [S, зеркало документации https://docsearch.algolia.com/mcp/docs/repo/kaggle/kaggle-cli].
- **OpenSpiel** (https://raw.githubusercontent.com/google-deepmind/open_spiel/master/docs/algorithms.md) [F]: α-Rank (оценка/визуализация), PSRO, exploitability / best response; встроенного тура для ботов не нашёл.
- **Melting Pot** (https://github.com/google-deepmind/meltingpot; Tech Report https://arxiv.org/abs/2211.13746) [F]: >50 substrate и >256 test-scenarios; сценарий = субстрат + «background population»; основная метрика — focal-population per-capita return (протокол оценки сгенерализации на незнакомых партнёров). Для Colosseum — прообраз «сценария арены»: набор фиксированных фоновых популяций и фокальный игрок.
- **RLlib** (пример `rllib/examples/self_play_league_based_with_open_spiel.py`) [S]: лига из main, main exploiters, league exploiters; колбэк после каждого обновления считает winrate политики по наградам батча и решает, снапшотить ли и замораживать; win_rate_threshold — порог, с которого вводятся эксплойтеры. Отдельной матрицы или рейтингов нет.
- **Unity ML-Agents** (https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-Configuration-File.md) [F]: self-play с параметрами `save_steps` (20000), `team_change` (5×save_steps), `swap_steps` (10000), `window` (10), `play_against_latest_model_ratio` (0.5); Elo доступен как мера для curriculum «только для self-play». Совпадает по смыслу с нашим `keep_last`/`latest` в SP3.
- **Sample Factory PBT** (https://www.samplefactory.dev/07-advanced-topics/pbt/) [F]: популяция политик, ранжирование по `true_objective` из `info`, нижние K% клонируются с верхних с мутацией гиперпараметров, `policy_NN_cfg.json` в каталоге эксперимента. Матчей между политиками с рейтингами нет.
- **Шахматные движки** (практика сравнения версий):
  - парные партии с зеркальными цветами; trinomial-модель неверна для парных партий (корреляция внутри пары), нужна pentanomial (5 исходов) — fishtest сообщает о «substantial saving of testing resources» [F, https://github.com/official-stockfish/fishtest/wiki/Fishtest-mathematics, https://www.chessprogramming.org/Match_Statistics];
  - SPRT: после каждой игры проверять H0 (Elo=elo0) против H1 (Elo=elo1) и останавливаться при пересечении границ; fishtest использует GSPRT;
  - LOS = ½[1 + erf((W−L)/√(2W+2L))] как «вероятность превосходства» [F].
  - Для нас: SPRT/пары — то, что даёт честный «стоп-критерий» для `tournament` (сколько игр играть до уверенного решения) [И].
- **Chatbot Arena** (https://arxiv.org/abs/2403.04132) [F]: перешли с онлайн-Elo на **Bradley-Terry MLE**, потому что «BT coefficients are better for the purpose of statistical estimation»; доверительные интервалы через bootstrap и sandwich-оценку; ранг модели = 1 + число моделей с нижней границей CI выше её верхней границы; **неравномерная выборка пар**: больше голосов парам с близкими результатами, adaptive sampling даёт выигрыш относительно случайной выборки.

### 3.3. Дублированные матчи, сиды, N игроков

- **Парные/зеркальные матчи** [F + И]: шахматы (пары с обменом цветов), cg-brutaltester `-s`, zoli800: «seed s с кандидатом в сиде 0 и затем в сиде 1; анализировать разность внутри блока», для многопользовательских игр «Treat the whole rotation as one resampling cluster». Связь: наши `schedule_lineups` уже вращают команды; надо добавить **один и тот же сид для всех вращений** и считать блок кластером при bootstrap.
- **Фиксированные наборы сидов** [И]: общий seed-list для всех кандидатов (общие случайные числа) убирает дисперсию карт при сравнении, но создаёт переобучение под набор: нужна «закрытая» часть, играемая один раз (quarantine). Orbit Wars: детерминированность между процессами есть, внутри одного процесса — нет [F], значит каждая ячейка (seed, config) должна идти в свежем процессе или хотя бы детерминизм проверяется тестом.
- **Расписание FFA** [И]:
  - для 4-местной FFA и N агентов число составов растёт как C(N,4)·перестановки мест; cg-brutaltester показывает приём «циклическая ротация мест» вместо всех 24 перестановок;
  - практичная схема: для каждого состава из k агентов делать k ротаций мест на каждом сиде, ротацию считать одним блоком; состав выбирать по «uniform over pairs» (каждая пара встречается примерно одинаково часто), а не по uniform over agents, иначе слабые пары недопредставлены. В Colosseum для этого пригодится уже имеющийся `rank`-отчёт с парной матрицей.
  - Рейтинг для рангов: Plackett-Luce (OpenSkill) принимает `ranks` или `scores`, ничьи — равные ранги.

### 3.4. Рейтинг-системы: что брать

| Система | Свойства | Источник |
|---|---|---|
| **OpenSkill** (`pip install openskill`, MIT) | Модели PlackettLuce (рекомендована по умолчанию), BradleyTerryFull/Part, ThurstoneMostellerFull/Part; произвольное число команд и игроков; `ranks` или `scores`; `margin`, `weights`; `predict_win`, `predict_draw`, `predict_rank`; `ordinal()` = μ−3σ; `limit_sigma`. README заявляет «150% faster than TrueSkill» и «Accuracy matches TrueSkill» без чисел. | https://openskill.me/en/stable/manual.html, https://github.com/philihp/openskill.py [F] |
| **TrueSkill** (`trueskill`) | Пакет BSD, но «Microsoft permits only Xbox Live games or non-commercial projects to use TrueSkill™». Дефолты: μ=25, σ=8.333, β=4.167, τ=0.0833, draw 0.1. OpenAI Five: σ=25/3, β=σ/2, τ=0.0, draw 0.02 (τ=0 — без дрейфа). | https://trueskill.org/, https://arxiv.org/pdf/1912.06680 [F] |
| **Elo (онлайн)** | Дёшево, но зависит от порядка и K; в Chatbot Arena заменён на BT. luxai-s3 использует K=32. | [F] |
| **BT / Plackett-Luce MLE по логу** | Порядконезависим, CI через bootstrap. | Chatbot Arena [F] |
| **Эволюционные: α-Rank, Nash averaging, RPP** | Для нетранзитивных популяций; AlphaStar RPP = Nash(Pab)ᵀ P Nash(Pab). | AlphaStar, OpenSpiel [F] |

**Рейтинг-практика OpenAI Five** (https://arxiv.org/pdf/1912.06680, Appendix J) [F]:
- Пул из **83 референсных агентов**, шкала от 0 (random) до 254 (версия, победившая чемпионов мира); скриптовый бот ~105 TrueSkill.
- Оценка во время обучения: отдельные машины скачивают свежие веса и играют против референсов; **не играют с теми, кто дальше 10 TrueSkill** (winrate < 15% или > 85%); рейтинг референсов считается константой; после **750 игр** версия логируется и берётся следующая; новая версия инициализируется μ предыдущей. Обновление примерно раз в два часа.
- Проблема: версия игры меняется, старые агенты пришлось обновлять в коде, иначе преимущество у того, кто тренировался на текущей версии.
- 80% игр против latest, 20% против старых; сэмплирование прошлых оппонентов динамическое (Appendix N).

**AlphaStar** [F]: внутренний Elo лиги с добавленными встроенными ботами, шкала привязана «Elite bot = 0»; Relative Population Performance; «validation agents»; payoff-матрица и Nash лиги.

---

## 4. Match log

### 4.1. Что хранить на каждый матч ([И], с опорой на практику ниже)

| Поле | Зачем | Практика-источник |
|---|---|---|
| `match_id`, `ts_start`, `ts_end`, `duration_s`, `env_steps` | Идентификация и скорость | Kaggle episodes API [S] |
| `layout` / game id / `env kwargs hash` | Фильтры по раскладке, регрессии при смене окружения | OpenAI Five: смена версии игры ломала сравнимость [F] |
| `seed` (+ флаг детерминизма) | Воспроизведение; реплей по сиду, если среда детерминирована | cg-brutaltester `-i`; Orbit Wars: недетерминизм внутри процесса [F] |
| по местам: `seat`, `team`, `role`, `agent_id`, `agent_kind` (trainable/frozen/scripted), `version` (policy_version), `weights_hash`, `config_hash`/`arch` | Однозначная идентификация участника; рейтинг по снапшотам | zoli800: хешировать артефакт [S] |
| по местам: `score`, `rank`, `terminated/truncated`, `error` (PlayerError/timeout/illegal) | Рейтинг, диагностика ошибок | psyleague: `errors` отдельным списком [F] |
| `source` / `purpose` (training / eval / tournament / calibration), `lineup_origin` (категория оппонента: latest/snapshot/rival/anchor) | Различить обучающие и оценочные игры; PFSP-статистика | SP3 уже пишет played shares |
| `worker_id`, `host` | Отладка | psyleague `n_workers`, перегрузка ≈ таймауты [F] |
| `replay_uri` / `replay_policy` | Указатель на реплей | Lux `replay_<episode_id>.json` [F] |
| `custom` (словарь пользовательских метрик игры) | Статистика по ходу/ресурсам | psyleague `[TDATA]/[PDATA]` [F], CG Arena графики по ходам [F] |

### 4.2. Форматы и размеры (мои измерения [И], Python 3.x stdlib; 1 000 000 матчей 1v1, 500 уникальных версий агентов)

| Формат | Размер | Комментарий |
|---|---|---|
| JSONL (по строке на матч, компактный JSON, ~330 байт) | **331 MB** | Дописывается, читается любым инструментом |
| JSONL + gzip | **66 MB** (5×) | Для архива/ротации |
| SQLite, 2 таблицы (`match`, `seat`) + индекс по agent | **190 MB**, вставка 5.7 с (executemany) | |
| то же + индекс `seat(match_id)` | **214 MB** | Без индекса по `match_id` соединение seat⋈seat не завершилось за 2 минуты (ловушка) |
| Запрос «агент A против всех оппонентов» (с индексом) | **8 мс** | 499 оппонентов |
| Полная матрица пар (seat0 vs seat1) по 1M матчей | **3.0 с** | 245 089 пар |
| Parquet | не измерял (нет pyarrow в окружении) | Оценка по аналогии: 10–30 MB на 1M коротких записей, [И] без проверки |

Выводы [И]: порядок 0.3 KB/матч в JSONL и 0.2 KB/матч в SQLite; 1M матчей — сотни MB, это «ничто» по сравнению с реплеями. Реплеи (десятки KB–MB на игру: Lux JSON/HTML, Kaggle episode JSON) нельзя хранить для всех; в источниках политика хранения не документирована, поэтому предлагаю свою (см. §6).

### 4.3. Что делают на практике
- **psyleague**: `psyleague.db` + `psyleague.games` — два файла, человекочитаемо [F].
- **Kaggle**: у каждого эпизода есть replay и логи агентов, скачиваются через CLI; сообщество скрапит эпизоды для имитации (Lux S3 9th: 7 935 эпизодов Stage 1 только победные, 1 550 Stage 2; HF-датасет Orbit Wars: `episodes/<date>.tar.gz` с сырыми JSON) [F/S].
- **Запросы, которые нужны**: рейтинги (BT/PL) по логу, матрица winrate (по ролям и раскладкам), история «агент → рейтинг по версии» и разбор регрессий («какие матчи проиграла версия N+1, которые выигрывала N»).

---

## 5. Дашборды и мониторинг лиг

### 5.1. Что показывают известные системы

| Система | Что видно | Источник |
|---|---|---|
| **OpenAI Five** | Кривая TrueSkill обучения против пула референсных агентов (Figure 3), метки «уровень, где бот начинает побеждать людей» на кривой; сравнение запусков по числу версий до порога TrueSkill (speedup) | https://arxiv.org/pdf/1912.06680 [F] |
| **AlphaStar** | Training Elo агентов лиги за 44 дня; доля validation-агентов, побеждённых main; распределение Нэша лиги; payoff-матрица (Extended Data Fig. 8); сравнение составов лиги по Elo и «min winrate vs past» (прокси забывания); RPP | AlphaStar PDF [F] |
| **RLlib league example** | Только winrate политики из батча (колбэк); никакой матрицы | [S] |
| **ML-Agents** | Self-play параметры; Elo как мера для curriculum | [F] |
| **Sample Factory** | PBT-конфиги по политикам в файлах; дашборд — W&B/TensorBoard (мой общий опыт, не проверял) | [F] |
| **PufferLib** | Локальный дашборд и W&B/Neptune (по вторичному описанию) | [S] |
| **CG Arena** | Веб-UI: управление ботами, лидерборд в реальном времени, фильтры, графики параметров по ходам; открывается с телефона | [F] |
| **psyleague** | CLI `show`; без UI | [F] |
| **Lux** | Загрузка replay в s3vis.lux-ai.org; lux-eye (репо); luxaistats (сторонний) | [F] |
| **W&B Tables** | Таблицы с типизированными столбцами, фильтр, сравнение моделей на одном наборе (https://docs.coreweave.com/guides/tables/) | [F] |

CleanRL (TensorBoard/W&B), Streamlit/Gradio/Dash/Panel и Grafana: общеизвестные инструменты, источники для них в этом исследовании я не читал; в таблице «стек» ниже они — мои оценки [И].

### 5.2. Какие виды важны (по фактам выше + [И])
1. **Рейтинг во времени по каждому агенту/снапшоту** (Elo/TrueSkill vs шаги обучения) — OpenAI Five Fig. 3, AlphaStar training Elo.
2. **Heatmap матрицы winrate** (строка бьёт столбец) с числом игр и CI, отсортированная по рейтингу — AlphaStar payoff matrix; CG Arena таблица; cg-brutaltester итог.
3. **Результаты против якорей** (скриптовые/замороженные) как доля — OpenAI Five reference pool, GRF winrate vs built-in AI.
4. **Распределение PFSP / доля сыгранных категорий оппонентов** (у нас уже пишется в `episodes`).
5. **Браузер матчей** с фильтрами (агент, раскладка, исход, ошибка) и ссылкой на реплей — CG Arena фильтры; Lux s3vis.
6. **Метрики здоровья**: число ошибок/таймаутов/нелегальных действий, распределение длительности, пропускная способность — psyleague предупреждает, что перегрузка воркеров выглядит как отказы ботов.
7. **Мера забывания**: min winrate против прошлых версий (AlphaStar) и `wr_vs_past` (уже есть).

### 5.3. Рекомендуемый стек ([И], для pip-пакета с минимумом зависимостей)

| Уровень | Что | Зависимости |
|---|---|---|
| 0 | `colosseum leaderboard runs/X` — текстовая таблица: агент/версия, рейтинг ± CI, игры, winrate против якорей | stdlib |
| 1 | `colosseum report runs/X -o report.html` — **один статический HTML-файл** с встроенными JSON-данными (таблицы, heatmap, линии рейтинга, PFSP) и небольшим JS; открывается офлайн, пересылается на телефон | stdlib (JS внутри); опциональный график через CDN-библиотеку (Plotly/uPlot) с локальным запасным рисованием на canvas |
| 2 | `colosseum dashboard runs/X --port N` — тот же HTML + `/api/*` поверх SQLite/JSONL (`http.server`/`wsgiref`), автообновление; CG Arena-стиль «открой с телефона» | stdlib |
| 3 | Экспорт в W&B: таблица лидерборда, heatmap как `wandb.Table`/Image, кривые рейтинга | extra `wandb` (уже есть) |
| 4 | Распределённый хаб: тот же уровень 2 читает общую БД хаба | то же |

Почему так: файл-отчёт работает без процесса и без сети (приоритет владельца «читаемо с телефона»), тот же код рендера переиспользуется живым сервером и хабом. Streamlit/Dash дают быстрее интерактив, но тянут большие зависимости и процесс; их можно сделать опциональным extra `colosseum[dashboard]`, но не основой [И]. Grafana нужна только при уже существующей Prometheus-инфраструктуре; для SP4 избыточна [И].

---

## 6. Рекомендация для SP4 (со стороны практики)

### 6.1. Минимальный набор, который даёт «выбрать лучшего и видеть всё»

**Первым делом (ядро):**
1. **Match log** как единственный источник правды. Одна запись на матч (поля §4.1), хранилище SQLite (`runs/<name>/matches.db`) + JSONL как append-журнал/экспорт. Без индекса по `match_id` соединения медленные (§4.2), делать индексы сразу. Рейтинги и матрицы **вычисляются из лога**, а не накапливаются только онлайн.
2. **Пакетный рейтинг** (Bradley-Terry / Plackett-Luce MLE по логу, bootstrap CI по блокам «сид + ротации мест») как основной, **OpenSkill** как онлайн-индикатор для живой таблицы (MIT, ranks/scores, команды). TrueSkill не брать из-за ограничения лицензии Microsoft.
3. **`colosseum tournament`**: режимы:
   - `gauntlet` (кандидаты против фиксированного набора якорей — режим по умолчанию для снапшотов);
   - `round-robin` (финалисты);
   - `calibrate` (набор игр до заданной ширины CI или SPRT-стоп).
   Парные сиды и ротация мест в блоке, общий список сидов для всех кандидатов, отдельный «закрытый» список сидов для финального прогона.
4. **Отчёт по выбору**: таблица «версия → рейтинг ± CI → игры → winrate против каждого якоря и против предыдущей лучшей → худшая семья/якорь», пометка «выбор подтверждён на свежих сидах: да/нет».
5. **Статический HTML-отчёт** (§5.3 уровень 1): рейтинг-во-времени, heatmap, якоря, PFSP, браузер матчей (список с фильтрами; реплеи по ссылке).

**Потом:** top-k хранение снапшотов по рейтингу (на основе пункта 2, а не по одиночной оценке), экспорт в W&B, живой сервер (уровень 2), интеграция с хабом SP5.

**Пропустить на первом этапе:** Swiss/king-of-the-hill/плей-офф сетки (нет подтверждённой пользы для выбора лучшего), α-Rank/PSRO (нужны только для нетранзитивной популяции с большим числом эксплойтеров; оставить как возможное расширение), Streamlit/Grafana как обязательные зависимости, онлайн-ладдер «одна игра в сутки».

### 6.2. Политика реплеев ([И])
Хранить реплей для: всех игр финального турнира (сжато), матчей с ошибкой/нелегальным действием/таймаутом, проигрышей версии против якоря, который прошлая версия выигрывала, и ~1% случайной выборки. В записи — `seed` + `lineup` + `weights_hash`, чтобы реплей можно было восстановить повторным запуском (при детерминированной среде).

### 6.3. 5 главных практических уроков
1. **Всегда оценивать на фиксированном наборе якорей с парными/ротированными сидами.** OpenAI Five: фиксированный пул из 83 агентов с шкалой от random=0; HG 3rd: 224 матча против стандартных агентов; Orbit Wars: «local win-rate is a regression gate, not proof». Блок «сид + все ротации» считать одним кластером.
2. **Число игр: сотни на отсев, тысячи на решение.** Для 55% нужно ~370 игр, для 52% ~2400 (§1.4); GRF 5th сравнивал кандидатов на 1000–2000 играх, Lux S3 9th — ~100 с p-value. Лучше использовать последовательный критерий (SPRT/доверительная последовательность), чем фиксированное N, и объявлять правило остановки заранее.
3. **Не доверять максимуму шумных оценок.** При K=20 кандидатов и 100 играх выбор по максимуму завышает winrate на ~9 п.п. (0.593 против 0.5); выбранного перепроверять на свежих сидах (promotion → quarantine арена). В Orbit Wars разрыв пик-стабильное ≈110 Elo.
4. **Рейтинг ≠ winrate, и онлайн-рейтинг дрейфует.** Kaggle/CodinGame рейтинг зависит от очереди и порядка; CodinGame в итоге усредняет TrueSkill по 200 матчам и переигрывает матчи топ-15 по 5 раз с новыми сидами; Chatbot Arena перешёл на BT MLE. Хранить сырые матчи и пересчитывать; показывать CI и число игр, а не только точку.
5. **Нетранзитивность реальна: смотреть payoff-матрицу, а не только рейтинг, и держать разнообразных якорей.** AlphaStar: ~3 000 000 циклов с экплойтерами против ~200 между main-агентами; Hungry Geese 1st: чистый self-play выучил негласную кооперацию, не работавшую против чужих; для финала выбирать агентов с дополняющими строками матрицы, а не соседние чекпойнты (zoli800 [S]).

Дополнительно (меньше уверенности): локальный пул имеет потолок силы (Orbit Wars), поэтому периодически сверять локальные выводы с внешним сигналом (ладдер, внешние боты); время оценки сабмита на ладдере ограничивает частоту A/B (GRF 5th не смог засабмитить свежую модель).

---

## 7. Источники (сводно)

- Kaggle Evaluation: https://www.kaggle.com/competitions/hungry-geese/overview/evaluation ; https://www.kaggle.com/competitions/connectx/overview/evaluation
- Kaggle Rules: https://www.kaggle.com/competitions/lux-ai-season-3/rules ; https://www.kaggle.com/competitions/hungry-geese/rules ; https://www.kaggle.com/competitions/lux-ai-2021/rules
- Lux S1 (агрегатор): https://www.for9a.com/en/opportunity/lux-ai-competition
- Hungry Geese 3rd: https://speakerdeck.com/hoxomaxwell/kaggle-hungry-geese
- GRF 5th (DeNA/HandyRL): https://www.kaggle.com/c/google-football/discussion/203412 ; HandyRL: https://github.com/DeNA/HandyRL
- Lux S3 9th: https://www.kaggle.com/competitions/lux-ai-season-3/writeups/team-k-9th-place-solution
- Lux S1 1st: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021 ; Lux S2 1st: https://github.com/ryandy/Lux-S2-public
- Halite 4th: https://github.com/0Zeta/HaliteIV-Bot ; Halite 1st: https://github.com/ttvand/Halite
- Orbit Wars: https://github.com/jacklvd/orbit_war_2026
- Агрегатор: https://www.kaggle.com/writeups/zoli800/winning-kaggles-reinforcement-learning-competitio
- CodinGame: https://forum.codingame.com/t/multiplayer-contests-ranking-system/381 ; https://forum.codingame.com/t/codebusters-final-ranking/1730 ; https://forum.codingame.com/t/smash-the-code-question-about-the-final-ranking/1515
- Инструменты: https://github.com/FakePsyho/psyleague ; https://github.com/dreignier/cg-brutaltester ; https://forum.codingame.com/t/cg-arena/207039 ; https://lib.rs/crates/cgarena ; https://github.com/Lux-AI-Challenge/Lux-Design-S3 ; https://github.com/Lux-AI-Challenge/Lux-Design-S2 ; https://github.com/Kaggle/kaggle-environments
- Рейтинги: https://openskill.me/en/stable/manual.html ; https://github.com/philihp/openskill.py ; https://arxiv.org/abs/2401.05451 ; https://trueskill.org/ ; https://arxiv.org/abs/2403.04132 ; https://arxiv.org/abs/2008.06787
- Лиги и оценка: https://arxiv.org/pdf/1912.06680 ; https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf ; https://arxiv.org/abs/2211.13746 ; https://github.com/google-deepmind/meltingpot ; https://github.com/google-deepmind/open_spiel/blob/master/docs/algorithms.md ; https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Training-Configuration-File.md ; https://www.samplefactory.dev/07-advanced-topics/pbt/
- Шахматные тесты: https://www.chessprogramming.org/Match_Statistics ; https://github.com/official-stockfish/fishtest/wiki/Fishtest-mathematics
- Дашборды: https://docs.coreweave.com/guides/tables/ (W&B Tables)

Собственные расчёты (скрипты): `research/sp4_sims/stats.py` (Wilson, проклятие победителя), `research/sp4_sims/size_bench.py` и `research/sp4_sims/size_q.py` (размеры и время запросов match log).

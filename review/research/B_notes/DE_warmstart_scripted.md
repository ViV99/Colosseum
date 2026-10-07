# Заметки D + E: скриптовые боты в популяции; warm start и переходы между фазами

Дата: 2026-10-07. Контекст: Colosseum (APPO = PPO + V-trace, IMPALA-style; BC -> self-play -> PFSP/league).

## Легенда статусов

- Без пометки: проверено по первоисточнику (текст статьи/код/supplement прочитан).
- `[вторично]`: подтверждено только вторичным источником (обзор, поиск, блог).
- `[вывод]`: мой вывод/экстраполяция из подтверждённых фактов, в источнике этого нет.
- `[не подтверждено]`: не удалось найти/проверить; не использовать как факт.
- `[эвристика]`: разумное значение по умолчанию без прямого обоснования в литературе; требует sweep.

Важное замечание про числа: коэффициенты потерь (KL, entropy) нельзя переносить между кодовыми базами напрямую: они зависят от нормировки advantage, суммирования по action-heads (sum vs mean), масштаба value loss и т.д. Все "коэффициенты из статей" ниже -- точки отсчёта для sweep, а не drop-in значения.

Критичное расхождение с CLAUDE.md: в проекте online BC записан как `λ·KL(student||teacher)`. Во всех трёх первоисточниках (Kickstarting, AlphaStar, VPT) используется направление teacher -> student, то есть `KL(teacher||student)` = cross-entropy(teacher, student) минус entropy teacher (forward KL, mode-covering). См. раздел E.2.3.

---

# ТЕМА D. Скриптовые боты в популяции

## D.1 Зачем нужны скриптовые боты (роли)

| Роль | Суть | Подтверждение |
|---|---|---|
| Anchor / ruler (фиксированная точка отсчёта) | Self-play Elo/TrueSkill относителен и дрейфует; нужны неизменяемые референсы, чтобы видеть абсолютный прогресс | OpenAI Five: оценка по пулу fixed reference agents с известным TrueSkill (Sec. 4); CTF (FTW): Elo-турнир включал Quake III scripted bots разных уровней и людей; TStarBot-X/AlphaStar Unplugged: built-in Elite/very_hard bot как бенчмарк |
| Curriculum / bootstrapping | Против слабого предсказуемого бота дообучение стартует с ненулевым winrate | Bansal 2018: dense exploration reward + sampling старых оппонентов; Lux S1: см. D.2 |
| Защита от forgetting | Фиксированные стратегии в пуле не "забываются" и штрафуют регресс | AlphaStar `_verification_branch` (проверка forgetting по historical checkpoints); OpenAI Five Appendix N |
| Защита от collusion / overfitting к self-play | При self-play агент и копия могут сойтись на взаимно-согласованном, но хрупком конвенциональном поведении (в cooperative-подобных играх N-player: "молчаливый сговор") | Lanctot 2017 (joint-policy correlation, InRL overfits to co-players); Gleave 2020; Wang 2023. Прямых работ именно про collusion в N-player FFA с self-play в моей выборке нет `[не подтверждено]` |
| Измерение exploitability | Скрипт-бот с "странной" но валидной стратегией выявляет дыры; но настоящий lower bound даёт обученный exploiter (D.5) | Gleave 2020, Wang 2023, Tseng 2024 |
| Источник экспертных меток (teacher) | Скрипт как teacher для kickstarting/RGPS | Toad Brigade (rules-agent написан, потом RL победил), TStarBot-X RGPS (D.2, E.2) |
| Guide-policy для roll-in | Скрипт ведёт первые h шагов эпизода, потом RL | JSRL (E.3) |

Важный нюанс: скриптовые боты в роли sparring partner НЕ заменяют league. Они закрывают узкий, заранее известный набор стратегий; non-transitive циклы и новые эксплойты лежат вне этого набора (Czarnecki 2020 "Spinning Tops": у реальных игр сильная non-transitive компонента, нужна population/diversity; Tseng 2024 вывод про "diversity in training").

## D.2 Практика: что делали победители/лаборатории

### OpenAI Five (arXiv 1912.06680)
- 80% игр: latest policy vs latest; 20% игр: против past versions (Appendix N). Причина (цитата-смысл): "avoid strategy collapse in which the agent forgets how to play against a wide variety of opponents because it only requires a narrow set of strategies to defeat its immediate past version".
- Dynamic sampling прошлых оппонентов: каждому прошлому i квалити-скор `q_i`; `p_i ∝ exp(q_i)`; каждые 10 итераций текущий агент добавляется в пул с `q = max(existing q)`; после игры: если past-оппонент победил - нет апдейта, если текущий победил: `q_i <- q_i - η/(N·p_i)`, `η = 0.01`. Это формула (14) из Appendix N. Эффект: быстро улучшающийся агент => старые оппоненты получают низкий q и почти не выбираются.
- Скриптов в пуле оппонентов нет; скриптованная логика была внутри политики (item purchase, courier, ability builds - Appendix F.1) и постепенно передавалась модели (см. E.6, annealing).
- Была "hand-scripted" baseline в Fig.16-ablation: агент без partial rewards "learned to play well enough to beat a hand-coded scripted agent consistently" - то есть скрипт использовался как ruler, не как тренировочный партнёр.

### AlphaStar (Nature 2019; pseudocode из supplement, прочитан файл `multiagent.py`)
Точная логика league (из `pseudocode/multiagent.py`, это "pseudocode", не production-код):
- `pfsp(win_rates, weighting)`: `variance: x(1-x)`, `linear: 1-x`, `linear_capped: min(0.5, 1-x)`, `squared: (1-x)^2`; если сумма < 1e-10 - uniform. `win_rates` - винрейт ИГРОКА против оппонента.
- Payoff: win-rate с экспоненциальным забыванием `decay = 0.99` на каждый результат пары; для нулевых игр 0.5.
- MainPlayer.get_match: `coin_toss < 0.5` -> PFSP по historical (`squared`) - 50%; иначе выбирается случайный main-агент; при `coin_toss < 0.65` (15%) - "verification branch" (проверка exploiters: если min winrate против historical exploiters < 0.3 - играть против них (squared); проверка forgetting: после `remove_monotonic_suffix`, если min winrate против historical checkpoint-ов оппонента < 0.7 - играть против них); иначе (35%) - self-play branch: если `payoff[self, opponent] > 0.3` - играть против opponent, иначе (оппонент слишком силён) - брать его historical checkpoints по `variance`-PFSP (curriculum).
  => итоговые доли для main: 50% PFSP vs league, 35% self-play (с автоматическим curriculum), 15% verification (exploiters + forgetting). Эти проценты выведены из кода, в тексте статьи Methods я их не читал.
- Checkpoint main-агента: не раньше 2e9 шагов; если min winrate против всех historical > 0.7 или шагов > 4e9.
- MainExploiter: оппонент - случайный main; если `payoff > 0.1` - играть с ним, иначе - его historical по `variance`. При checkpoint - сброс весов на `initial_weights` (т.е. на supervised-агента). Условие checkpoint: min winrate vs main > 0.7 или > 4e9 шагов (min 2e9).
- LeagueExploiter: PFSP `linear_capped` по всем historical; при checkpoint с вероятностью 0.25 reset на initial weights.
- Состав: 3 main (по расе), 3 main exploiters, 6 league exploiters; 32 TPUv3 на агента, 44 дня, ~900 distinct players.
- Нет скриптовых ботов в league. Built-in bots (Elite/ very hard) использовались как внешняя оценка (Fig. "Supervised win-rate vs Elite bot"). Источник "human exploration" - supervised-политика и replays (E.1-E.2).

### TStarBot-X (arXiv 2011.13729; воспроизведение AlphaStar в малом масштабе)
- Baseline supervised agent: 90% vs built-in Elite-bot после 43 часов imitation learning (раздел 4.4).
- Rule-Guided Policy Search (RGPS): `L_RGPS = KL(π_expert(·|s) || π_θ(·|s))` при `s ∈ S_critical` (малое подмножество состояний, выбранное людьми), иначе 0; `π_expert` задаёт человек (one-hot / multi-hot, "if-else"-логика, можно label smoothing). Включён только для main agent. Это ровно мост "скриптовые правила -> KL-регуляризация в RL" (см. рецепт R3).
- DAPO: `L_DAPO = KL(π_{θ_{t-1}} || π_{θ_t})`, t - индекс learning period (~12 часов), teacher - предыдущая сохранённая версия себя; активировали в последние 15 из 57 дней, после чего "significant policy improvement speedup". Обоснование: фиксированная KL к слабому supervised-агенту сдерживает main agent.
- Exploiters сбрасываются на supervised; добавлен "AEE" (continue/inherit) чтобы exploiter не терял прогресс.

### Lux AI Season 1, Toad Brigade (1st; README в GitHub IsaiahPressman/Kaggle_Lux_AI_2021)
- Первоначально параллельно писали rules-based агента и RL; "within the first month the RL approach began to beat the rules-based one", rules-based агент заброшен.
- Алгоритм: FAIR IMPALA + UPGO + TD(lambda); "frozen teacher model perform inference on all states, and added a KL loss term for the current model's policy from that of the teacher. This helped to stabilize behavior and prevent strategic cycles - both of which are problems that plague a pure self-play setup" (прямая цитата-смысл).
- Reward shaping только первые 20M шагов (за города/юниты/research/fuel + win/loss), потом sparse win/loss.
- Цепочка сетей: 8-block с shaped reward -> 16-block и затем 24-block на sparse reward, "with the smaller previous networks as teachers each time". Это готовый паттерн "model growth через distillation + KL" (рецепт R8b).
- Имитации/BC не использовали (начинали со случайной инициализации).
- Коэффициент KL и расписание в README не указаны `[не подтверждено]`.

### Другие Kaggle/соревнования
- Google Research Football 2020 (WeKick, 1st): imitation learning + multi-head value trick + distributed league training [TiZero/TiKick, вторично]. Пул оппонентов инициализирован стратегиями из RL и GAIL `[вторично, из поисковой выдачи; первоисточник-writeup не читал]`.
- TiZero (arXiv 2302.07515): curriculum self-play с adaptive difficulty -> "Challenge & Generalise Self-play" (dynamic opponent pool, PFSP-подобный). Без human data/BC. Детали сложности - в статье, не извлекал.
- Hungry Geese (DeNA/HandyRL 1st): детали (имитация top-агентов?) `[не подтверждено]`.
- Lux AI S2 (NeurIPS 2023): организаторы дали rule-based, RL и IL baselines и ">1 млрд кадров play data" с предыдущей итерации (small-scale env) - страница competition. Какие подходы выиграли - `[не подтверждено]`.
- Lux AI S3: детальные writeup найти не удалось `[не подтверждено]`.
- Capture the Flag (FTW, arXiv 1807.01281): популяция агентов, обучаемых параллельно (PBT), matchmaking по Elo (число агентов не проверял); tournament оценка против scripted Quake III bots и людей; "internal reward" эволюционирует PBT. Скрипты - только для оценки.
- Hide & Seek (arXiv 1909.07528): чистый self-play как "natural curriculum"; скриптовых ботов не используют.

## D.3 В какой пропорции

Прямого исследования "какая доля скриптовых оппонентов оптимальна" я не нашёл `[не подтверждено]`. Подтверждены только ориентиры для past/historical оппонентов и выведенные из них эвристики:

| Источник | Доля | Что именно |
|---|---|---|
| OpenAI Five | 80% latest / 20% past (softmax-q sampling) | Appendix N, Table 2 ("Past opponents 20%") |
| AlphaStar main (код) | 50% PFSP(league), 35% self-play (+auto-curriculum), 15% verification | выведено из `multiagent.py` |
| Bansal 2018 (1710.03748) | оппонент ~ Uniform(δ·v, v), δ=1: только последний, δ=0: вся история. На Humanoid Sumo δ=0.5 лучший, на Ant δ=0 | раздел 5.4; "random old versions" стабильнее, чем latest |
| Toad Brigade | 100% self-play + постоянная KL к frozen teacher | README |

`[эвристика]` для Colosseum (финальная арена, scripted ботов подмешивать как постоянные anchor-слоты, не обучаемые, `collect_mask=False`):
- 10-20% матчей включают >=1 слот scripted/frozen anchor; доля НЕ затухает (anchor должен оставаться, иначе теряется функция guard-а от забывания).
- Внутри anchor-слотов: PFSP `squared` по винрейту (как в AlphaStar), с нижней границей (floor) вероятности для каждого скрипта (например 2-5%), чтобы "пройденный" бот не исчезал полностью (иначе forgetting не ловится).
- Ранняя фаза (после BC): доля scripted оппонентов может быть выше (30-50%), затем линейно к 10-20% за первые X% бюджета self-play.

## D.4 Curriculum оппонентов

Подтверждённые механизмы:
1. Self-play как автоматический curriculum (Bansal 2018; Hide&Seek: "agents always play opponents of an appropriate level").
2. Автоматический откат на слабых при слишком сильном оппоненте: AlphaStar `_selfplay_branch`: если `winrate(self, opponent) <= 0.3`, брать historical checkpoints оппонента по `variance` PFSP (`x(1-x)`: фокус на ~50%-матчах).
3. Dense "exploration reward" -> sparse, линейный annealing коэффициента `α` (Bansal: за 500 итераций, 1000 для kick-and-defend).
4. Постепенное включение новой механики: OpenAI Five - 0% -> 100% игр с новой средой/действием, откат при просадке TrueSkill (Appendix B).
5. Размер сети/сложность: Toad Brigade chain 8->16->24 блоков; TiZero adaptive curriculum сложности сценариев.
6. JSRL: curriculum по "сколько шагов ведёт guide-policy": h уменьшается (E.3).

Рекомендуемый порядок для Colosseum `[вывод]`: scripted(слабые) -> смесь scripted + замороженные BC/ранние checkpoints -> self-play (latest + пул) -> PFSP league (+ exploiters + anchor-доля). Переход между стадиями по gate-метрике (winrate vs anchor > порога), а не по числу шагов (по аналогии с AlphaStar `ready_to_checkpoint`: min winrate > 0.7 либо лимит шагов).

## D.5 Exploiters и измерение exploitability

Определение: exploitability политики π (в 2-player zero-sum) = насколько best response выигрывает у π сверх value игры; на практике - lower bound через обученный adversary (Gleave: "constructively lower-bounding the exploitability of a victim - its performance against its worst-case opponent - by training an adversary").

### Gleave et al. 2020 "Adversarial Policies" (arXiv 1905.10615, ICLR 2020)
- Adversary обучается model-free RL против замороженной black-box victim. Adversary выигрывает, потратив <3% timesteps, которые ушли на обучение victim; Kick-and-Defend/You-Shall-Not-Pass/Sumo; победа за счёт "естественных, но off-distribution observations": adversary лежит/дёргается; masking позиции оппонента убирает эксплойт (victim blind to adversary выигрывает 99% vs 86% проигрыша).
- Defense - fine-tune victim против adversary: "single" (только adversary) вызывает catastrophic forgetting против нормального оппонента (opponent winrate растёт); "dual" (рандомно adversary или Zoo оппонент на эпизод) лучше, но всё равно хуже оригинала (57% vs 48% у нормального оппонента). Новый adversary находится снова ("attack method can be successfully reapplied"). Предлагают PBT с постоянным добавлением новых агентов.
- Вывод для framework: dual-training (mixed pool) обязателен при патче эксплойта; один патч не закрывает класс уязвимостей.

### Wang, Gleave et al. 2023 "Adversarial Policies Beat Superhuman Go AIs" (arXiv 2211.00241, ICML 2023)
- Adversary выигрывает >99% против KataGo без search и >97% при "superhuman" search, затратив <14% compute обучения KataGo (из abstract). "Cyclic" эксплойт: приманка в виде циклической группы, KataGo не видит захват. Transfer zero-shot на другие Go AI; люди могут повторить.
- Обучение: A-MCTS-S (adversarial MCTS, victim заморожен), с curriculum: начало с cp39 без search, затем рост числа visits у victim (до 131,072; Fig. 5.1).
- Hard-coded defense (pass-alive) закрыл один "pass" эксплойт, но cyclic adversary выигрывает 95.7% (1052 игры) против Latest_def при 4096 visits, и 72% против Latest при 1e7 visits.

### Tseng et al. 2024 "Can Go AIs Be Adversarially Robust?" (arXiv 2406.12843)
- Три защиты: positional adversarial training, iterated adversarial training (9 итераций), ViT вместо CNN. Все провалились против "свежих" adversaries: positional - adversary дообучается за 8% compute защиты и возвращает 0% -> 92% winrate; iterated - вариант атаки 90% winrate за 26% compute защиты (другой - 81% за 5%, но неэффективен при большем search).
- Гипотеза авторов: self-play "only one opponent identical in strength and reasoning" ограничивает exploration и фиксирует local equilibria. Предлагают PSRO/DeepNash-подобные схемы и "diversity in training".
- Следствие: для exploitability нужны постоянные exploiters, а не разовая проверка; и защита должна обобщать, а не патчить конкретную атаку.

### AlphaStar exploiters (см. D.2)
- Main exploiter: играет только против текущих main; reset на supervised при checkpoint; цель - найти дыры main-агента; main затем их закрывает (verification branch проверяет min winrate против historical exploiters < 0.3).
- League exploiter: PFSP по всей лиге, не targeted main exploiters; ищет "systemic weaknesses of the entire league"; reset w.p. 0.25.
- Результат: производительность main exploiters со временем падает ("main agents grew increasingly robust").
- Minimax Exploiter (arXiv 2311.17190): game-theoretic reward для exploiter; "converges faster than standard exploiters"; детали формулы не извлекал `[не подтверждено]`.

### Практический протокол измерения (для `colosseum eval`) `[вывод]`
1. Заморозить snapshot (checkpoint) агента A; обучить exploiter с нуля (или с BC-инициализации) против A с ограниченным бюджетом (ориентир из литературы: 3% [Gleave], <14% [Wang] от бюджета обучения A).
2. Метрика: winrate exploiter vs A с Wilson CI (в `eval.py` уже есть) + winrate A vs обычные anchor-боты (чтобы убедиться, что A не стал хуже в среднем).
3. Exploiter-чекпоинт добавить в пул (frozen, `collect_mask=False`) -> A обязан учиться против него (AlphaStar: main exploiters feed back).
4. Повторять периодически (исчерпывающего закрытия не будет - Tseng).
5. Для N-player: exploiter занимает 1 слот, остальные слоты - A (или mixture); меряется pairwise результат.

## D.6 Catastrophic forgetting в self-play: причины и лекарства

Причины:
- Агент оптимизируется против текущего оппонента и теряет навыки против ранее побеждённых стратегий: "strategy collapse" (OpenAI Five Appx N); cyclic non-transitivity (A>B>C>A; AlphaStar: self-play "may chase cycles indefinitely"; Balduzzi 2019, Czarnecki 2020).
- Independent RL overfits co-players: Lanctot 2017 (arXiv 1711.00832) - joint-policy correlation (JPC) показывает падение результата при смене партнёра; PSRO как обобщение.
- Bansal 2018: тренировка против latest оппонента => дисбаланс: один агент сильно вырывается, другой "unable to recover"; random old versions стабильнее.

Лекарства (подтверждены):
| Метод | Механизм | Источник |
|---|---|---|
| Past checkpoints (FIFO/uniform) | Простейший anti-forgetting | Bansal 2018 (δ), OpenAI Five (20%) |
| FSP | best response против uniform mixture всех прошлых; сходится к Nash в 2p zero-sum | Heinrich & Silver 2016 (arXiv 1603.01121), AlphaStar FSP |
| PFSP | веса `f(winrate)`: `(1-x)^p` (hard), `x(1-x)` (var) - больше игр с трудными; в AlphaStar Extended Data Fig. 5: PFSP лучше FSP по population performance и exploitability | AlphaStar Nature |
| Verification branch | отдельная доля игр целенаправленно проверяет forgetting (min winrate vs historical < 0.7) и exploiters (< 0.3) | `multiagent.py` |
| Exploiters (main/league) | находят слабости, закрепляют их в пуле | AlphaStar, TStarBot-X |
| KL к frozen teacher | не даёт policy уйти далеко: Toad Brigade (против cycles), AlphaStar (supervised KL), DAPO (к предыдущей версии) | см. выше |
| Dual/mixed fine-tuning | при патчинге против adversary смешивать с обычным оппонентом | Gleave 2020 |
| Нормализованное сравнение Elo с anchor | видеть регресс абсолютно | CTF/OpenAI Five |

## D.7 OpenAI Five: surgery и "rerolling" (Rerun)

Определения: "surgery" - набор offline-операций над параметрами старой модели для получения новой, совместимой со сменившейся средой/архитектурой, которая играет на том же уровне; "Rerun" - повторное обучение с нуля на финальной среде/архитектуре для валидации.
- Масштаб: >20 surgeries за 10 месяцев (≈ одна в 1-2 недели), много неудачных попыток; Rerun занял 2 месяца, 150±5 PFlops/s·days, ~20% ресурсов OpenAI Five; достиг >98% winrate против финального OpenAI Five; наивный пересчёт: без surgery проект занял бы ~40 месяцев вместо 10.
- Rerun hyperparams (Fig. 7 / Appendix C): менялись только 4 вещи: learning rate, entropy coef, team spirit, GAE horizon; каждое изменение применялось плавно 1-2 дня: team spirit 0.3 -> 0.8 (план: -> 1.0), horizon 180 -> 360 сек (план -> 840), entropy 1e-2 -> 1e-3, LR 5e-5 -> 5e-6 (план -> 1e-6).
- Детали операций - в разделе E.6.
- Вывод для framework: "exact function preservation" нужна ещё и потому, что все замороженные past opponents надо было преобразовать так же ("if the surgery fails to preserve policy, these frozen past agents will forever play worse, reducing quality of the opponent pool").

---

# ТЕМА E. Варианты warm start и переходы между фазами

## E.1 Offline BC на больших датасетах реплеев

### AlphaStar supervised (Nature; `supervised.py`, `detailed-architecture.txt`)
- Данные: публичные анонимизированные human replays Blizzard; MMR cutoff = 3500 (топ ~22%); число реплеев 971,000 `[вторично: обзор arXiv 2111.07631; в тексте Methods Nature не читал]`.
- Из `supervised.py`: BATCH_SIZE 512, TRAJECTORY_LENGTH 64, Adam lr 1e-3 (beta 0.9/0.999, eps 1e-8), loss = cross-entropy(MLE по человеческим действиям) + 1e-5·L2; 128-core TPUv3 slice; LSTM state переносится между соседними траекториями; `FINE_TUNING=True`: MMR cutoff 6200, lr 1e-5 (дообучение supervised на топ-играх; именно эта fine-tuned supervised политика служит teacher для KL в RL).
- Conditioning на стратегию z: build order включается с p=0.8, build units с p=0.5 (masking); MMR игрока подаётся на вход при supervised ("Elsewhere fixed at 6200"), т.е. политика условная и при RL ставят высокий MMR.
- Результат: AlphaStar Supervised: средний рейтинг 3,699 MMR (выше 84% людей); Final 6,048-6,275 MMR.
- AlphaStar Unplugged (arXiv 2308.03526): публично ~20M игр; фильтр версий 4.8.2-4.9.2 -> ~5M; MMR>3500 -> ~1.4M игр = 2.8M эпизодов (>30 лет игры); BC cosine LR `λ0 = 5e-4`; FT-BC: `λ0 = 1e-5` на отфильтрованном top-tier. Выводы: Return-Conditioned BC, Q-learning подходы, off-policy evaluation "sometimes fail to win a single game against weakest opponent" и не превзошли unconditional BC; работают one-step offline RL: сначала оценить behavior policy и behavior value function, потом улучшить политику (при обучении или inference). 90% winrate против прежнего AlphaStar BC. Значит: фильтровать данные по качеству игрока важно (Table 2 в статье).

### Датасеты соревнований Kaggle
- Meta Kaggle (kaggle.com/datasets/kaggle/meta-kaggle): публичные таблицы активности Kaggle, обновляются ежедневно; наличие таблиц Episodes/EpisodeAgents `[вторично]`, схему страницы проверить не удалось.
- Kaggle CLI для simulation: `kaggle competitions episodes <SUBMISSION_ID>`, `kaggle competitions replay <EPISODE_ID> -p <PATH>`, `kaggle competitions logs <EPISODE_ID> <AGENT_INDEX>` `[вторично]` (в docs/competitions.md в main этих команд не нашёл - вероятно, в более новой версии CLI; проверить `kaggle competitions --help`).
- Примеры публичных дампов: Orbit Wars episodes на HuggingFace (35k+ эпизодов, фильтр "strong play: оба агента >= 1500 ladder rating") `[вторично]`.
- "Episode scraper" (публичные ноутбуки через внутренние endpoints ListEpisodes/GetEpisodeReplay) `[не подтверждено]` - конкретные URL не нашёл.
- Lux AI S2: организаторы предоставили >1 млрд кадров play-данных (small-scale) для IL.
- Практика фильтрации `[вывод из AlphaStar/Unplugged]`: брать только эпизоды топ-N агентов по leaderboard (аналог MMR>3500), хранить рейтинг игрока как условный вход (MMR-conditioning), на финале - fine-tune на самых сильных с lr в ~10-100 раз ниже.

### BC pitfalls
- Compounding error / covariate shift (Ross, Gordon, Bagnell 2011, arXiv 1011.0686, DAgger): BC с ошибкой ε на распределении эксперта даёт `J(π) ≤ J(π*) + T²ε` (Theorem 2.1 из Ross & Bagnell 2010); DAgger при N = Õ(T) итераций даёт оценку, линейную по T (Theorems 3.1-3.2).
- DAgger: `π_i = β_i π* + (1-β_i) π̂_i`; обучать на агрегате всех собранных данных (состояния посещены текущим учеником, метки - эксперта); `β_1 = 1`; простая версия `β_i = I(i=1)` "often performs best in practice"; требование `(1/N)·Σβ_i -> 0`.
- В Colosseum: "Mode B online BC" = DAgger/kickstarting; scripted bot как эксперт работает, если его решение можно вычислить на наблюдении ученика (stateless/по observation) `[вывод]`. Для чужих реплеев эксперт недоступен -> только Mode A + KL к BC.
- Другие pitfalls: BC политика обычно недооценивает редкие, но критичные действия; multi-modal expert (используй cross-entropy на дискретных/композитных действиях, не MSE-среднее); данные разной силы - фильтр/conditioning (AlphaStar MMR); для LSTM нужны длинные последовательности с переносом hidden (AlphaStar: "remember final LSTM state ... reuse for the following trajectory").

### GAIL / AIRL / IQL (кратко)
- GAIL (Ho & Ermon 2016, arXiv 1606.03476): `min_π max_D E_π[log D(s,a)] + E_{π_E}[log(1-D(s,a))] - λH(π)`; policy step TRPO с cost `log D`. Минус: нужен on-policy RL внутри; нестабилен; для framework - потенциальная фаза после BC без reward-сигнала.
- AIRL (Fu et al. 2018, arXiv 1710.11248): `D_θ(s,a) = exp f_θ / (exp f_θ + π(a|s))`; цель - восстановить reward, устойчивый к смене динамики (disentangled reward); reward ambiguity = potential shaping `r̂ = r + γΦ(s') - Φ(s)` (Ng 1999) - формула (3) в статье.
- IQL (Kostrikov 2021, arXiv 2110.06169): expectile regression `L_V = E[ L_2^τ(Q(s,a) - V(s)) ]`, `L_2^τ(u) = |τ - 1(u<0)| u²`; τ=0.5 -> SARSA, τ->1 -> максимум; policy extraction через advantage-weighted regression с inverse temperature β (малые β ≈ BC). Для framework - вариант "offline critic pretraining" (V из реплеев), а также offline->online fine-tuning.
- Применимость к соревнованиям: GAIL/AIRL требуют environment interaction; реплеи чаще дают пары (obs, action) без reward -> BC + KL проще; offline RL имеет смысл при наличии outcome (win/loss) для V^β (Unplugged: one-step).

## E.2 Kickstarting, distillation, KL-regularization

### E.2.1 Policy distillation (Rusu et al. 2015, arXiv 1511.06295)
- Student обучается на данных teacher (DQN): три loss: NLL по argmax-действию teacher; MSE по Q; KL с температурой `L_KL = Σ_i softmax(q^T_i/τ) · ln( softmax(q^T_i/τ) / softmax(q^S_i) )`. Для policy distillation из Q-функций лучший - KL с низкой температурой (τ=0.01: "sharper" targets); MSE хуже (малые различия Q получают малый вес). Дистиллированный student в 4 раза меньше DQN превосходит его; в 15 раз меньше - на уровне.
- Для Colosseum: если teacher - policy (скрипт/BC/старая сеть) -> softmax не нужен, KL между распределениями действий напрямую.

### E.2.2 Kickstarting (Schmitt et al. 2018, arXiv 1803.03835)
Формулы (из статьи):
- Policy distillation: `l_distill(ω,x,t) = H( π_T(a|x_t) || π_S(a|x_t,ω) )` (cross-entropy).
- Kickstarting: `l_kick^k(ω,x,t) = l_RL(ω,x,t) + λ_k · H( π_T(a|x_t) || π_S(a|x_t,ω) )`, `λ_k >= 0` - вес на итерации k; траектории x генерирует СТУДЕНТ (on-policy), teacher только оценивает состояния. `λ_k = 0` для `k > T_0` (после T_0 ученик независим).
- A3C/IMPALA: `l_A3C + λ_k H(π_T || π_S)`, где `l_A3C = log π_S(a_t|x_t)(r_t + γ v_{t+1} - V(x_t)) - β H(π_S)`; value loss отдельно, не дистиллируется.
- Связь: CE = KL(π_T||π_S) + H(π_T); авторы подчёркивают аналогию с entropy-регуляризацией: entropy bonus = KL(π_S||Uniform), kickstart = KL к teacher; цель - "not to converge to teacher's policy", а вспомогательный loss.
- Расписания λ_k (Table 2/Fig. 3, DMLab-30, IMPALA, 10B кадров, teacher маленький): constant (1 или 2) - хорош в начале, но плато около teacher ("trying too hard to match"); linear 1->0 (на 1B/2B/4B кадров) и 2->0 - лучше, "important to reduce this weight quickly, a fact that was not apparent a priori"; PBT-контроль λ - "nearly as well as best manual schedule". На Fig. 2 λ падает до "almost negligible weight after 2 billion steps".
- Эффект (Table 1, single small teacher, large student): score at 0.5B кадров 37.4 (kickstarted) vs 24.1 (scratch); at 10B 56.9 vs 51.9 (mean capped normalised score); кадры до 30.0: 0.13B vs 0.99B.
- PBT: популяция меняет hyperparams (lr, entropy, λ); multi-teacher: `λ_k^i = α_k ρ_k^i` (общий множитель α и per-teacher ρ), чтобы эволюция сдвигала все веса согласованно. PBT не обязателен.
- Multi-teacher: по одному эксперту на задачу; student получает KL от соответствующего teacher.
- Замечание: студент "can also exceed teacher", teacher может быть меньше (1-bot teacher "ignored" by learned λ - PBT выключает вредного teacher).

### E.2.3 Направление KL: что в источниках и что в Colosseum
- Kickstarting: `H(π_T||π_S)` ~ `KL(π_T||π_S)` (forward).
- AlphaStar `rl.py`: `kl(student_logits, teacher_logits, mask) = teacher_probs * (t_logprobs - s_logprobs) * mask` = `KL(teacher||student)`; считается по КАЖДОМУ action argument с маской допустимых аргументов (в Colosseum - по head-ам `CompositeDist`, с маской action_masks).
- VPT: `L_klpt = ρ KL(π_pt, π_θ)` (pretrained первым).
- QDagger: `λ_t E_s[ Σ_a π_T(a|s) log π(a|s) ]` (CE teacher->student).
- Все используют teacher-first. Причина `[вывод]`: forward KL заставляет ученика покрыть все моды teacher (не обнулять вероятности действий, которые teacher иногда делает), что нужно для сохранения разнообразия/exploration; reverse KL `KL(S||T)` mode-seeking: допускает схлопывание на одну моду teacher и даёт градиент через сэмплы ученика. Реверс-KL используется в RLHF-практике (KL(π_θ||π_ref) как штраф за сэмплированные токены) - но это другая литература, в этой подборке не проверял.
- Рекомендация: в `kickstart.py` сделать направление параметром `kl_direction: forward|reverse`, default `forward`, и обновить CLAUDE.md.
- Важно для APPO/V-trace: KL считается на состояниях из trajectory chunk (оффполиси на ~несколько шагов) - это ок; teacher logits можно (а) пересчитывать на learner-е (дороже, но проще, учитывает N teachers/версий), (б) считать на worker-е и хранить в chunk (AlphaStar: поле `teacher_logits` в trajectory) - экономит GPU learner, но требует teacher на CPU worker-ах.

### E.2.4 AlphaStar: KL к human policy в RL (код + `detailed-architecture.txt`)
- Loss RL = actor-critic по V-trace (policy weights: action_type, delay, args по 1.0), UPGO (вес 1.0), TD(λ=0.8) baseline (вес 10.0); pseudo-reward baselines с отдельными весами (E.4); `loss_he = KL_COST·mean(KL) + ACTION_TYPE_KL_COST·mean(KL on action_type)`.
- Из `detailed-architecture.txt`: "entropy loss with weight 1e-4 on all action arguments"; "distillation loss with weight 2e-3 on all action arguments, to match the output logits of the fine-tuned supervised policy which has been given the same observation"; если траектория conditioned на `cumulative_statistics` (z) - дополнительный distillation weight 1e-1 на action_type logits в первые 4 минуты игры. (В `rl.py` маска записана как `game_seconds > 4*60`, что выглядит как "после 4 минут" - расхождение между кодом-pseudocode и текстом, `[не подтверждено]` какой вариант реальный.)
- KL постоянная (не отжигается) в описанных константах: агент "receive a penalty whenever action probabilities differ from supervised policy", "ensures a wide variety of relevant modes of play continue to be explored". Зачем: exploration в огромном action space и сохранение стратегического разнообразия (см. Methods).
- Экспloiters сбрасываются на supervised-агента (весь "human prior" остаётся доступным).
- TStarBot-X: фиксированная KL к слабому supervised сдерживает main agent -> заменили/дополнили DAPO (teacher = предыдущая версия, период ~12 ч).

### E.2.5 VPT (Baker et al. 2022, arXiv 2206.11795): BC + RL fine-tuning с KL
- Данные: ~70k часов видео с IDM-псевдоразметкой; BC foundation model; RL: PPG (Phasic Policy Gradient).
- Loss: `L_klpt = ρ · KL(π_pt, π_θ)`, "this KL divergence loss REPLACES the common entropy maximization loss" (для fine-tuning; при RL с нуля - entropy 0.01).
- Гиперпараметры (Table 6): lr 2e-5; weight decay 0.04; batch 40; batches per iteration 48; context 128; γ=0.999; GAE λ=0.95; PPO clip 0.2; max grad norm 5; max staleness 2; PPG sleep cycles 2; sleep value coef 0.5; sleep aux-value coef 0.5; sleep KL coef 1.0; sleep max sample reuse 6; **KL coefficient ρ = 0.2; ρ decay = 0.9995 per iteration** ("start with a relatively high coefficient and decay it by a fixed factor after each iteration").
- Один ablation (Fig. 16) использовал ρ=0.4 и lr 6e-5; без KL lr пришлось снизить до 3e-6 (sweep из 5 lr), потому что "KL loss prevents making optimization steps that change the policy too much in a single step, especially in early iterations when the value function has not been optimized yet" - KL работает как страховка, пока critic не обучен.
- Эффект без KL: агент получает только ранние предметы (logs, planks, sticks, crafting table), "subsequent skills ... are lost due to catastrophic forgetting"; с RL с нуля - почти 0 reward.
- Value function: "the weights of the (regular) value function are initialized with ZERO weights, which appeared to prevent destructive updates early in training that could happen with a randomly initialized value function"; aux value - random init; value target нормализуется mean/std EMA.
- Оценка шкалы `[вывод]`: ρ_n = 0.2·0.9995^n: half-life ≈ 1386 iterations, через ~4600 итераций ρ≈0.02. Одна PPG-итерация = 48 батчей × 40 × 128 ≈ 246k кадров (оценка по Table 6, если каждый батч свежий).

### E.2.6 QDagger / Reincarnating RL (Agarwal et al. 2022, arXiv 2206.01626) - distillation с "weaning"
- Задача PVRL (policy-to-value reincarnating RL): суб-оптимальный teacher policy + немного его данных -> student другой архитектуры/алгоритма; требования: teacher-agnostic, weaning (отучение), sample-эффективность.
- QDagger: сначала pretrain на данных teacher `D_T` с `L_QDagger(D_T)`; потом на replay студента `D_S`; `L_QDagger(D) = L_TD(D) + λ_t E_{s~D}[ Σ_a π_T(a|s) log π(a|s) ]` (в статье знак/форма как в Eq. 2), `λ_0 = λ`, λ_t убывает линейно по шагам ИЛИ как функция отношения student/teacher performance ("both worked well", Appendix A.3; точная формула ниже).
- Протокол из статьи: student обучается 10M кадров (в 40 раз меньше teacher), "wean off the teacher at 6 million frames"; sweep: температура τ ∈ {0.1, 1.0}, начальный коэффициент λ0 ∈ {1.0, 3.0} (для kickstarting с DQN-студентом лучшим было λ0=3.0, τ=0.1); в workflow-эксперименте Impala-CNN Rainbow использовали λ0 = 1.0, sweep τ ∈ {0.1, 1.0} (для более слабого teacher DQN@20M лучше τ=1.0); lr при pretrain 1e-5, при дообучении уже дообученного DQN сниженный 3e-6 лучше 1e-5 (Appendix A).
- Точная формула weaning для ALE (Appendix A.3): `λ_t = 1[t < t0] · max(1 - G_π / G_{π_T}, 0)`, пересчитывается раз в training iteration (1M кадров), G - средний return студента/teacher (проверено).
- Результаты: kickstarting и DQfD дают деградацию при weaning (особенно 1-step returns); QDagger её избегает; QDagger превышает teacher в 75% запусков; pretrain+fine-tune: "fine-tuning a value-based agent can be an effective reincarnation strategy" (но привязывает к архитектуре).
- Рабочий процесс: "Reincarnating RL" - переиспользовать weights/replay/teacher при смене архитектуры/алгоритма; пример - Nature DQN -> fine-tune c Adam -> QDagger в Impala-CNN Rainbow, превосходит teacher в 5M кадров и tabula rasa на всём обучении 50M. Упоминает OpenAI Five surgery как "ad hoc reincarnation".
- Для Colosseum: если новая архитектура/encoder -> QDagger-подобная схема для policy-based алгоритма = kickstarting + (опционально) pretrain на данных teacher + weaning по ratio производительности: `λ_t = λ0 · 1[t<t0] · max(1 - G_S/G_T, 0)` (формула QDagger, Appendix A.3, проверено).

## E.3 Другие мосты BC -> RL

### JSRL: Jump-Start RL (Uchendu et al. 2022, arXiv 2204.02372)
- Guide-policy π^g (любая: BC, скрипт, человек) ведёт первые `h` шагов эпизода, затем exploration-policy π^e (обучаемая) продолжает; `h` убывает по curriculum (`H_1 = H`, затем H_2 < H_1 ...) пока score комбинированной политики достигает порога β; JSRL-Random: h ~ Uniform из набора.
- Не требует KL/BC-loss, работает с любым RL; есть теория: от exponential-in-horizon до polynomial.
- Нужное условие: возможность "передать управление" mid-episode (для Colosseum: слот игрока управляется скриптом на первых `h` шагах, потом политикой; в соревнованиях с симметричной симуляцией - реализуемо через `PlayerSlot`/wrapper `GuideRollin(env, guide, h)`) `[вывод]`.
- Важное наблюдение (Fig. 2, App. Fig. 7-8): наивная инициализация политики из pretrain + СЛУЧАЙНО инициализированный critic -> actor performance "decays, as the untrained critic provides a poor learning signal, causing the good initial policy to be forgotten". Baseline с "critic warm-up": 100k шагов rollout-а pretrained policy для обучения critic при замороженном actor, затем fine-tune.

### Offline -> online без Q-pretrain: PORL (Xiao et al. 2025, arXiv 2505.16856)
- Показывают, что предобученные консервативные Q-функции мешают exploration online; предлагают быстро инициализировать Q с нуля при онлайн-фазе, имея только offline политику (в т.ч. из BC). Статья свежая, использование как референс `[вторично: прочитан abstract/intro]`.

### Behavior-regularized / AWAC / Lowe и др.
- Отдельные работы "Behavior-regularized actor critic", "Lowe" и т.п. в этой выборке не проверял `[не подтверждено]`; их место заняли проверенные VPT/JSRL/QDagger/IQL.

## E.4 Reward shaping и annealing

### Potential-based shaping (Ng, Harada, Russell 1999)
- `F(s,a,s') = γΦ(s') - Φ(s)`, `r' = r + F`; такая форма сохраняет оптимальную политику для любого Φ и, в отсутствие знания динамики, это единственный класс с инвариантностью (в статье - necessary and sufficient). Я проверил формулу через цитирование в AIRL (Eq. 3: `r̂(s,a,s') = r(s,a,s') + γΦ(s') - Φ(s)`); PDF оригинала не распарсился, `[вторично]` для точной формулировки теоремы.
- Шейпинг ВНЕ этого класса меняет оптимум; OpenAI Five: "shaped reward is modeled loosely after potential-based shaping functions, though the guarantees therein do not apply".
- `[вывод]` Annealing: `r'_t = r + α_t · (γΦ(s') - Φ(s))`, α: 1 -> 0 линейно. Для каждого фиксированного α оптимум не меняется, но значения V сдвигаются/масштабируются со временем: нужен value-normalization (running mean/std; VPT это делает) и осторожность с V-trace/GAE бутстрапом (non-stationary targets).

### Практики annealing
- Toad Brigade: shaped reward только 20M шагов, затем sparse; более крупные сети учат сразу с sparse + teacher KL.
- Bansal 2018: dense "exploration reward" `α_t · r_explore + (1-α_t) · r_competition`, α линейно -> 0 за 500 итераций (1000 для kick-and-defend); агенты, оптимизирующие sparse competition reward, побеждают тех, кто всё время получал dense (Fig. 3).
- AlphaStar pseudo-rewards (supplement): build order reward = отрицательное расстояние Левенштейна между build order людей и агента (стоимость замены по квадрату расстояния до сущности, масштаб [0, 0.8]), награда при каждом изменении order; built units / upgrades / effects reward = Hamming distance к человеческой игре; множитель 0.5 после 8 мин, ещё ×0.5 после 16 мин, после 24 мин - 0 ("time decay"); веса policy/baseline: build order 4.0/1.0, units 6.0/1.0, upgrades 6.0/1.0, effects 6.0/1.0; без UPGO. Это "reward для следования стратегии z", отключаемый по времени эпизода (не по обучению).
- OpenAI Five: (i) `ρ_i <- ρ_i · 0.6^(T/10 min)` - экспоненциальное затухание каждой награды по времени игры; (ii) team spirit `r_i = (1-τ)ρ_i + τ·ρ̄` (ρ̄ - среднее по команде), τ=0: каждый сам за себя, τ=1: награда делится поровну; "lower team spirit reduces gradient variance in early training"; Rerun: τ 0.3 -> 0.8 (цель 1.0); (iii) zero-sum: из награды героя вычитается среднее награды противников. Полный список hyperparam-annealing выше (D.7).
- Lux S1 (Toad): shaping за города/юниты/research/fuel.
- Для Colosseum: реализовать `RewardSchedule`: `r = r_sparse + Σ_k w_k(t) · r_shaping_k`, где `w_k(t)` задаётся по числу learner steps/frames (линейно/exp), и параметр `team_spirit(t)` для N-player/командных режимов.

## E.5 Value warmup (предобучение critic при замороженной политике)

Подтверждённые факты:
1. JSRL (Fig. 2): random critic после actor pretrain -> политика забывается, performance падает; critic warm-up на rollout-ах pretrained policy (100k шагов; в Fig. 7/8 pretrain на 100k и 1M offline transitions) убирает проблему и является более сильным baseline.
2. VPT: value head с нулевыми весами (последний слой), value targets нормализованы EMA mean/std; KL к pretrained policy (ρ=0.2 -> decay) замещает entropy; без KL нужен lr в 6.7 раз меньше (2e-5 -> 3e-6 при sweep из 5).
3. Unplugged: value `V^β` behavior policy можно обучить офлайн на реплеях (one-step offline RL), потом использовать для улучшения политики: "first train a model to estimate the behavior policy and behavior value function, then use the behavior value function to improve the policy".
4. Andrychowicz et al. 2020 "What Matters in On-Policy RL" (arXiv 2006.05990; continuous control, PPO, 250k моделей): (a) "initial policy has surprisingly high impact": last policy layer init ×100 меньше, std offset; (b) "Always use observation normalization and check if value function normalization improves performance"; (c) GAE λ=0.9 без Huber loss и без PPO-style value-loss clipping; (d) γ - один из самых важных, начинать с 0.99; (e) Adam, β1=0.9, tuned lr. Это НЕ про BC warm start: применимость к дискретным/composite action spaces ограничена `[вывод]`.
5. Nikishin et al. 2022 "Primacy Bias" (arXiv 2205.07802): агенты с replay переобучаются на ранних данных; решение - periodic reset последних слоёв агента (+ Q-функции) при сохранении буфера. Для on-policy APPO (без replay) прямой применимости нет; полезная идея `[вывод]`: после долгого BC (нейросеть "переобучена" на replay-распределении) и перед RL можно "сбросить" только value head и (опционально) последний слой policy-головы к функционально-эквивалентному/нулевому виду.
6. Pitfall "BC -> RL падение": initial performance drop при fine-tuning объясняют рассогласованием actor/critic [вторично, поиск]; PORL (2025): critic init from scratch online.

Предлагаемая процедура Value Warmup `[вывод на базе JSRL + VPT + Unplugged]`:
- Stage V0 (опционально, если есть исходы/реплеи): обучить value head офлайн на Monte-Carlo/λ-returns реплеев (`V^β`) при замороженных encoder+policy (или маленьком lr). Проблема: V^β - значение политики людей/скрипта, не текущей политики => нужен V1.
- Stage V1: policy ЗАМОРОЖЕНА (lr_policy = 0, KL не нужен), workers играют BC-политикой против целевого пула; learner обучает ТОЛЬКО value (и опционально aux) на V-trace/GAE-таргетах; длительность по числу шагов/до плато explained-variance (ориентир JSRL: 100k env steps; для сложных игр - существенно больше `[эвристика]`).
- Stage V2: размораживание политики с linear LR warmup (0 -> lr за N шагов) + KL к BC (E.2.5), как у OpenAI Five после surgery (LR=0 первые часы, чтобы Adam моменты приспособились).
- Мониторинг: explained variance critic, KL(π||π_BC), entropy, winrate vs anchors (не должен падать).

## E.6 Смена архитектуры/пространств посреди соревнования

### E.6.1 Распределение: какой метод когда
| Ситуация | Метод |
|---|---|
| Те же входы/выходы, шире/глубже сеть | Net2Net или surgery с точным сохранением функции |
| Добавили входные признаки (observation) | Нулевая инициализация новых входных весов (OpenAI Five Eq. 9) |
| Новые действия (расширение action head) | Zero-init новых логитов/маска + annealing включения (OpenAI Five), либо distillation |
| Радикальная смена encoder/архитектуры (CNN -> transformer, другая длина LSTM) | Policy distillation + kickstarting (Toad Brigade chain, QDagger) |
| Сменились правила среды | Annealing доли игр в новой среде 0 -> 100%; возможно fine-tune |
| Нужно удалить параметры/признаки | Surgery не умеет; оставить "deprecated" входы константами (OpenAI Five), либо distillation |

### E.6.2 OpenAI Five surgery: формулы (Appendix B, arXiv 1912.06680)
- Цель: `∀o: π̂_θ̂(o) = π_θ(o)` (Eq. 1); при добавлении наблюдений - равенство "на шаг раньше": `π̂(Ê(s)) = π(E(s))` для всех game states s (Eq. 8).
- Расширение слоя `y = W1 x + B1`, `z = W2 y + B2`: размер y: `d_y -> d̂_y`: `Ŵ1 = [W1 | R()]`, `B̂1 = [B1 ; R()]`, `Ŵ2 = [W2 ; 0]` (Eq. 6): первые `d_y` активаций совпадают со старыми, новые случайны, но следующий слой их игнорирует (нулевые строки); нулевые веса "move away from zero due to gradients"; симметрия сломана случайной инициализацией входящих весов новых юнитов.
- Увеличение LSTM 2048 -> 4096: нельзя разделить как в Eq. 6 (рекуррентность), поэтому новые веса - маленькие случайные; масштаб подобран эмпирически: "highest scale which did not noticeably decrease agent's TrueSkill".
- Новые входные признаки: `Ŵ = [W | 0]` (Eq. 9) - выход `ŷ = y`.
- Изменение среды/действий: annealing "starting with 0% of rollout games played with the new environment or actions, and slowly ramping up to 100%"; при падении TrueSkill - откат и более медленный ramp. Пример: передача Buyback из скрипта в модель - без annealing упало бы качество (модель сначала хуже скрипта, и агент подстраивает стратегию под "плохого" союзника/врага).
- Удаление параметров: "most surgeries which remove parameters are not possible" -> deprecated inputs остаются константами.
- Smooth Training Restart: Adam моменты ломаются при смене формы -> LR = 0 первые несколько часов после surgery; это же даёт rollout-играм выйти на steady state.
- Past opponents: все замороженные предыдущие версии надо преобразовывать тем же surgery (rollout GPUs запускают новый код) => точность сохранения функции критична.
- Для Colosseum `[вывод]`: у нас каждый checkpoint может хранить собственную `NetworkConfig` + версию encoder-а, а worker network pool уже работает по `(agent_id, network_id)`; поэтому старые чекпоинты в пуле можно НЕ конвертировать, а загружать со старой архитектурой/encoder-ом (требуется versioned `BaseEncoder` для старого observation spec). Это снимает жёсткое требование exact-preservation для пула, но не для самого обучаемого агента.

### E.6.3 Net2Net (Chen, Goodfellow, Shlens 2015, arXiv 1511.05641)
- Net2WiderNet: случайное отображение `g: {1..q} -> {1..n}` (первые n - тождественно), `U^(i)_{k,j} = (1/c_j)· W^(i)_{g(k),g(j)}`, где `c_j` - сколько раз исходный юнит реплицирован; сохраняет функцию точно. Для break symmetry - небольшой шум на копиях ("add a small amount of noise to all but the first copy"). Для свёрток - то же по каналам.
- Net2DeeperNet: вставка слоя, инициализированного identity (для ReLU точно; для conv - identity filters; BatchNorm требует особой настройки, чтобы быть идентичным); шум не нужен.
- Преимущества: новая сеть сразу не хуже старой; любое локальное улучшение - улучшение. Минусы для RL: оптимизатор/Adam state сбрасывается; дубликаты нейронов "слипаются" без шума.
- Для рекуррентных/attention слоёв прямого рецепта нет в статье (в основном Inception CNN) `[вывод]`.

### E.6.4 Distillation в новую сеть (когда exact preservation невозможна)
- Student = новая сеть (любая архитектура/encoder), teacher = старая политика (frozen). Loss: `L = L_APPO + λ_k KL(π_T||π_S)` (kickstarting) + опционально supervised distillation на данных из replay buffer teacher (QDagger-pretrain). Для value: добавить `L_V-distill = c_v · MSE(V_S, V_T)` на тех же состояниях, затем отключить `[вывод]` (kickstarting сам value не дистиллирует).
- Опыт Toad Brigade: каждая следующая сеть (16-блок, 24-блок) учится с предыдущей как teacher'ом и sparse reward; итоговые 24 residual blocks (~20M параметров).
- OpenAI Five упоминает, что альтернативу surgery предлагают "[24]" (ссылка на работу по другому способу surgery без требования exact equality) `[не подтверждено, что именно]`.

### E.6.5 Resume с изменённым observation/action spaces: чек-лист
1. Обновлять `ActionSpec`/`CompositeDist` версионированно; новый head - zero-init весов и bias = -10 (≈0 вероятности) ИЛИ маска `action_masks` выключена до включения (annealed) `[вывод]`.
2. Observation: добавлять каналы/признаки на конец, `W_new_cols = 0`; нормализаторы (running mean/std) для новых признаков инициализировать нейтрально.
3. Optimizer: сбросить Adam, `lr = 0` на N шагов, затем linear warmup; для APPO также сбросить value normalizer? (нет - сохранить, но проверить).
4. Mixture-annealing среды/действий 0 -> 100% по доле worker-ов или матчей; gate по winrate vs anchor.
5. Проверка эквивалентности: на фиксированном наборе состояний `max |logits_new - logits_old| < ε` (для exact surgery) или `KL < ε` (approximate).
6. LSTM: не менять hidden size без surgery; либо distill.
7. Пересчитать Elo/winrate матрицу (после изменения среды старые рейтинги несопоставимы; OpenAI Five: TrueSkill считается на финальной среде - "biased against earlier models").

## E.7 Reincarnating RL / persistent workflow
- Идея (Agarwal 2022): результаты предыдущих вычислений (weights, replay buffer, teacher policy, данные) переиспользуются при смене архитектуры/алгоритма/гиперпараметров; "persistent" workflow = каждая итерация исследования стартует из лучшей имеющейся версии, а не tabula rasa; цель - сократить стоимость research-итераций (пример: DQN fine-tune за часы вместо недели; Impala-CNN Rainbow reincarnated превосходит tabula rasa на 50M кадров).
- OpenAI Five обосновал то же в среде с меняющейся кодовой базой (surgery), Toad Brigade - chain of teachers.
- Для Colosseum: хранить в Weight Store/CheckpointManager: (a) weights+optimizer+config+encoder version; (b) небольшой replay "teacher dataset" (obs, teacher_logits/actions, value) для QDagger-pretrain; (c) eval-манифест (набор anchor-игр) для проверки отсутствия регресса после миграции.

---

# РЕКОМЕНДОВАННЫЕ РЕЦЕПТЫ ДЛЯ ФРЕЙМВОРКА

Общая договорённость: все коэффициенты ниже - стартовые значения для sweep. Везде логировать: winrate vs anchors (scripted + frozen BC), KL(π||π_ref), entropy, explained variance critic, Elo.

## R0. Cold start (baseline, ничего не делаем)
- Для сравнения: Toad Brigade показал, что со случайной инициализации + shaped reward 20M шагов + sparse reward + self-play + KL к teacher можно выиграть Lux S1.

## R1. Offline BC -> APPO с KL-якорем (VPT/AlphaStar-стиль) [основной режим, если есть реплеи]
1. Offline BC: cross-entropy на (obs, action[, action_masks]) из реплеев, cosine LR, `lr0 ≈ 5e-4` (Unplugged BC) для основной стадии; fine-tune на top-tier фильтре `lr ≈ 1e-5` (AlphaStar/Unplugged). Фильтр данных по рейтингу игрока (топ ~20% [AlphaStar: MMR>3500]) `[эвристика]`; условный вход - рейтинг игрока, при RL ставить максимум.
2. Value head: zero-init (VPT) + value normalization (EMA mean/std).
3. Value warmup V1 (R5), затем APPO с loss: `L = L_APPO + ρ_t · KL(π_BC || π_θ)`; entropy bonus отключить, пока ρ_t > 0.01 (VPT: KL заменяет entropy; AlphaStar: entropy 1e-4 + KL 2e-3 вместе - можно оставить крошечный entropy) `[эвристика]`.
4. `ρ_0 = 0.2` (VPT), `ρ_t = ρ_0 · decay^n`, `decay = 0.9995` на "learner iteration" (≈ один проход по накопленным chunk-ам, определить в конфиге: `kl_decay_per_updates`), floor `ρ_min = 0` (VPT) либо `1e-3` (AlphaStar-стиль постоянный якорь). Sweep ρ_0 ∈ {0.02, 0.1, 0.2, 0.5}; lr RL ≈ 2e-5 (VPT).
5. Направление KL: `forward` (teacher->student), per-head для `CompositeDist` с учётом action_masks.
6. Критерий выхода из якоря: ρ -> 0 или winrate vs BC-политики > 70% (по аналогии с AlphaStar `0.7`).

## R2. Online kickstarting (Mode B, teacher = скрипт/BC/предыдущая сеть)
- `L = L_APPO + λ_k · KL(π_T || π_S)`, траектории генерирует ученик; teacher считается на learner или хранится в chunk (`teacher_logits`).
- Расписание: `λ_0 = 1.0` (в нормировке лосс-компонентов APPO; sweep {0.5, 1, 2}), linear -> 0 за `10-20%` общего бюджета (Schmitt: за 1-4B из 10B; "reduce quickly"), после T0 `λ=0`. Альтернатива QDagger (формула проверена): `λ_t = λ_0 · 1[t<t0] · max(1 - G_S/G_T, 0)`, G - скользящий средний return/score ученика и teacher'а против одного и того же пула, пересчёт раз в N шагов (в статье - 1M кадров); перенос на winrate в играх `[вывод]`.
- Multi-teacher: `λ^i = α·ρ^i`; один глобальный α затухает, ρ^i - относительные веса.
- Скрипт-бот как teacher: распределение - one-hot с label smoothing 0.1 `[эвристика]` (TStarBot-X: one-hot/multi-hot/label smoothing).

## R3. Rule-Guided KL (RGPS) - постоянный маленький якорь на критических состояниях
- `L_RGPS = μ · 1[s ∈ S_critical] · KL(π_rule(·|s) || π_θ(·|s))`, `S_critical` задаётся предикатом скрипта (например, "могу захватить", "угроза базе"); μ ≈ 0.01-0.1 `[эвристика]`; не отжигать до нуля (TStarBot-X: активен постоянно); включать только на main agent.
- Полезен там, где RL не находит редкие критические решения за тысячи шагов.

## R4. Online BC / DAgger (Mode B строго)
- `π_i = β_i π* + (1-β_i) π_θ`: действия при сборке траекторий выбирает эксперт с вероятностью β_i (по эпизодам или по шагам), метки эксперта на состояниях ученика копятся в aggregated buffer; `β_1 = 1`; default `β_i = I(i=1)` (Ross: "often performs best") с вариантом экспоненциального затухания `β_i = 0.5^{i-1}` `[эвристика]`.
- Применимо только при доступности эксперта в рантайме (скрипт).

## R5. Value Warmup (обязательный шаг между BC и RL)
- V0 (опционально) - офлайн value на реплеях; V1 - on-policy с замороженной политикой: `policy_lr = 0`, `value_lr = обычный`, шагов >= 100k env steps (JSRL) и до плато explained variance > 0.5 `[эвристика]`; V2 - linear warmup policy LR за ~1-5% шагов, KL из R1.
- Опция: `reset_value_head=true` перед RL (primacy-bias идея, `[вывод]`).

## R6. Guide roll-in (JSRL) со скриптом
- `h` - число шагов, которые ведёт скрипт в начале эпизода; `h_0 = эпизод целиком / 50% длины`, уменьшать по curriculum, когда winrate комбинированной политики >= β (0.5-0.7 `[эвристика]`); для N-player: скрипт в слотах-"учениках" первые h шагов; collect_mask для шагов скрипта = False (off-policy данные для V-trace не годятся без поправок) `[вывод]`.

## R7. Reward shaping annealing
- `r = r_sparse + Σ_k w_k(t) r_k`, `w_k(t)` linear -> 0 за 10-20% бюджета (Toad Brigade: 20M шагов; Bansal: 500 итераций), либо time-in-episode decay `0.5` после T1/T2 (AlphaStar-стиль) либо `0.6^(T/10min)` (OpenAI Five); `team_spirit τ: 0.3 -> 0.8 -> 1.0` в командных играх.
- Предпочитать potential-based `γΦ(s') - Φ(s)`; для непотенциальных - обязательно annealing + мониторинг winrate по sparse-метрике.
- Value normalizer обязателен.

## R8. Миграция архитектуры
- R8a (surgery, exact): расширение слоя (Eq. 6), новые входы `[W|0]` (Eq. 9), `lr=0` + reset Adam, LR warmup; проверка эквивалентности на фиксированном батче; annealing новых действий/механик 0->100%.
- R8b (distill): новая сеть как student: `L = L_APPO + λ_k KL(π_old||π_new) [+ c_v MSE(V_new, V_old)]`, `λ_0=1`, linear -> 0 за ~10-20% бюджета; chain: small -> big (Toad Brigade).
- R8c (QDagger-pretrain): перед онлайн-фазой несколько эпох на teacher-dataset (obs, π_T) с `lr ≈ 1e-5..` (Agarwal использовал 1e-5 для pretrain) `[эвристика для policy-based]`.
- R8d (Net2Net): только для чисто feed-forward/conv блоков, шум 1e-3·std на копиях `[эвристика]`.

## R9. Скриптовые боты в финальной арене / league
- Пул: trainable агенты + frozen checkpoints + scripted anchors (fixed, `collect_mask=False`, не выходят из пула).
- Для каждого trainable: 50% PFSP по league (`squared`) / 35% self-play (+auto-fallback на historical при winrate <= 0.3) / 15% verification (forgetting: min winrate vs historical < 0.7; exploiters < 0.3) - AlphaStar-доли; внутри "league" scripted anchors входят как historical с floor-вероятностью `>= 2%` `[эвристика]`; суммарно 10-20% матчей с anchor-слотами `[эвристика]`.
- Exploiters: main exploiter (играет только против текущего main; reset при checkpoint), league exploiter (PFSP `linear_capped`; reset с p=0.25); условия checkpoint: min winrate > 0.7 либо лимит шагов (AlphaStar: 2e9/4e9; в Colosseum масштабировать пропорционально бюджету).
- Quality-score sampling прошлых версий (OpenAI Five): `p_i ∝ exp(q_i)`, `q_i -= η/(N p_i)` при победе текущего, `η=0.01`, snapshot каждые 10 итераций с `q = max q`.

## R10. Exploitability audit (`colosseum eval --exploit`)
- Заморозить A; обучить exploiter (R1 с BC от лучшего скрипта/реплеев или с нуля) на ~3-10% бюджета A; отчёт: winrate (Wilson CI) exploiter vs A, A vs anchors; добавить exploiter в пул; повторять каждые K checkpoint-ов. Cap на длительность. Для N-player - exploiter в одном слоте.

## R11. Forgetting guard / регрессия
- Постоянная матрица winrate "текущий vs последние N historical + anchors"; алерт, если min < 0.7 (AlphaStar-порог) -> автоматически повысить долю verification-матчей и/или поднять KL-якорь ρ.

## Приоритет реализации в Colosseum `[вывод]`
1. R5 (value warmup) + R1 (BC->APPO + KL с decay, forward KL) - максимальная отдача при малой цене (есть `kickstart.py`, нужно лишь поменять направление KL + добавить decay-режим и freeze-policy phase).
2. R9/R11 (anchors + verification branch) - внутри `matchmaker.py`/`coordinator`.
3. R2 с performance-based weaning (QDagger) + R8b (distill) - общий механизм "teacher" (скрипт/BC/старая сеть).
4. R7 (RewardSchedule), R3 (RGPS), R10 (exploit audit).
5. R8a/R8d (surgery утилиты), R4/R6 (DAgger/JSRL) - по мере необходимости.

---

# Что осталось неподтверждённым

- Точное число 971,000 replays и MMR 3500 из текста Nature Methods: подтверждено только MMR 3500 по коду `supervised.py` и Unplugged; число - вторичный источник.
- Точное значение KL_COST/ACTION_TYPE_KL_COST в `rl.py` (в pseudocode - константы без значения); значения 2e-3 / 1e-1 / entropy 1e-4 взяты из `detailed-architecture.txt`.
- Расхождение "первые 4 минуты" vs `game_seconds > 4*60`.
- Доля scripted ботов в оптимальной популяции - прямых исследований не найдено.
- Детали Lux S2/S3 победителей, Hungry Geese, WeKick writeup (только вторичные упоминания).
- Точная форма Theorem 1 Ng 1999 - читал только через AIRL.
- Minimax Exploiter - детали reward.
- Kaggle Episode scraper / Meta Kaggle Episodes - схема не проверена.

---

# Источники (URL)

Тема D
- Gleave et al., Adversarial Policies: https://arxiv.org/abs/1905.10615
- Wang et al., Adversarial Policies Beat Superhuman Go AIs: https://arxiv.org/abs/2211.00241
- Tseng et al., Can Go AIs Be Adversarially Robust?: https://arxiv.org/abs/2406.12843
- Minimax Exploiter: https://arxiv.org/abs/2311.17190
- OpenAI Five (Dota 2 with Large Scale Deep RL): https://arxiv.org/abs/1912.06680
- AlphaStar (Nature): https://www.nature.com/articles/s41586-019-1724-z ; supplement (pseudocode.zip, detailed-architecture.txt): https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-019-1724-z/MediaObjects/41586_2019_1724_MOESM2_ESM.zip ; blog: https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/
- TStarBot-X: https://arxiv.org/abs/2011.13729
- Bansal et al., Emergent Complexity via Multi-Agent Competition: https://arxiv.org/abs/1710.03748
- Lanctot et al., PSRO / unified game-theoretic approach: https://arxiv.org/abs/1711.00832
- Czarnecki et al., Real World Games Look Like Spinning Tops: https://arxiv.org/abs/2004.09468
- Heinrich & Silver, NFSP: https://arxiv.org/abs/1603.01121
- Jaderberg et al., CTF (FTW): https://arxiv.org/abs/1807.01281
- Baker et al., Hide & Seek: https://arxiv.org/abs/1909.07528
- TiZero: https://arxiv.org/abs/2302.07515 ; TiKick/WeKick mention: https://arxiv.org/abs/2110.04507
- Toad Brigade (Lux S1): https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
- Lux AI S2 NeurIPS: https://neurips.cc/virtual/2023/competition/66593 ; Lux S3 repo: https://github.com/Lux-AI-Challenge/Lux-Design-S3

Тема E
- Kickstarting: https://arxiv.org/abs/1803.03835
- Policy Distillation: https://arxiv.org/abs/1511.06295
- VPT: https://arxiv.org/abs/2206.11795
- Reincarnating RL (QDagger): https://arxiv.org/abs/2206.01626
- JSRL: https://arxiv.org/abs/2204.02372
- PORL: https://arxiv.org/abs/2505.16856
- Net2Net: https://arxiv.org/abs/1511.05641
- DAgger: https://arxiv.org/abs/1011.0686
- GAIL: https://arxiv.org/abs/1606.03476 ; AIRL: https://arxiv.org/abs/1710.11248 ; IQL: https://arxiv.org/abs/2110.06169
- AlphaStar Unplugged: https://arxiv.org/abs/2308.03526
- What Matters in On-Policy RL: https://arxiv.org/abs/2006.05990
- Primacy Bias: https://arxiv.org/abs/2205.07802
- Ng, Harada, Russell 1999: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- Meta Kaggle: https://www.kaggle.com/datasets/kaggle/meta-kaggle ; Kaggle API docs: https://github.com/Kaggle/kaggle-api/blob/main/docs/competitions.md
- Обзор AlphaStar (вторичный, 971k replays): https://arxiv.org/abs/2111.07631

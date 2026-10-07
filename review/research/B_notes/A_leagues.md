# Заметки: лиги, self-play (тема A) и N-игроковые FFA / командные игры (тема B)

Дата: 2026-10-07. Контекст: Colosseum (APPO, IMPALA-style, BC -> self-play с FIFO-пулом, latest_prob=0.5 -> PFSP/League, solo/arena матчи, ELO). Команда 1-3 человека, 2-8 недель.

Конвенции:
- Каждый факт снабжён URL (arXiv/первоисточник). Тексты PDF были скачаны и прочитаны через pdftotext; числа ниже взяты из текста статей.
- "[не подтверждено]" = не удалось проверить в первоисточнике или это моя экстраполяция.
- "[расчёт]" = мой арифметический вывод из приведённых чисел (не цитата).
- "[рекомендация]" = мой инженерный вывод для Colosseum, не факт из статьи.
- Мелкая оговорка по извлечению текста: в PDF Nature показатели степени потеряны ("2 * 10s steps"); значения 2e9 и 4e9 подтверждены независимо в TStarBot-X и в DI-engine (см. A.1).

Сокращения: MA = main agent, ME = main exploiter, LE = league exploiter, SP = self-play, FSP = fictitious self-play, PFSP = prioritized FSP, PSRO = Policy-Space Response Oracles, RPP = Relative Population Performance, BT = Bradley-Terry, PL = Plackett-Luce.

---------------------------------------------------------------------------------------------------

# ЧАСТЬ A. Схемы лиг и self-play

## A.1 AlphaStar League (Vinyals et al., Nature 2019)

Источники:
- Статья: https://www.nature.com/articles/s41586-019-1724-z
- Копия текста (Methods), которую я читал: https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf
- Независимая реимплементация с теми же числами: https://di-engine-docs.readthedocs.io/en/latest/_modules/ding/league/starcraft_player.html

### Три типа агентов (отличаются только распределением оппонентов, моментом снапшота и вероятностью сброса)
- Main agents (MA): никогда не сбрасываются. Оппоненты: 35% SP, 50% PFSP против всех прошлых игроков лиги, 15% PFSP против "forgotten" main-игроков (тех, кого агент уже не может побеждать) и прошлых main exploiters. Если нет forgotten/сильных exploiters, эти 15% идут в self-play. Снапшот (замороженная копия -> новый player в лиге) каждые 2e9 шагов. URL: Nature (раздел Methods "Populating the League").
- League exploiters (LE): PFSP против всей лиги. Снапшот, когда побеждают всех игроков лиги более чем в 70% игр, либо по таймауту 2e9 шагов; после снапшота с вероятностью 25% сбрасываются на supervised-параметры. Цель: найти "системные слепые пятна" лиги. На LE не нацелены main exploiters. URL: Nature.
- Main exploiters (ME): играют только против main agents. В 50% случаев и если текущая вероятность победы < 20%, используют PFSP c f_var по прошлым снапшотам main-агентов (curriculum); иначе играют против текущих MA. Снапшот, когда побеждают всех трёх MA более чем в 70% игр, либо по таймауту 4e9 шагов; затем всегда сбрасываются на supervised-параметры. URL: Nature.
- Подтверждение 2e9/4e9/25%/70% по коду: DI-engine `MainPlayer` (branch 0.5 pfsp / 0.35 sp / 0.15 veri; snapshot каждые 2e9), `MainExploiter` (>70% против всех main или 4e9, всегда mutate/reset), `LeagueExploiter` (>70% против лиги или 2e9, mutate_prob 0.25). URL: https://di-engine-docs.readthedocs.io/en/latest/_modules/ding/league/starcraft_player.html
- Оценка шагов на рестарт из сторонней статьи: ME ресетится примерно раз в сутки (4e9 шагов), LE примерно раз в 2 дня (2e9 шагов, p=0.25 сброса). URL: https://arxiv.org/pdf/2011.13729 (раздел 4.5).

### PFSP формулы (Nature, Methods "Prioritised Fictitious Self-Play")
- Агент A выбирает замороженного оппонента B из кандидатов C с вероятностью P(B) = f(P[A beats B]) / sum_{C} f(P[A beats C]).
- f_hard(x) = (1 - x)^p, где p in R+ управляет энтропией распределения. f_hard(1)=0, то есть игры против уже побеждаемых не тратятся. Это дефолт. Интерпретация авторов: сглаженная аппроксимация max-min (в отличие от max-avg у FSP); помогает учитывать редкие сильные контрстратегии (exploits).
- f_var(x) = x(1 - x): играть с оппонентами "около своего уровня". Используется для main exploiters и "struggling" main agents.
- Конкретное p в статье в тексте Methods не указано; в открытой реимплементации DI-engine "squared" = (1-x)^2, "variance" = x(1-x), нормировка probs = f / sum f, при всех нулевых win rates - uniform. URL: https://raw.githubusercontent.com/opendilab/DI-engine/main/ding/league/algorithm.py. p=2 для самого AlphaStar [не подтверждено, берём из реимплементации].

### Размеры и вычисления
- 3 main + 3 main exploiter + 6 league exploiter агентов (по расам: 1 MA, 1 ME, 2 LE на расу) = 12 параллельных actor-learner копий. Каждый обучался на 32 TPUv3 в течение 44 дней; за обучение создано почти 900 различных players. URL: Nature.
- На каждого агента: 16 000 параллельных матчей StarCraft II, 16 actor-задач (каждая на TPU v3 из 8 ядер) для inference, learner 128 ядер TPU, batch 512 (4 последовательности на ядро), данные реплеятся дважды, около 50 000 agent steps/сек, обновление параметров у акторов каждые 10 сек. Итого 3072 TPU-ядра, около 50 400 preemptible CPU-ядер (цифры TPU/CPU сведены в TStarBot-X Table 1). URL: Nature; https://arxiv.org/pdf/2011.13729
- Один центральный coordinator хранит оценку payoff-матрицы, выдаёт матчи по запросу, ресетит exploiters; отдельные CPU-evaluator воркеры доуточняют payoff. URL: Nature (Infrastructure). Это прямой аналог Colosseum Coordinator + WinRateTracker.
- Основной агент потребил около 1.9e11 шагов за 44 дня (около 50 000 шагов/сек). URL: https://papers.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf (Appendix A.6, ROA-Star).

### Абляции (Fig. 3 Nature; упрощённая постановка: одна карта, Protoss vs Protoss, 1e10 шагов)
- Состав лиги, Test Elo / RPP: только main agents 1540 / 6%; + main exploiters 1693 / 35%; + league exploiters 1824 / 62%. То есть exploiters дают +284 Elo и сильно снижают exploitability лиги. URL: Nature, Fig. 3A,B.
- Алгоритм мультиагентного обучения, Test Elo / min win-rate против прошлых версий (мера забывания): pFSP+SP 1540 / 71%; SP 1519 / 46%; pFSP 1273 / 70%; FSP 1143 / 69%. Вывод: чистый SP почти такой же по Elo, но сильно забывает (46%); FSP по всей истории медленно учится (1143); смесь SP+PFSP лучшая. URL: Nature, Fig. 3C,D.
- Структура payoff: main agents ведут себя транзитивно (новые побеждают старые); взаимодействия exploiters сильно нетранзитивны: около 3 000 000 rock-paper-scissors циклов (при порогах win-rate 70%) с участием хотя бы одного exploiter, и около 200 циклов только среди main agents. URL: Nature, Extended Data Fig. 8.
- Нет регресса: Nash-распределение лиги в каждый момент даёт малую вероятность старым игрокам; MA росли стабильно, ME со временем стали слабее (MA стали устойчивее). URL: Nature (Empirical Evaluation).

## A.2 Упрощённые лиги

### A.2.1 TStarBot-X (Tencent AI Lab, 2020-2021)
URL: https://arxiv.org/pdf/2011.13729
- Ресурсы: 144 Nvidia V100 (96 на обучение, 48 на inference), 13 440 CPU-ядер, 6 720 одновременных матчей, 57 дней, модель 20M параметров (AlphaStar: 139M). Скорости потребления/генерации данных около 1/30 и 1/73 от AlphaStar. Таблица 1.
- Главный вывод: "faithful reimplementation of AlphaStar's methods can not succeed" при малых ресурсах. "AlphaStar Surrogate" (1 MA + 2 ME + 2 LE) в нескольких повторах сходился в ранние Zergling Rush или Roach Push за несколько дней, разнообразие лиги было ограничено. Abstract и раздел 1.
- Проблемы реимплементации: при тех же числах шагов один период exploiter занимает >10 дней (лига слишком разрежена); при меньшем числе шагов ME/LE слабы и не догоняют непрерывно учащийся MA. Раздел 4.5.
- Новые роли: Specific Exploiter (старт от fine-tuned supervised модели под конкретную стратегию, сброс как у ME); Evolutionary Exploiter (сброс на лучшего исторического EE против MA; снапшот при timeout или >70%); Adaptive Evolutionary Exploiter (AEE; наследуется от "листового" исторического узла, чей win-rate против MA в интервале 20-50%, иначе сброс на supervised; если несколько - тот, чей win-rate ближе к 50%). AEE "continually adds comparable opponents for MA", стабилизируя обучение MA. Раздел 4.5.
- Параметры: MA использует 25% SP, 60% PFSP, 15% PFSP против forgotten; снапшот MA каждые 3e8 шагов (около 12 часов); у exploiters период: минимум 7.65e7 шагов, затем проверки до максимума 1.53e8 шагов. MA - 32 GPU, каждый exploiter - 8 GPU. Раздел 4.5.
- Итоговая лига: 1 MA + 8 exploiters (ME/LE/SE + AEE); за 57 дней 25 708 491 матч, 583 агента (124 main, 452 exploiters, 7 supervised). Раздел 5.1.2.
- DAPO (agent-level distillation к прошлой версии) включён в последние 15 дней из 57; даёт значимый прирост всех агентов. Раздел 5.1.2 / 4.7.
- Evaluation: RPP (payoff Nash-решения между двумя лигами), Elo относительно Elite-бота (=0), round-robin по половине сохранённых моделей, 100 матчей на пару (около 4 млн тестовых матчей). Раздел 5.1.2-5.1.3.

### A.2.2 ROA-Star (Tencent, NeurIPS 2023)
URL: https://papers.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf
- Ресурсы на одного агента лиги: 64 V100, 4600 CPU-ядер, 2400 параллельных игр, learner около 11 000 шагов/сек (AlphaStar: 256 TPU-ядер, 4100 preemptible CPU, 16 000 игр, 50 000 шагов/сек). Table 1.
- 50 дней обучения; всего 768 моделей (221 MA, 113 ME, 216+218 для двух LE). Снапшот MA каждые 2e8 шагов (всего 4.42e10 шагов MA), ME сбрасывается максимум через 4e8 шагов, LE максимум через 2e8 шагов. Appendix A.6.
- Идея: goal-conditioned exploiters (EIE: условие на z с высоким win-rate у MA, сброс на последнюю MA; ERE: условие на недоисследованные z, сброс на supervised) и opponent modeling. Причина: в реимплементации AlphaStar эксплойтеры со временем "теряют способность" находить слабости MA/лиги. Разделы 1, 3.1.
- Для обычной (не StarCraft) задачи применима только общая идея: exploiter должен стартовать не всегда с нуля, а иметь "наследование" (TStarBot-X AEE, ROA-Star EIE) - иначе при сильном MA он не догоняет.

### A.2.3 SCC (StarCraft Commander, ICML 2021)
URL: https://arxiv.org/pdf/2012.13169
- Лига в стиле AlphaStar (main, main exploiter, league exploiter), но с несколькими main agents (три; два от supervised модели со сдвигом старта на 15 дней, один через agent branching). Раздел 6.2.
- Agent branching: новый агент инициализируется не от supervised, а от текущего main и учится по win/loss + плотной награде по статистике z. Решает проблему, что "чем сильнее лига, тем дольше учить exploiter с нуля". Раздел 6.2.
- Эффект: main agent 2, стартовавший на 15 дней позже в уже существующей лиге, догнал main 1 по Elo примерно за 15 дней. Раздел 6.3.
- Масштаб: на агента 1000 параллельных сред (AlphaStar 16 000), около 800 agent steps/сек (AlphaStar 50 000); Elo приведена к supervised агенту = 0, финальные main > 1500. PPO c асинхронным сэмплингом. Приложение B.
- Системная часть: MySQL для информации о лиге, общий кластер Predictors (GPU и CPU) для inference всех агентов лиги, Scheduler, Evaluator для win rates. Приложение B. Для Colosseum важно: общий inference-пул для замороженных оппонентов, не отдельный процесс на каждый.

### A.2.4 TLeague (Tencent, 2020)
URL: https://arxiv.org/pdf/2011.12895
- Фреймворк CSP-MARL (Competitive Self-Play MARL): Actor, Learner, InfServer, ModelPool, LeagueMgr + GameMgr (payoff-матрица, алгоритмы выбора оппонента) + HyperMgr (гиперпараметры, PBT). Раздел 3.2 упоминает GameMgr с разными алгоритмами выбора оппонента (SelfPlayGameMgr и др.); в 3.1 перечислены варианты Q: uniform, смесь current/historic, probabilistic Elo matching, функция от win-rate.
- FSP для K>=2 оппонентов: опонент выбирается независимо для каждого слота, φ ~ Q(M), в начале эпизода. Раздел 3.1. Это прямой рецепт для N-игроковых матчей.
- Пример из статьи: Pommerman Team mode (2 vs 2), смесь 35% pure SP и 65% PFSP "like how the Main Agent samples in AlphaStar", обучено с нуля. Раздел 4.3.
- Тезис: SP страдает от policy forgetting на цикличных играх; FSP добавляет "centripetal force". Раздел 3.1.

### A.2.5 AlphaStar Unplugged (DeepMind, 2023) - только для протокола оценки
URL: https://arxiv.org/pdf/2308.03526
- Offline RL, лиги нет. Но полезный протокол оценки: фиксированный набор из 7 оппонентов (very_hard бот + 6 референсных агентов); метрики Elo (логистическая модель, 400 points, якорь very_hard=1000, рейтинг нового агента подгоняется минимизацией cross-entropy без изменения референсов) и robustness = 1 - min_q f(p, q) (то есть 1 - exploitability относительно референсного набора). Раздел 3.3, Appendix A.2.
- [рекомендация] Для Colosseum: держать фиксированный "gauntlet" замороженных референсов + scripted bots для абсолютной шкалы, независимо от меняющегося пула.

### A.2.6 Honor of Kings (Ye et al., "Towards Playing Full MOBA Games", NeurIPS 2020)
URL: https://arxiv.org/pdf/2011.12692
- Лиги нет: curriculum self-play learning (CSPL): фаза 1 - self-play на фиксированных составах (10-герой группы, сбалансированные 5v5 с win-rate близким к 50%), фаза 2 - multi-teacher policy distillation в одну студенческую модель, фаза 3 - обучение на случайных составах. Переход фаз по сходимости Elo. Раздел 3.3.
- Ресурсы: 320 GPU и 35 000 CPU = "one resource unit"; batch на GPU 8192; teacher-модели по 9M параметров. Раздел 4.
- Урок: балансировка матчей (около 50% win rate) считается практически полезной для self-play ([33, 15] в статье). Раздел 3.3.
- Для Colosseum: идея "распределённое обучение узких teacher-ов + distillation" потенциально полезна, если игра имеет много конфигураций (карты, наборы юнитов).

### A.2.7 OpenAI Five (Dota 2)
URL: https://arxiv.org/pdf/1912.06680 (Appendix N "Self-play", Table 2, Appendix G)
- 80% игр против последних параметров, 20% против прошлых версий. Цель: избегать strategy collapse, когда агент забывает играть против разнообразных оппонентов.
- Dynamic sampling: каждому прошлому оппоненту i=1..N присваивается quality score q_i; выбор по softmax: p_i ∝ exp(q_i). Каждые 10 итераций текущий агент добавляется в пул с q = max(существующих q). После каждой игры: если прошлый оппонент выиграл у текущего - обновления нет; если текущий выиграл: q_i <- q_i - eta/(N p_i), eta = 0.01. Эффект: быстрые улучшения -> старые оппоненты имеют низкие q и не играются; медленный прогресс -> широкий разброс. Appendix N, формула (14).
- Гиперпараметры (Table 2): past opponents 20%, "Past Opponents Learning Rate" 0.01; PPO clip 0.2, GAE lambda 0.95, entropy 0.01->0.001, lr 5e-5->5e-6, team spirit 0.3->0.8 (Rerun 0.3->1.0).
- Compute: 770 +- 50 PFlops/s-days за около 10 месяцев; max 1536 optimizer GPU; batch 2 949 120 timesteps; Rerun: 150 +- 5 PFlops/s-days, около 2 месяцев. Разделы 2, 4.
- Team spirit: r_i = (1 - tau) rho_i + tau * mean(rho); tau=0 каждый сам за себя, tau=1 полностью разделённый reward. Ранний низкий tau снижает дисперсию градиента. Appendix G, формула (12).
- Оценка: TrueSkill, разница около 8.3 между агентами = около 80% win rate. Раздел 4.

### A.2.8 FTW: Quake III Capture the Flag (Jaderberg et al., Science 2019)
URL: https://arxiv.org/pdf/1807.01281
- Популяция из P=30 одновременно обучающихся агентов (PBT). 1920 arena-процессов; игры 2v2 (N=4) на процедурных картах; обучение 2e9 шагов (около 450K игр), эпизод 5 минут = 4500 agent steps. Разделы 2.3, 5.4.
- Matchmaking: сначала равномерно выбирается агент pi_p, затем ещё три без возвращения по распределению P(pi | pi_p) ∝ N(P(pi_p beats pi) - 0.5; sigma), то есть нормальное распределение вероятности победы (по Elo), центрированное на одинаковом уровне. В тексте sigma показан как "16" (вероятно 1/6) [не подтверждено]. Разбиение на команды случайное. Раздел 5.4.1.
- PBT: периодически сэмплируется другой агент; если оценка P(победа) < 70% (по Elo из тренировочных матчей), проигравший копирует policy, внутренний reward и гиперпараметры победителя и мутирует (+-20% с вероятностью 5%; tau LSTM равномерно из [5,20)); burn-in 1K игр после мутации. Раздел 2.2.
- Elo для команд: аддитивная модель, P(blue wins) = 1 / (1 + 10^(-psi^T m / 400)), m_i = (число появлений агента i в blue) - (в red); подгон psi по максимуму правдоподобия. Раздел 5.1. Это готовый рецепт рейтинга N-vs-N.
- Внутренний reward по 13 игровым событиям эволюционирует PBT (замена hand-shaped reward). Разделы 2.2, 5.5.
- Практическое: все 30 агентов тренируются параллельно, так что матчи "arena" с trainable-агентами - ровно как в Colosseum arena-режиме; learner на агента; каждый траекторный кусок идёт learner-у соответствующей policy (Раздел 2.3).

### A.2.9 Hide & Seek (OpenAI, ICLR 2020)
URL: https://arxiv.org/pdf/1909.07528
- Чистый self-play без пула прошлых версий ("self-play acts as a natural curriculum as agents always play opponents of an appropriate level"). Team-based reward: hiders +1 если все скрыты, -1 если кто-то замечен; seekers противоположно (zero-sum между командами, разделённый внутри). Разделы 3, 4.
- PPO + GAE, batch 64 000 chunks по 10 timesteps, 380 млн эпизодов до самой поздней стадии; в статье упоминается полный запуск 31.7 млрд frames за 34 часа при самом большом batch. Разделы 5, B.5.
- Урок: для транзитивно-ориентированных, "эмерджентных" сред чистый SP достаточен; большие batch сильно ускоряют по wall-clock при мало влияющем sample efficiency.

### A.2.10 Bansal et al. "Emergent Complexity via Multi-Agent Competition" (ICLR 2018)
URL: https://arxiv.org/pdf/1710.03748
- Sampling прошлых оппонентов: opponent ~ Uniform(delta * v, v), v - номер последней доступной итерации оппонента, delta in [0,1]. delta=1.0 = только последний; delta=0 = равномерно по всей истории. Раздел 5.4.
- Результат (Table 1, Sumo): обучение против последнего оппонента - худшее; Humanoid: delta=0.5 лучший win-rate и меньший loss (E[Loss] 0.17 при delta=0.5 против 0.53 при delta=1.0); Ant: лучшей оказалась delta=0 (uniform по всей истории). Значит delta=0.5 - не универсальная константа, а эмпирический оптимум для одной задачи.
- Гиперпараметры PPO: Adam lr 1e-3, clip 0.2, gamma 0.995, lambda 0.95, 409 600 samples на итерацию, mini-batch 5120, 6 эпох SGD для MLP и 3 для LSTM, без entropy bonus; exploration reward аннулируется линейно за 500 итераций (1000 для kick-and-defend). Разделы 4.1, B.
- Overfitting к одному оппоненту: решение - ансамбль из нескольких одновременно обучаемых политик (пул из 3 политик, каждая играет против случайной другой). Раздел 5.5.2. Это обоснование arena-режима Colosseum.
- Ранний dense exploration reward, затухающий в 0, критичен при sparse-соревновательной награде. Раздел 4.1, 5.3.

## A.3 PSRO и практичные варианты

### A.3.1 PSRO / DCH (Lanctot et al., NeurIPS 2017)
URL: https://arxiv.org/pdf/1711.00832
- PSRO: на каждой эпохе обучается "oracle" (approx best response) против mixture оппонентов из текущей популяции по meta-solver (Nash, uniform, regret matching, Hedge, projected replicator dynamics). Обобщает независимое RL (mixture = последняя политика) и fictitious play (uniform). Разделы 2-3.
- DCH (Deep Cognitive Hierarchy) - параллельное приближение PSRO (K уровней, уровень k против mixture уровней <k), память O(n^2 K^2). Раздел 3.
- Joint Policy Correlation (JPC): независимое RL в многоагентных играх переобучается на co-players; DCH существенно снижает JPC-потери (в тексте: снижение растёт с 28.7% до 56.7% по мере уровня/карты; точная привязка к картам не перепроверена). Раздел 4.1.

### A.3.2 Pipeline PSRO (McAleer et al., NeurIPS 2020)
URL: https://arxiv.org/pdf/2006.08555
- Иерархия worker-ов: "fixed" политики не обучаются, "active" обучаются параллельно; каждый active играет против meta-Nash fixed + active на уровнях ниже. Когда нижняя active перестаёт улучшаться (plateau по порогу за заданное время), она фиксируется и добавляется новая. Раздел 3.
- Теория: сходится к approx. Nash; DCH и Rectified PSRO не сходятся даже в малых играх. Эксперимент Barrage Stratego: средний выигрыш 71% против существующих ботов после 820 000 эпизодов, >65% против каждого. Разделы 1, 5.
- Для Colosseum: идея plateau-based freezing (фиксировать snapshot когда рост остановился, а не по таймеру) прямо реализуется поверх текущих LR/ELO метрик, но теоретические гарантии нужны только в 2p0s.

### A.3.3 NeuPL (Liu et al., ICLR 2022) и Simplex-NeuPL (ICML 2022), NeuPL-JPSRO (AAMAS 2024)
URLs: https://arxiv.org/pdf/2202.07415 ; https://arxiv.org/pdf/2205.15879 ; https://arxiv.org/pdf/2401.05133
- NeuPL: вся популяция - ОДНА conditional сеть Pi_theta(a | s, sigma_i); policy i условно обучается как best response на mixture sigma_i по своему meta-graph solver F (напр. PSRO-Nash: Sigma_{i+1,1:i} = SOLVE-NASH(U_{1:i,1:i})); все уникальные политики обучаются параллельно и постоянно (без truncation "когда пора остановить BR"). Алгоритм 1, раздел 1.
- Практика: интерактивный граф обновляется каждые 1000 градиентных шагов (epoch); максимальный размер популяции 8 уже достаточен (рост выше 8 даёт мало), эффективный размер популяции (UNIQUE(Sigma)) выходит на плато около 12 [как в тексте раздела 2.3; цифры 8 и 12 относятся к разным экспериментам, не противоречие - не перепроверено]. Если игра транзитивна, NeuPL деградирует к self-play. Разделы 2.3-2.4.
- Simplex-NeuPL: та же сеть учится best-response к любой смеси базисных политик (mixture сэмплируется из симметричного Дирихле), даёт Bayes-optimal поведение против смесей; полезно как auxiliary task для exploration. Abstract/раздел 3.
- NeuPL-JPSRO: расширение на N-player general-sum, сходимость к CCE; валидация в OpenSpiel, деплой в MuJoCo-футбол и capture-the-flag. Abstract.
- Оценка для Colosseum: красивая идея (один набор весов для всей популяции, transfer между политиками), но переписывает Policy-интерфейс и требует conditional-сеть. Для 1-3 человек за 2-8 недель - [рекомендация] НЕ делать в первом релизе; вариант "условная политика" можно отложить до Tier 3.

### A.3.4 Fictitious Cross-Play (Xu et al., 2023) - смешанные кооперативно-соревновательные игры
URL: https://arxiv.org/pdf/2310.03354
- Показывают контрпример, где SP на mixed cooperative-competitive играх не сходится к глобальному NE с высокой вероятностью; PSRO расширяется на команды без потери сходимости, но требует обучения с нуля; предлагают Fictitious Cross-Play (FXP) для масштабирования. Abstract.
- [рекомендация] Для командных игр: пул должен содержать не только "команды-как-есть", но и cross-play с разными партнёрами.

### A.3.5 "Real World Games Look Like Spinning Tops" (Czarnecki et al., NeurIPS 2020) и "Open-ended learning in symmetric zero-sum games" (Balduzzi et al., ICML 2019)
URLs: https://arxiv.org/abs/2004.09468 ; https://arxiv.org/pdf/1901.08106
- Spinning top: стратегическое пространство реальных игр - транзитивная ось (сила) + нетранзитивная радиальная (циклы), причём популяция нужна, и размер популяции влияет на сходимость (по аннотации). Что нетранзитивность максимальна в середине шкалы силы и сужается у вершины - это моё прочтение названия/идеи, не проверено по тексту [не подтверждено].
- Согласуется с AlphaStar: main agents транзитивны между собой, нетранзитивность "живёт" в exploiters (Nature, Extended Data Fig. 8).

### A.3.6 Omidshafiei et al. "Navigating the Landscape of Multiplayer Games" (Nature Comms 2020), alpha-Rank
URLs: https://arxiv.org/pdf/2005.01642 ; https://arxiv.org/pdf/1903.01373
- Response graph по empirical payoff-таблице позволяет измерять транзитивность/циклы; AlphaStar League относится к играм с "сильной транзитивной компонентой", а спектральный анализ эмпирической payoff-таблицы лиги даёт кластеры: слабые ранние агенты, специализированные exploiters (красный кластер), LE и MA (MA сильнее LE). Fig. 7 в 2005.01642.
- alpha-Rank (1903.01373): ранжирование через стационарное распределение Markov-цепи эволюционной динамики на empirical game; применим к >2 игрокам и general-sum; полиномиален по числу профилей стратегий (но число профилей растёт как prod |S_k|).
- OpenSpiel имеет реализацию alpha_rank для N игроков (в т.ч. HPT-таблицы) и α-PSRO. URL: https://openspiel.readthedocs.io/en/latest/alpha_rank.html ; https://arxiv.org/pdf/1908.09453

### A.3.7 XLand (DeepMind, 2021): generational training + PBT
URL: https://arxiv.org/pdf/2107.12808
- Пять поколений; в каждом популяция обучается с PBT, лучший агент поколения становится teacher (distillation) и дополнительным co-player (по аналогии с AlphaStar League). Метрика: normalised percentiles (нормировка по Nash-значению известных политик); co-player множество растёт с поколениями. Раздел 5.3, 6.
- Конфиг: каждый агент 8 TPUv3, около 50 000 agent steps/сек; PBT: первые 5e8 шагов без эволюции, затем каждые 1e8 шагов проверка; пара eligible если у ребёнка не было эволюции 2.5e8 шагов и родитель не хуже по перцентилям 10/20/50. Разделы 6.1, A.7. Обучение 20 млрд шагов на агента в абляциях (Раздел 6.2).

### A.3.8 Delta-uniform, FSP, historical averaging - сведение
- delta-uniform (Bansal): Uniform(delta*v, v). delta -> 0 = FSP; delta=1 = SP. URL: https://arxiv.org/pdf/1710.03748
- FSP в deep RL реализуется Monte-Carlo сэмплированием оппонента из пула в начале эпизода (TLeague). URL: https://arxiv.org/pdf/2011.12895
- "Mixed-Opponent" как отдельный именованный метод в моих источниках не найден [не подтверждено]; под ним, видимо, понимается смесь "latest + past + scripted + exploiters" (так делают AlphaStar MA, OpenAI Five, Lux AI winner - см. ниже).
- Survey самих методов self-play: https://arxiv.org/pdf/2408.01072 (читал только заголовок/аннотацию, детали не использовались) [не подтверждено в деталях].

### A.3.9 Kaggle-пример: что реально выигрывает при малых ресурсах
- Lux AI Season 1 (2 игрока), 1-е место (Toad Brigade, Isaiah Pressman): IMPALA (реализация FAIR) + UPGO + TD(lambda), сеть ResNet 24 блока x 128 каналов, около 20M параметров; sparse reward +1/-1 в конце; shaped reward только первые 20M шагов; замороженная teacher-модель с KL-штрафом к ней "для стабильности"; curriculum 8 -> 16 -> 24 блоков (меньшие сети как teachers); обучение на персональном ПК (8 ядер/16 потоков, 2 GPU). Лига/пула прошлых версий в описании не указаны [не подтверждено]. URL: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
- Hungry Geese (4 игрока FFA), 1-е место DeNA (команда HandyRL): "simple distributed off-policy RL" со self-play против копий той же политики; reward = ранговая награда (см. B.2). Подробный writeup победителей мне недоступен [не подтверждено]. URL: https://github.com/DeNA/HandyRL ; https://zenn.dev/ktechb/articles/e2394bc27358c4 (сторонний разбор, победитель не раскрывается).

## A.4 Итоги A: что реально стоит усилий для команды 1-3 человека, 2-8 недель

Ориентиры по ресурсам из источников (для масштаба сравнения): AlphaStar 3072 TPU-ядра x 44 дня; TStarBot-X 144 GPU x 57 дней (около 1/30 AlphaStar); ROA-Star 64 GPU на агента x 50 дней; Pluribus 12 400 CPU-core hours (8 дней на 64-ядерном сервере) [https://noambrown.github.io/papers/19-Science-Superhuman.pdf]. Colosseum на 1-8 машин - на 3-4 порядка меньше. Следствия:

1. Что окупается почти всегда (дёшево, подтверждено абляциями):
   - Смесь "SP + PFSP по пулу" лучше чистого SP по забыванию (46% -> 71% min win-rate против прошлых) и лучше чистого PFSP/FSP по скорости (Elo 1540 vs 1273/1143). Рецепт: p(SP/latest) 35-50% + 50% PFSP по пулу. URL: Nature Fig. 3C,D.
   - Пул чекпоинтов с регулярными снапшотами и f_hard по win rate: уже есть в Colosseum (PFSPMatchmaker).
   - Разреженный/обучаемый к последней модели учитель (KL к замороженной модели): Lux AI winner. URL: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
   - Фиксированный gauntlet референсных оппонентов для абсолютной шкалы (Unplugged robustness = 1 - min win rate). URL: https://arxiv.org/pdf/2308.03526
2. Что окупается, но только если есть запас вычислений:
   - Main exploiter: по абляции +153 Elo (1540 -> 1693) и RPP 6% -> 35%. URL: Nature Fig. 3A,B. Дешёвая версия: один ME на каждый MA, стартует с BC/supervised чекпоинта, играет только против текущего MA. Снапшот при win rate >70% против MA или по таймауту.
   - League exploiter: ещё +131 Elo и RPP 35% -> 62% - но платит за это большим числом параллельных learner-процессов (в AlphaStar 2 LE на расу).
3. Что можно выкинуть/отложить:
   - 12 параллельных актор-learner копий и 44-дневные расписания: не нужны. TStarBot-X показывает, что буквальное масштабирование таймеров вниз ломает лигу.
   - NeuPL/Simplex-NeuPL, P2SRO, alpha-PSRO, DCH: теоретически красиво, но требуют нового интерфейса (conditional policy / иерархия worker-ов); при 2-8 неделях не стоят. Исключение - идея plateau-based freeze из P2SRO.
   - Elo -> TrueSkill/Nash averaging для рейтинга: нужна только в N-игроках (см. Часть B).
   - PBT гиперпараметров (FTW/XLand): полезно при очень дорогом обучении, но удваивает инфраструктуру (копирование весов + мутации). Отложить; можно взять только упрощённую версию (раз в K шагов копировать веса лучшего и мутировать lr/entropy).
4. Рекомендуемые минимальные версии лиги (в порядке усложнения) [рекомендация]:
   - L0 (1-2 дня работы, уже почти есть): FIFO-пул + latest_prob. Заменить латентную 0.5 на OpenAI-Five-подобные 0.8/0.2 в режиме "стабильный прогресс"; убедиться, что в пул кладутся снапшоты каждые K шагов (K = около 0.5-2% полного бюджета шагов; ориентир: AlphaStar MA 2e9 из около 1.9e11 [расчёт около 1%]; ROA-Star 2e8 из 4.42e10 [расчёт около 0.45%]).
   - L1 (около 2-4 дня): PFSP f_hard c p=2 (в смеси: 35% SP, 50% PFSP, 15% "forgotten/exploiter-branch"; если пусто - в SP). Пул не FIFO, а: все чекпоинты с прореживанием (например, держать последние N + логарифмическую сетку старых); win-rate матрица со скользящим окном. Quality-score альтернатива OpenAI Five (softmax по q_i, обновление q_i -= eta/(N p_i), eta=0.01) - проще, не требует матрицы win rate.
   - L2 (около 1 неделя): один ME на MA (ресет на BC-чекпоинт; снапшот >70% или timeout около 2x интервала MA-снапшота, т.к. в AlphaStar 4e9 против 2e9). При нехватке GPU - временно делить learner (time-sharing, как уже заложено в CLAUDE.md).
   - L3 (опционально): LE (PFSP по всей лиге, ресет с p=0.25), AEE-стиль наследования (TStarBot-X) вместо ресета в supervised, если exploiters перестают догонять.
   - Arena режим (FTW/Bansal-стиль, несколько trainable): оставить как есть; matchmaking внутри arena - по Elo-близости (FTW: normal over P(win) - 0.5), чтобы "учебный сигнал" не терялся. Ресурс: каждая extra trainable policy = один learner-процесс.
5. Метрики здоровья лиги (дёшево и информативно) [рекомендация]:
   - min win-rate против прошлых чекпоинтов (мера забывания; Nature Fig. 3D).
   - RPP или хотя бы Nash-вознаграждение между "лигой A" и "лигой B" - не нужно в первой версии; достаточно exploitability_proxy = 1 - min_q win(p, q) по gauntlet (Unplugged).
   - Спектральный/кластерный анализ payoff-таблицы (Omidshafiei) - диагностический ноутбук, не часть пайплайна.

---------------------------------------------------------------------------------------------------

# ЧАСТЬ B. N-игроковые FFA и командные игры

## B.1 Формирование матчей (сколько слотов "своих/чужих")

Факты из источников:
- AlphaStar (2 игрока): оппонент выбирается одним сэмплом из пула (A.1). URL: Nature.
- TLeague для N>=2 оппонентов: φ ~ Q(M) независимо для каждого слота в начале эпизода. URL: https://arxiv.org/pdf/2011.12895 (раздел 3.1).
- FTW (2v2): первый агент uniform из популяции, три остальных - без возвращения из N(P(win) - 0.5; sigma) вокруг "равного уровня"; команды случайные; эпизод 5 минут. URL: https://arxiv.org/pdf/1807.01281 (5.4.1).
- Bansal et al.: в Sumo/Kick-and-Defend пул из ансамбля, для каждой rollout выбирается случайная другая политика; для симметричных игр оппонентом может быть и та же политика. URL: https://arxiv.org/pdf/1710.03748 (5.5.2).
- OpenAI Five (5v5): обе команды управляются копиями одной политики; 80% latest vs latest, 20% vs past. URL: https://arxiv.org/pdf/1912.06680.
- Pluribus 5H+1AI и 1H+5AI: единственный ИИ против 5 людей (10 000 рук за 12 дней, 13 про) и один человек против 5 копий Pluribus (по 5000 рук Ferguson и Elias; копии "не знают друг друга" и не могут колюзировать). URL: https://noambrown.github.io/papers/19-Science-Superhuman.pdf.
- Diplomacy (Gray et al.): оценка в формате 1v6 - один агент типа A против шести агентов типа B; средний счёт идентичного агента = 1/7 = 14.3%. URL: https://arxiv.org/pdf/2010.02923 (раздел 4.2).
- Diplodocus (Bakhtin et al.): реальные турниры 1 бот + 6 людей, анонимно; 200 игр, 62 человека, 2 версии бота заняли 1-е и 3-е места по Elo. URL: https://arxiv.org/pdf/2210.05492.

[рекомендация] Для Colosseum (N слотов, K trainable в матче):
- "Solo" матч (1 learning slot): остальные N-1 слотов - независимые сэмплы из пула (как TLeague), 1 слот-learner; разумно также "1 vs (N-1) копий одного оппонента" (формат 1v6) для чистой оценки "эксплуатации" конкретного оппонента.
- "Self" матч: все слоты = текущая политика (все траектории собираются) - дешёвый SP с N-кратным сигналом; для FFA нужно проверить симметричность (разные seat'ы = одна политика, значит seat embedding / seat-agnostic obs).
- "Arena": N слотов заполняются trainable-агентами (повтор при нехватке): совпадает с FTW/PBT.
- Смешивание: доля слотов с latest vs past: начать с 1 learner-слот + остальные из пула по PFSP; или случайное число learner-слотов k ~ Binomial.

## B.2 Награды: zero-sum, rank-based, placement

- Hungry Geese (4 игрока FFA), HandyRL (победитель Kaggle): итог эпизода переводится в попарные сравнения: outcome_p = sum_{q != p} sign(reward_p - reward_q) / (N - 1); 1-е место +1.0, 2-е +0.33, 3-е -0.33, 4-е -1.0 (для N=4 и без ничьих; при равных reward пары дают 0). Это "rank -> pairwise -> mean" шкала, симметричная и zero-sum (sum outcomes = 0). URL: https://raw.githubusercontent.com/DeNA/HandyRL/master/handyrl/envs/kaggle/hungry_geese.py
- Эта форма даёт: (а) sum-zero, (б) одинаковую "цену" любого перехода на одно место, (в) совпадает с pairwise-извлечением, нужным для PFSP/Elo (B.3).
- Hide&Seek: team-based zero-sum +1/-1; фаза подготовки (первые 40% эпизода, reward = 0) с нулевой наградой; штраф -10 за выход за арену. URL: https://arxiv.org/pdf/1909.07528 (раздел 3).
- OpenAI Five: shaped rewards + team spirit (B.5), в конце win/loss; shaped сохраняют даже в конце, но их вес по времени затухает (rho <- rho * 0.6^(T / 10 min)). URL: https://arxiv.org/pdf/1912.06680 (Appendix G).
- Lux AI winner (2p): sparse +1/-1; shaping первые 20M шагов. URL: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021
- Diplomacy: Sum-of-Squares scoring score_i = C_i^2 / sum_j C_j^2 (C_i - число supply centers) использовалось для обучения/оценки; draw-size scoring (равное деление) - альтернатива. Это пример не-zero-sum "общий выигрыш". URL: https://arxiv.org/pdf/2210.05492 (Appendix, "Diplomacy rules").
- Neural MMO v1.3: награда = выживание: -1 в момент смерти, иначе 0, gamma=0.95; роли policy делят веса внутри популяции. Награда за "место" не используется. URL: https://arxiv.org/pdf/2001.12004
- Hungry Geese (студенческий DQN-разбор): dense reward по длине гуся/манхэттену не дал улучшения, "converging was difficult due to random geese initialization"; предупреждение, что shaping по расстоянию до еды может навредить. URL: https://arxiv.org/pdf/2109.01954 [слабый источник, статья-учебная].
- Pluribus: payoff = деньги, не zero-sum в строгом смысле при N>2; используется "по денежным выигрышам mbb/game с AIVAT" как метрика. URL: https://noambrown.github.io/papers/19-Science-Superhuman.pdf.

[рекомендация] Для Colosseum: BaseEnv возвращает по умолчанию final_ranks (+ ties) и фреймворк сам строит reward через "pairwise-mean" (HandyRL-формула) или линейную placement-схему (1, 0.33, -0.33, -1); а custom shaped reward - как опциональный dense сигнал с затуханием (OpenAI Five/Bansal).

## B.3 Как считать результат против конкретного оппонента в FFA

1. Pairwise-извлечение из ранга (rank -> парные исходы)
   - Для каждого матча для каждой пары слотов (i, j): исход 1 / 0.5 / 0 по сравнению места. Это использует HandyRL (outcome выше), Gray et al. (ratings из пар "i achieved better outcome than j" ), Diplodocus.
   - Gray et al. (Appendix B): рейтинговый вектор s подгоняется градиентом на L(s) = sum_{(i,j) in D} -log sigma(s_i - s_j) + lambda * |s|^2; авторы пишут, что такой подход дал "более правдоподобные оценки, чем Elo и TrueSkill". URL: https://arxiv.org/pdf/2010.02923 (Appendix B).
   - Caveat [рекомендация]: пары внутри одной игры не независимы; для доверительных интервалов делать bootstrap по играм, а не по парам.

2. Generalized Bradley-Terry / Plackett-Luce (по полному рангу)
   - Plackett-Luce: P(ранжирование sigma | силы s) = prod_{k=1..N} exp(s_{sigma(k)}) / sum_{l=k..N} exp(s_{sigma(l)}) (стандартное определение; не цитата из одной статьи). Weng-Lin Bayesian approximation реализует этот вариант в OpenSkill: default mu=25, sigma=25/3. URL: https://arxiv.org/html/2401.05451v1 (OpenSkill; формулы обновления в статье не приведены, смотрите библиотеку).
   - Diplodocus: "standard generalization of BayesElo (Coulom 2005) to multiple players (Hunter 2004)". Ожидаемая доля счёта игрока в 2-игроковой игре ∝ exp((r_i + b_{s(i)}) / c), c = 400 log10(e), b_s - преимущество/недостаток позиции (seat); r_i, b_s подгоняются MAP с weak prior N(0, около 350 Elo). Для Diplomacy позиция = держава (7 штук) - то есть seat bias встроен прямо в рейтинг. URL: https://arxiv.org/pdf/2210.05492 (Section 4, Appendix I).
   - TrueSkill поддерживает FFA и частичные ранги (Gaussian mu, sigma). URL: https://arxiv.org/pdf/2101.00400 (описание в сравнении Elo-MMR); сам алгоритм Herbrich 2007 [первоисточник мною не читался].
   - OpenAI Five: TrueSkill, разница около 8.3 = около 80% win rate. URL: https://arxiv.org/pdf/1912.06680.

3. Командный Elo (аддитивный)
   - FTW: P(blue wins) = 1 / (1 + 10^{-psi^T m / 400}), m_i = (число раз, что агент i в blue) - (в red), подгон psi по максимуму правдоподобия; для PBT win prob агента i против j = m_i = 2, m_j = -2. URL: https://arxiv.org/pdf/1807.01281 (5.1). Работает только для транзитивных составов команд и взаимодействий (так же отмечают Marris et al.). URL: https://arxiv.org/pdf/2210.02205

4. Агрегация "кто против кого"
   - Для PFSP нужны x = P(A превосходит B) по парам: в N-игровом матче это pairwise outcome между слотом A и слотом B в тех матчах, где они оба участвовали. [рекомендация] Хранить WinRateTracker по "парам (agent A, agent B)" и обновлять его (i,j)-парами из каждого матча, с весом 1 / (число пар, содержащих A), чтобы большие N не раздували вес матча.
   - Дополнительно полезна "1vN-1 gauntlet": A против N-1 копий B (Gray 1v6) - прямая оценка "насколько A эксплуатирует B"; baseline = 1/N (в Diplomacy 14.3%).

5. Evaluation под неполной информацией/без Elo
   - Elo легко ломается клонами (дубликаты B в пуле меняют рейтинг A и C) и не умеет нетранзитивность; maximal lotteries/Nash averaging (VasE) инвариантны к клонам; на данных 7-player Diplomacy (webDiplomacy) предсказывают исходы лучше Elo. URL: https://arxiv.org/pdf/2312.03121
   - Для N-player general-sum: payoff rating с (C)CE (Marris et al. 2022); alpha-Rank (полиномиален по числу профилей, а число профилей растёт степенью N). URLs: https://arxiv.org/pdf/2210.02205 ; https://arxiv.org/pdf/1903.01373
   - Для 1-3 человек: Elo/BT на парах + win-rate матрица + gauntlet; maximal lotteries/alpha-Rank - опциональный диагностический скрипт.

6. Метрики для N-игроков:
   - Exploitability обобщается как e(pi) = (1/N) sum_i [ max_{pi*_i} v_i(pi*_i, pi_{-i}) - v_i(pi) ] (Gray et al.). Практически - best-response по тренируемому exploiter. URL: https://arxiv.org/pdf/2010.02923 (раздел 4.3).
   - Variance reduction: Pluribus использует AIVAT (для покера/случайных исходов) и 95% доверительный уровень (one-tailed t-test). URL: https://noambrown.github.io/papers/19-Science-Superhuman.pdf. Для детерминированных сред - много матчей + CI (Wilson для бинарных исходов, bootstrap по играм для рангов) [рекомендация].

## B.4 Seat/position bias и как его убирать

Факты:
- Pluribus: расположение игроков за столом (player order) определяется случайно в начале каждого дня; также используется "Control" - для оценки "luck" каждую руку переигрывают с копией Pluribus на месте человека. URL: https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf (разделы "Experimental setup", "AIVAT").
- Diplodocus: Elo-модель оценивает seat/power bias (b_{s(i)}); авторы специально указывают, что BayesElo корректирует и "силу соперников", и "какую из семи держав получил игрок, так как стартовые позиции неравны". URL: https://arxiv.org/pdf/2210.05492 (Section 4).
- Gray et al.: бот случайно назначается на одну из 7 держав; для контроля приводят также счёт, когда все 7 держав взвешены поровну: 26.9% +- 3.3% (вместо сырых). URL: https://arxiv.org/pdf/2010.02923 (сноска 6).
- FTW: агенты в игре случайно распределяются по красной/синей командам, карты случайно вращаются (чтобы агенты не эксплуатировали skybox). URL: https://arxiv.org/pdf/1807.01281 (5.2, 5.4.1).

[рекомендация] Для Colosseum:
- Обучение: случайная перестановка seat'ов в каждом эпизоде (slot_agent_map перемешивается независимо от room-индекса). Наблюдение содержит "относительный seat" (ego-центричное), если игра симметрична.
- Eval: полный цикл ротации. Для N игроков и 2 агентов - все C(N, k) назначения; для M агентов - Latin-square / сбалансированные блоки, где каждый агент занимает каждый seat равное число раз. Round-robin (уже в eval.py) нужно проверить на балансировку по seat'ам.
- Если игра асимметрична по seat'ам (как Diplomacy powers или Halite starting corners): ввести b_seat в рейтинговую модель (BayesElo/BT) вместо или вместе с ротацией; логировать winrate по seat'ам как диагностику.
- Симметризация данных: если среда зеркальна, зеркалить наблюдения/действия для augmentation (не из источников; общий приём).

## B.5 Self-play в N-игровых играх; командные игры; credit assignment

### B.5.1 Свидетельства: SP работает / не работает вне 2p0s
- Pluribus (6-max покер): блюпринт через MCCFR self-play (Linear CFR, pruning негативных регретов в 95% итераций), 8 дней на 64-ядерном сервере, 12 400 CPU core hours, <512 GB RAM, около $144 по spot-ценам; real-time search с k=4 continuation strategies. Теоретических гарантий вне 2p0s нет, но на практике SP работает "reasonably well" (формулировка статьи); обыгрывает профи: 48 mbb/game (stderr 25) в 5H+1AI и 32 mbb/game (stderr 15) в 1H+5AI. URL: https://noambrown.github.io/papers/19-Science-Superhuman.pdf (+ Supp: https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf).
  Вывод: SP успешен и в highly adversarial N-игроковом покере (так же формулирует Diplodocus: https://arxiv.org/pdf/2210.05492); проблемы начинаются там, где нужна кооперация/конвенции (Diplomacy).
- Diplomacy (7 игроков, смешанные кооперативно-соревновательные мотивы): SP с нуля (DORA) даёт сильного агента против копий самого себя и слабого против 6 human-like; независимые запуски DORA сходятся в разные равновесия и проигрывают друг другу ("outside agents cannot compete, even agents trained independently by the same method"). Вывод авторов: self-play from scratch может быть недостаточен в 7p no-press Diplomacy, "unlike 6-player poker". URL: https://arxiv.org/pdf/2110.02924 (раздел 5-6).
- Решения: SearchBot (supervised blueprint + one-step regret-minimization search, ранг 17 из 901 по Ghost-Rating, top около 2%); DiL-piKL / RL-DiL-piKL (Diplodocus): регуляризация к human-imitation policy по KL, 200 игр, 62 человека, 1-е и 3-е места по Elo. URLs: https://arxiv.org/pdf/2010.02923 ; https://arxiv.org/pdf/2210.05492
- Cicero (Meta, Science 2022; Diplomacy с естественным языком): 40 игр с 82 анонимными людьми, средний счёт 25.8% (>2x средний 12.4%), топ 10% среди игравших более одной игры; 2-е место по рейтингу среди игроков с 5+ игр. URL (вторичный источник, статья Science недоступна для загрузки): https://www.mit.edu/~gfarina/2022/cicero ; https://ai.meta.com/research/cicero/diplomacy/ [частично не подтверждено, смотрите Science: https://www.science.org/doi/10.1126/science.ade9097].
- Hungry Geese (4 игрока, ранговый reward): чистое self-play против копий той же политики с off-policy RL (HandyRL) выиграло Kaggle. URL: https://github.com/DeNA/HandyRL ; https://zenn.dev/ktechb/articles/e2394bc27358c4 (подробности награды из репозитория HandyRL, см. B.2).
- Pommerman (TLeague): 2v2 Team mode от нуля, смесь 35% SP и 65% PFSP, победа над Simple Agent и Navocado (лучший learning-based на NeurIPS 2018). URL: https://arxiv.org/pdf/2011.12895 (4.3).
- Halite (4 игрока), Kore: данные о победителях/подходах (RL vs rules) найти не удалось [не подтверждено].
- Neural MMO: v1.0/1.3: до 128 агентов на карте, популяции независимых политик (обучение сэмплирует политики из 8 популяций), tournament-style evaluation - склеивание популяций из разных экспериментов на один сервер и сравнение по среднему времени жизни; обученные в больших популяциях агенты стабильно побеждают обученных в малых (small populations learn brittle policies). Neural MMO 2.0: 128 агентов, 3x быстрее, task system, оценка на unseen tasks/maps/opponents; соревнование NeurIPS 2023. URLs: https://arxiv.org/pdf/1903.00784 ; https://arxiv.org/pdf/2001.12004 ; https://arxiv.org/abs/2311.03736

### B.5.2 Не-транзитивность в N-игроках
- AlphaStar: основная нетранзитивность сосредоточена в exploiters (около 3 млн RPS-циклов с exploiters против около 200 между main). URL: Nature Extended Data Fig. 8.
- Spinning tops: две оси (транзитивная сила + нетранзитивные циклы), популяция нужна; детали распределения циклов по шкале силы не проверялись [не подтверждено]. URL: https://arxiv.org/abs/2004.09468
- Omidshafiei: response-graph/spectral analysis - способ измерения нетранзитивности payoff-таблицы; для N-player вводятся профили стратегий (таблицы-тензоры). URL: https://arxiv.org/pdf/2005.01642
- N-игроков: независимые запуски дают несовместимые равновесия (DORA), то есть нетранзитивность проявляется как "несовместимость соглашений/конвенций". URL: https://arxiv.org/pdf/2110.02924
- FXP: SP не гарантирует глобальный NE в mixed cooperative-competitive играх. URL: https://arxiv.org/pdf/2310.03354
- [рекомендация] Для Colosseum: держать в пуле минимум 2-3 независимых запуска/архитектуры (а не только один lineage) и мерить cross-play матрицу между ними; расхождение >> 50% между независимыми seed'ами = признак нетранзитивных конвенций.

### B.5.3 Командные игры: credit assignment и team reward
- OpenAI Five: team spirit tau (формула в A.2.7, B.2) как интерполяция между индивидуальным и общим reward для снижения дисперсии; архитектура: пять реплик политики с cross-hero pool слоем (отдельного centralized critic в тексте Appendix G/H я не искал) [не подтверждено]. URL: https://arxiv.org/pdf/1912.06680 (Appendix G, H).
- FTW: внутренние (эволюционируемые PBT) награды по 13 событиям; вклад каждого игрока в команду моделируется аддитивным Elo. URL: https://arxiv.org/pdf/1807.01281
- Hide&Seek: общий командный reward (hiders: +1 если все скрыты). URL: https://arxiv.org/pdf/1909.07528
- Pommerman в TLeague: две политики одной сети, централизованная value-функция по двум LSTM-embedding'ам (centralized critic) + decentralized actors; обучено PPO. URL: https://arxiv.org/pdf/2011.12895 (4.3).
- Neural MMO: реварды "на агента", кооперация/конкуренция возникает как побочный эффект. URL: https://arxiv.org/pdf/2001.12004
- [рекомендация] Для Colosseum: добавить в config `team_spirit` (интерполяция reward внутри команды) и опциональный centralized critic (value получает embeddings всей команды) - не более двух простых опций; QMIX/MADDPG и т.п. отложить (CLAUDE.md и так ставит APPO первым).

## B.6 Практические выводы по Части B для Colosseum (N слотов)

1. Reward: по умолчанию "pairwise-mean rank reward" (HandyRL) + опциональный shaped dense reward с затуханием (Bansal/OpenAI Five). Для командных игр - team_spirit.
2. Матч: независимое сэмплирование оппонентов на слот (TLeague), случайная перестановка seat'ов, K learning-слотов (solo K=1, self K=N, arena K=trainable).
3. Результат: из каждого матча извлекать парные исходы (i,j), питать PFSP win-rate матрицу и Elo/BT; хранить полные ранги для Plackett-Luce / OpenSkill как "медленный" рейтинг; CI через bootstrap по играм.
4. Seat bias: ротация в eval + (при асимметрии) параметр b_seat в рейтинге (Diplodocus).
5. Non-transitivity: держать gauntlet из замороженных и scripted оппонентов + несколько независимых lineage, cross-play матрица.
6. Чего не делать: alpha-Rank/CCE-рейтинг как основной рейтинг (дорого при N>2, нужен только в анализе), NeuPL, PSRO с LP-Nash в N-игроках (Nash не уникален, вычислительно тяжёл).

---------------------------------------------------------------------------------------------------

# ПРИЛОЖЕНИЕ. Что не удалось подтвердить / ограничения

- Точное значение p в f_hard у AlphaStar: в статье указано только "p in R+". Значение (1-x)^2 взято из реимплементации DI-engine. [не подтверждено для оригинала]
- Показатели степени 2e9/4e9 в PDF Nature извлеклись некорректно ("10s"), но подтверждены TStarBot-X (раздел 4.5) и DI-engine.
- sigma в FTW matchmaking: в извлечённом тексте "16"; вероятно 1/6. [не подтверждено]
- "Mixed-Opponent" как отдельный метод не найден.
- Cicero (Science 2022): страница Science вернула 403, цифры (40 игр, 82 игрока, 25.8% vs 12.4%, топ 10%) взяты из вторичных источников.
- Halite/Kore: данных о решениях-победителях не найдено.
- Hungry Geese: кроме формулы outcome() в HandyRL и общего описания (self-play, off-policy), деталей pipeline победителя нет. Lux AI S1 winner: нет описания пула оппонентов.
- TrueSkill (Herbrich 2007) и Hunter (2004, MM для generalized BT) первоисточники сами не читал; использованы описания в Diplodocus/Gray/OpenSkill.
- Neural MMO: подробные формулы оценки по "tournament" не выписаны; для 2.0 использована только аннотация.
- Self-play survey (https://arxiv.org/pdf/2408.01072) и PufferLib/Cleanba - не разбирались в деталях.

# Список основных первоисточников (быстрый индекс)
- AlphaStar: https://www.nature.com/articles/s41586-019-1724-z ; https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf
- TStarBot-X: https://arxiv.org/pdf/2011.13729 ; ROA-Star: https://papers.neurips.cc/paper_files/paper/2023/file/94796017d01c5a171bdac520c199d9ed-Paper-Conference.pdf ; SCC: https://arxiv.org/pdf/2012.13169 ; TLeague: https://arxiv.org/pdf/2011.12895 ; Unplugged: https://arxiv.org/pdf/2308.03526 ; HoK: https://arxiv.org/pdf/2011.12692
- OpenAI Five: https://arxiv.org/pdf/1912.06680 ; FTW: https://arxiv.org/pdf/1807.01281 ; Hide&Seek: https://arxiv.org/pdf/1909.07528 ; Bansal: https://arxiv.org/pdf/1710.03748 ; XLand: https://arxiv.org/pdf/2107.12808
- PSRO: https://arxiv.org/pdf/1711.00832 ; P2SRO: https://arxiv.org/pdf/2006.08555 ; NeuPL: https://arxiv.org/pdf/2202.07415 ; Simplex-NeuPL: https://arxiv.org/pdf/2205.15879 ; NeuPL-JPSRO: https://arxiv.org/pdf/2401.05133 ; FXP: https://arxiv.org/pdf/2310.03354
- alpha-Rank: https://arxiv.org/pdf/1903.01373 ; Navigating multiplayer: https://arxiv.org/pdf/2005.01642 ; Spinning tops: https://arxiv.org/abs/2004.09468 ; Open-ended learning: https://arxiv.org/pdf/1901.08106 ; Social choice evaluation: https://arxiv.org/pdf/2312.03121 ; Marris payoff rating: https://arxiv.org/pdf/2210.02205 ; OpenSpiel: https://arxiv.org/pdf/1908.09453
- Pluribus: https://noambrown.github.io/papers/19-Science-Superhuman.pdf (+Supp) ; Diplomacy: https://arxiv.org/pdf/2010.02923 , https://arxiv.org/pdf/2110.02924 , https://arxiv.org/pdf/2210.05492
- Neural MMO: https://arxiv.org/pdf/1903.00784 , https://arxiv.org/pdf/2001.12004 , https://arxiv.org/abs/2311.03736
- HandyRL: https://github.com/DeNA/HandyRL ; Lux AI S1 winner: https://github.com/IsaiahPressman/Kaggle_Lux_AI_2021 ; DI-engine league: https://github.com/opendilab/DI-engine/tree/main/ding/league

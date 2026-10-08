# Исследование для SP2 (GameSpec / MultiAgentEnv): реальные соревнования и существующие multi-agent API

Дата: 2026-10-08. Метод: исходники и README из GitHub (скачаны напрямую), spec-файлы `kaggle-environments`, статьи на arXiv, посты на Kaggle/CodinGame (многие страницы Kaggle и CodinGame рендерятся JS и недоступны для чтения; такие места помечены). Все утверждения снабжены URL; непроверенное помечено [не подтверждено].

---

## Часть A. Реальные соревнования

### A.1 Сводная таблица

Обозначения: sim = одновременные ходы, seq = по очереди, FFA = каждый за себя. «Юниты» = сколько единиц контролирует один бот.

| Соревнование | Места / команды | Ход | Юниты у одного бота | Birth/death | Observation | Action | Elimination | Исход | Длина / лимиты | Симулятор |
|---|---|---|---|---|---|---|---|---|---|---|
| **Lux AI S3** (2024–25) | 2 игрока, 2 команды | sim | до `max_units=16` | да: юниты уничтожаются и респавнятся, ID переиспользуются | fog of war, тензоры с масками: `units (T,N,3)`, `units_mask`, `sensor_mask`, карта 24x24 | `(16,3)` на команду: тип действия 0..5 + dx, dy для sap | нет | **серия best-of-5** матчей, победитель серии = больше выигранных матчей; в матче очки реликвий, тай-брейк по энергии, потом случайно | 5 матчей x 100 шагов (505 шагов эпизода); `actTimeout` 3 с | JAX (`Lux-Design-S3`), Python/JS kit |
| **Lux AI S2** (2022–23) | 2 игрока | sim (+ фаза bid/placement) | не ограничено явно (десятки роботов + фабрики) | да: роботы строятся и гибнут | полный словарь, fog в specs не упомянут | per-unit **очередь действий** (до 20) + действия фабрик | проигрывает при потере всех фабрик | win/loss/draw по lichen, «потеря всех фабрик» = поражение | 1000 шагов; 9 с на ход + 60 с резерв | Python (`luxai_s2`), JAX-версия `juxai_s2` |
| **Orbit Wars** (Kaggle 2026) | **2 или 4** игрока FFA | sim | список планет игрока (до 40 планет всего) и неограниченное число флотов | да: флоты создаются и гибнут, кометы появляются и исчезают | **entity-list**: `planets`, `fleets`, `comets` (списки списков) | **список** ходов переменной длины `[from_planet, angle, ships]` | да: игра кончается, когда остался один (или ноль) игрок с планетами или флотами | score = число кораблей, побеждает наибольший; на Kaggle рейтинг Elo | 500 шагов; `actTimeout` 1 с | Python (`kaggle_environments`) |
| **Kore 2022** | env поддерживает 1, 2 и 4 игрока (в соревновании [не подтверждено]) | sim | много флотов + верфи | да | словари/списки по `uid` | dict `uid -> действие` | **да**, reward = `step_eliminated - episode_steps - 1` | kore игрока | 400 шагов; 3 с | Python |
| **Halite IV** (2020) | 1, 2, 4 игрока | sim | много кораблей + верфей | да (SPAWN/CONVERT) | словари `uid -> позиция` | dict `uid -> {CONVERT,SPAWN,N,S,E,W}` | **да**, тот же отрицательный reward | halite игрока / ранг | 400 шагов; 3 с | Python |
| **Hungry Geese** (2021) | в env 1..8; соревнование 4 [по памяти, не подтверждено] | sim | 1 гусь | гусь гибнет | списки позиций | одно из 4 направлений | **да**, гуси гибнут в середине | reward = `steps_survived * (max_len+1) + length` (ранг) | 200 шагов; 1 с | Python |
| **Lux AI 2021** (S1) | 2 | sim | много юнитов + города | да | fog | per-unit строки | потеря городов | win/loss | 360 шагов [не подтверждено] | Python/JS |
| **Google Research Football** (Kaggle 2020) | 2 команды по 11 | sim | в соревновании бот управляет **1** игроком (`controlled_players` в env до 11) | нет | `players_raw` (список по контролируемым игрокам) | список действий 0..19 на игрока | нет | счёт (`+1/-1` за гол) | 3002 шага; 0.5 с | C++ ядро, EnvPool-версия |
| **Neural MMO 2022 (Team Battle)** | 16 команд x 8 агентов, карта 128x128 | sim | каждый агент отдельный (команды из 8 агентов) | да: агенты гибнут | структурные таблицы `Tile`, `Entity`, `Inventory`... фиксированного размера + маски | `Dict` по системам: Move, Attack (Target = индекс в таблице Entity), Use... | да, агент гибнет, последняя команда выигрывает | last team standing | 1024 тика | Python |
| **Neural MMO 2023 (Multi-task)** | 128 агентов, оценка группами по 14 | sim | агент = 1 юнит | да | то же | то же | да | число выполненных tasks (1298 train / 63 eval) | 1024 тика | Python |
| **Pommerman** (NIPS 2018) | 4 агента, режимы FFA / Team (2v2) / TeamRadio | sim | 1 | агенты гибнут | доска 11x11, в Team частичная (view 9x9) | 6 дискретных действий | **да** | +1 победа / -1 остальные; FFA по шагам: 0 живой, -1 мёртвый | 800 шагов | Python |
| **MicroRTS** | 2 | sim (RTS, real-time) | все юниты (десятки) | да | 29 бинарных плоскостей h x w | MultiDiscrete **на каждую клетку** (GridNet) + маска source unit | нет (но можно потерять всех) | +1/0/-1 | зависит от карты | Java; Gym-µRTS deprecated с 2025-08 |
| **Generals.io** | 1v1, FFA N, 2v2 | sim | все клетки игрока | клетки меняют владельца | `(H,W)` плоскости, fog | `H x W x 9` (pass/клетка/направление/половина) | **да** (захват general) | +1 победившей команде / -1 остальным | сотни ходов | JAX (`generals-bots`), 10M+ steps/s |
| **CodinGame Winter 2026 (SnakeByte)** | 2 игрока | sim | команда змей (до 4 по форуму [не подтверждено]) | змеи гибнут (падение) | текстовый протокол stdin | текстовая команда на каждую змею | змеи, не игроки | по числу частей тела, часто ничьи | ~155 ходов в среднем (локальная арена) | C++ / Go (`codingame-arena`) |
| **Battlecode** (2025/26) | 2 команды | sim по раундам, **у каждого робота свой код** | каждый юнит выполняет копию кода | да | локальные API-вызовы Java | вызовы методов робота | да | серия матчей, турнир | лимит bytecode на робота | Java |
| **Kaggriculture** (Kaggle 2026) | 2 | sim | фермер + нанятые помощники (`hands`) | да (найм) | `farms` (shared), `private`, `market` | dict `{farmer, hands[], market[]}` | банкротство не указано | деньги в конце | 720 шагов | Python |
| **Pokémon TCG AI Battle** (Kaggle, env `cabt`) | 2 | seq | карты | нет | скрытая информация | **индекс опции из списка** | нет | `-1/0/1` | до 10^7 шагов (лимит из JSON) | Python |

Источники по строкам:
- Lux S3: правила https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/docs/specs.md (best of 5, fog of war, респавн, ID 0..max_units-1 рециклируются, тай-брейк по энергии), параметры `max_units: int = 16`, `max_steps_in_match = 100`, `match_count_per_episode = 5` в https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/src/luxai_s3/params.py, `EnvObs` с `units_mask`, `sensor_mask` в https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/src/luxai_s3/state.py, `action_space` `(max_units, 3)` и `reward[f"player_{k}"] = state.team_wins[k]` в https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/src/luxai_s3/env.py, `actTimeout: 3`, `episodeSteps: 506` в https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/lux_ai_s3/lux_ai_s3.json.
- Lux S2: https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/specs.md (очередь действий до 20; «winner ... most lichen value»; «If any team loses all of their factories, they automatically lose»; 9 с на ход + 60 с пул; фаза bid и placement), `max_episode_length = 1000`, `UNIT_ACTION_QUEUE_SIZE = 20`, `MIN_FACTORIES=4`, `MAX_FACTORIES=10` в https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/luxai_s2/luxai_s2/config.py. Наличие JAX-версии (`juxai_s2`) видно в листинге https://github.com/Lux-AI-Challenge/Lux-Design-S2. Явного лимита числа роботов в config не найдено.
- Orbit Wars: https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/orbit_wars/README.md и `orbit_wars.json` рядом (`"agents": [2, 4]`, `episodeSteps: 500`, `actTimeout: 1`, формат action «List of moves: [from_planet_id, direction_angle, num_ships]»); «The game ends when ... Elimination: Only one player (or zero) remains with any planets or fleets»; 20–40 планет; соревнование и участие (4 729 команд) https://www.kaggle.com/competitions/orbit-wars (через результаты поиска).
- Kore/Halite: `kore_fleets.json` https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kore_fleets/kore_fleets.json и `halite.json` https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/halite/halite.json: `"reward": "...if the player has not been eliminated, else step_eliminated - episode_steps - 1"`. Что Kore 2022 играли именно 2 игрока, подтвердить не удалось.
- Hungry Geese: https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/hungry_geese/hungry_geese.json (`agents [1..8]`, reward «steps survived * (max goose length + 1) + current goose length», `episodeSteps 200`, `actTimeout 1`, доска 11x7).
- GRF: https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/football/football.json (`controlled_players` 0..11, `players_raw`, `episodeSteps 3002`, `actTimeout 0.5`, reward `+1/-1` за гол); API `number_of_left_players_agent_controls` и 19 действий https://github.com/google-research/football (репозиторий архивирован 2026-08-19); «в соревновании можно управлять только одним игроком (мяч у атаки, ближайший защитник в защите)» — из результатов поиска со ссылкой на https://arxiv.org/pdf/2302.07515 (TiZero), не первичный источник.
- NMMO: `Env(ParallelEnv)`, структурные obs, `ActionTargets` в https://github.com/NeuralMMO/neural-mmo/blob/master/nmmo/core/env.py; Team Battle = 16 команд по 8, карта 128x128, 1024 тика, Multi-task = 128 агентов, 1298 train tasks / 63 eval, Sandwich/King of the Hill https://arxiv.org/abs/2406.05071 (текст PDF); итоги 2023 (оценка группами по 14 агентов из 128, 256 held-out карт) https://arxiv.org/abs/2508.12524 (через сводку, [частично подтверждено]); 2022: «16 populations» https://arxiv.org/abs/2311.03707.
- Pommerman: `MAX_STEPS = 800`, `BOARD_SIZE = 11`, `AGENT_VIEW_SIZE = 4` https://github.com/MultiAgentLearning/playground/blob/master/pommerman/constants.py; формулы reward (+1/-1, `[int(is_alive) - 1]` в FFA, команды `[0,2]` vs `[1,3]`) https://github.com/MultiAgentLearning/playground/blob/master/pommerman/forward_model.py; режимы FFA/Team/TeamRadio https://github.com/MultiAgentLearning/playground.
- MicroRTS: https://github.com/Farama-Foundation/MicroRTS-Py («(h, w, 29)», MultiDiscrete на клетку, deprecated 2025-08-11); GridNet, «all units must be controlled simultaneously», invalid action masking, UAS vs GridNet https://arxiv.org/abs/2105.13807.
- Generals: https://github.com/strakam/generals-bots («same env plays 1v1, N-player free-for-all, and team games», «Capturing a general ... the victim is eliminated», «every player on that team gets reward +1, everyone else -1», shared sight); `H x W x 9` и U-Net https://arxiv.org/abs/2507.06825.
- CodinGame Winter 2026: форум-фидбек https://forum.codingame.com/t/winter-challenge-2026-feedbacks-and-strategies/208044 (несколько змей у игрока, одновременные ходы, топ — Smitsimax/MCTS, один PPO в топ-7; числа получены через сводку, [частично подтверждено]); локальная арена blue vs red, `avg_turns=155`, доля ничьих 39% https://github.com/mrsombre/codingame-arena. Страницы самих контестов не читаются (JS), правила Fall 2025 и Summer 2025 не найдены.
- Battlecode: «Each robot runs a copy of your code independently», лимит bytecode, 2026 «Uneasy Alliances» — результаты поиска по https://battlecode.org/about, детали [не подтверждено].
- Kaggriculture: https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kaggriculture/kaggriculture.json и https://www.kaggle.com/competitions/kaggriculture.
- `cabt`: https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/cabt/cabt.json (action «List of option index», reward `-1/0/1`); что это именно Pokémon TCG соревнование Kaggle (https://www.kaggle.com/competitions/pokemon-tcg-ai-battle) — предположение по названиям [не подтверждено].

Какие ещё симуляции Kaggle лежат в `kaggle-environments` (не все были соревнованиями; статус каждой [не подтверждено]): `crawl`, `kargo` (2 или 4 игрока), `pyxis`, `reinforce_tactics`, `werewolf` (6–15 игроков), `planet_wars`, `llm_20_questions`, `chess` — листинг https://github.com/Kaggle/kaggle-environments/tree/master/kaggle_environments/envs.

### A.2 Краткие заметки

1. **Количество мест**: почти всё 2 игрока; 4 игрока — Orbit Wars, Halite, Kore, Hungry Geese, Pommerman, Generals (FFA N), Kargo; больше 4 — Hungry Geese в env (до 8), NMMO (16 команд), Werewolf (до 15). Seat symmetry в Orbit Wars сделана зеркально (https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/orbit_wars/README.md).
2. **Одновременные ходы** — норма (Lux, Orbit Wars, Halite, Kore, Hungry Geese, Pommerman, Generals, CodinGame SnakeByte). Последовательные: карточные игры (`cabt`), PGX-игры. Смешанный режим: Lux S2 в стартовой фазе (поочерёдная расстановка фабрик и bid) и затем sim.
3. **Один бот — много юнитов** — доминирующий случай (Lux, Halite, Kore, Orbit Wars, µRTS, Generals). Три формы выходного действия: (а) фиксированный массив на `max_units` слотов с маской (Lux S3: `(16,3)`); (б) **список переменной длины** действий над сущностями (Orbit Wars: `[[from, angle, n], ...]`; Halite/Kore: dict `uid -> действие`); (в) тензор на клетку (µRTS GridNet `h*w*...`, Generals `H*W*9`). Battlecode — крайний случай: юниты вообще не имеют центрального контроллера.
4. **Переменное число юнитов** — везде. Решения: паддинг + маска (Lux S3 `units_mask`; NMMO `Entity (PLAYER_N_OBS, attrs)`), списки (Orbit Wars), тензоры на клетку (Generals).
5. **Исход** бывает пяти видов: win/loss/draw (cabt, Pommerman, Generals), ранг по времени выживания и длине (Hungry Geese), score (Orbit Wars, Kaggriculture, Halite), серия (Lux S3: `reward = team_wins`, суммарное число выигранных матчей), множество task-метрик (NMMO 2023). Итоговый рейтинг на Kaggle — непрерывный Elo (в Orbit Wars см. репозиторий участника https://github.com/jacklvd/orbit_war_2026).
6. **Elimination**: Halite/Kore штрафуют отрицательным reward по времени вылета; Generals, NMMO, Pommerman, Hungry Geese выбывают посреди игры, остальные продолжают.
7. **Асимметричные роли** в проверенных соревнованиях редки. Единственный структурно асимметричный случай в нашем списке — NMMO-вариации (Protect the King: лидер команды, смерть лидера выбивает команду: https://arxiv.org/abs/2406.05071) и GRF (атака/защита одним игроком). Классического «hunter vs prey» среди найденных нет [не подтверждено].
8. **Скорость симулятора**: JAX (Lux S2/S3, Generals до 50.7M steps/s на H200 по https://arxiv.org/abs/2507.06825 через сводку), C++ (ядро GRF; Pommerman на Python), Java (µRTS, Battlecode, CodinGame engine), Python (Orbit Wars, Halite, Kore, Hungry Geese). Orbit Wars победитель ~6.3M шагов на GPU-час на 200M модели: https://tufalabs.ai/research/orbit-wars/.

### A.3 Что использовали топовые RL-решения (коротко)

- **Orbit Wars, 1 место**: один трансформер 200M (38 блоков, 768-мерные токены), токены = флоты, планеты, кометы + 17 служебных токенов (сводка игрока x4, глобальный, plan-токены, value-токены); действие = Bernoulli на каждый source-планету + выбор цели attention-ом + размер флота смесью логистических распределений; 2p и 4p в одной сети; 15 млрд шагов self-play; reward +1 победителю, -1 остальным; число видимых флотов ограничено. https://tufalabs.ai/research/orbit-wars/
- **Orbit Wars, 2 и 13 места**: 2-е — ModernBERT поверх 1D-CNN эмбеддингов, 4.3M параметров, PPO в PufferLib, один и тот же net для 2p и 4p; 13-е — трансформер с токеном на планету, около 1.2M параметров, отдельные сети для 2p/4p, PPO + league. Первоисточники Kaggle (https://www.kaggle.com/competitions/orbit-wars/writeups/2nd-place-solution-for-orbit-wars, https://www.kaggle.com/competitions/orbit-wars/writeups/13th-place-solo-gold-solution) не читаются (JS), данные из результатов поиска [частично подтверждено].
- **Lux S3**: 1-е — IMPALA, ResNet + ConvLSTM + Transformer, 200M параметров, per-tile кодирование >1000 признаков, всё зеркалится в перспективу player-0, две головы (движение/sap + позиция sap 15x15), маска ослаблена намеренно, reward только win/loss; 2-е — PPO + ResNet, среда переписана на Rust; 10-е — PPO + PFSP (75% latest vs latest). https://zenn.dev/kurupical/articles/61dbeedf89a29d (сводка статьи).
- **Hungry Geese, 1 место**: HandyRL (policy gradient с off-policy коррекцией); сеть GeeseNet из тор-свёрток (`TorusConv2d`, 17 входных плоскостей). https://github.com/DeNA/HandyRL/blob/master/handyrl/envs/kaggle/hungry_geese.py ; 1 место подтверждено README https://github.com/DeNA/HandyRL.
- **GRF**: WeKick (1 место 2020): imitation + multi-head value + distributed league training; в соревновании один управляемый игрок. https://arxiv.org/pdf/2302.07515 (через поиск, [частично подтверждено]).
- **NMMO 2023**: baseline = энкодеры тайлов/агентов/задач/предметов/рынка + LSTM + pointer-декодер действий, IPPO с historical self-play в PufferLib; победитель «Takeru» менял в основном reward/конфиг. https://arxiv.org/abs/2406.05071 , https://arxiv.org/abs/2508.12524
- **Generals.io**: U-Net-торс, policy head `H x W x 9`, reward +1/-1 + potential-based shaping по логарифмам отношений земли/армии/замков, 3 500 FPS на 12 ядрах CPU. https://arxiv.org/abs/2507.06825
- **µRTS**: GridNet (conv, действие на каждую клетку) + invalid action masking, PPO. https://arxiv.org/abs/2105.13807
- **CodinGame SnakeByte 2026**: топ — Smitsimax (MCTS на каждую змею), beam search; один PPO-бот в топ-7. https://forum.codingame.com/t/winter-challenge-2026-feedbacks-and-strategies/208044
- **Lux S2, Halite IV, Kore 2022**: победители RL-решения не найдены в доступных источниках [не подтверждено].

---

## Часть B. Существующие multi-agent API

### B.0 Сводная таблица

| API | Acting set | Смерть / elimination | Reward вне хода | Команды | Разные spaces | Truncation | Переменное число агентов | Global state |
|---|---|---|---|---|---|---|---|---|
| PettingZoo Parallel | все `agents` | агент уходит из `agents` | в dict `rewards` | нет | `observation_space(agent)` | `terminations` + `truncations` раздельно | `agents` меняется, `possible_agents` фиксирован | `state()` |
| PettingZoo AEC | `agent_selection` | dead step `step(None)` | `_cumulative_rewards` | нет | то же | то же | то же | `state()` |
| RLlib MultiAgentEnv | ключи dict obs | `terminateds[id]`, `__all__` | да, любому агенту в любой шаг | `with_agent_groups` | `get_observation_space(agent_id)` | `terminateds` + `truncateds` | `agents` / `possible_agents` | нет в базовом API |
| OpenSpiel | `CurrentPlayer()` / simultaneous | нет выбывания | `Rewards()` всем игрокам | нет | единый `NumDistinctActions`, `LegalActions(player)` | только `IsTerminal` | `NumPlayers` фиксирован | сам `State` |
| Kaggle environments | статус `ACTIVE`/`INACTIVE` | статус `DONE/ERROR/INVALID/TIMEOUT` | кумулятивный `reward` | нет | схема на env, не на агента | только `DONE` | число агентов фиксировано | `shared` поля |
| Melting Pot | все игроки | респавн, не выбытие | список rewards | нет (collective reward wrapper) | списки specs | dm_env `LAST` | фиксировано | global observations |
| JaxMARL | все `agents` | `dones[agent]`, `__all__` | в dict | нет | `observation_space(agent)` | один `done` | фиксировано (`num_agents`) | `obs["world_state"]` |
| PufferLib | все `possible_agents` | паддинг + `masks=False` | теряется для отсутствующих | нет | берёт space первого агента | `terminals`+`truncations` | паддинг до `possible_agents` | нет |
| NMMO | `agents` | `terminated` в шаг смерти + последняя obs | да | задачи команд, comm | один space | `truncated` в конце горизонта | `agents` сжимается | нет |
| Sample Factory | списки на `num_agents` | `info["is_active"]=False` | в списке | нет | один space | auto-reset | фиксировано | нет |
| EnvPool | `env_id` + player id | нет | нет | нет | `max_num_players` | `info` | `max_num_players` | нет |

### B.1 Acting set (кто ходит на этом шаге)

- **PettingZoo Parallel**: «all agents have simultaneous actions and observations»; `agents` — «may be changed as an environment progresses (i.e. agents can be added or removed)»; `possible_agents` «cannot be changed through play or resetting». https://github.com/Farama-Foundation/PettingZoo/blob/master/docs/api/parallel.md
- **PettingZoo AEC**: «agents act sequentially»; `agent_selection`, `last()`, `agent_iter()`. https://github.com/Farama-Foundation/PettingZoo/blob/master/docs/api/aec.md
- **RLlib new stack**: «The returned observation dict must contain only those agent IDs that compute and send actions into the next `step()` call.»; «the agent IDs contained in or missing from your observations dicts determine the exact order and synchronization of agent actions». Это самый общий механизм: турн-based, simultaneous и любая смесь описываются одним правилом. https://github.com/ray-project/ray/blob/master/doc/source/rllib/multi-agent-envs.md ; код: «Only agents that are supposed to act in this timestep should be present in this dict» https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py
- **OpenSpiel**: `CurrentPlayer()` возвращает id игрока, `kChancePlayerId`, `kSimultaneousPlayerId` или `kTerminalPlayerId`; `IsPlayerActing(player)` = `CurrentPlayer() == player || IsSimultaneousNode()`; `LegalActions(player)` пуст для не ходящих. `rl_environment`: в simultaneous играх ожидается «a list of actions, one per player». https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/spiel.h , https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/rl_environment.py
- **Kaggle environments**: статус агента `ACTIVE` (должен ходить) или `INACTIVE` (не ходит, например в шахматах); в `train()`-обёртке `advance()` крутит игру, пока статус нашего агента `INACTIVE`. https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/core.py
- **Melting Pot**: все игроки ходят одновременно, наблюдения/награды/действия — списки по игрокам. https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/utils/substrates/wrappers/multiplayer_wrapper.py
- **EnvPool**: turn-based (PGX) — `info["current_player"]` указывает игрока, «each environment consumes one action for that current player»; параметр `max_num_players`. https://github.com/sail-sg/envpool/blob/main/docs/env/pgx.rst , https://github.com/sail-sg/envpool/blob/main/docs/content/python_interface.rst
- **HandyRL** (создан под Kaggle-соревнования): `players()`, `turn()` для последовательных, `turns()` для simultaneous («returns the list of player id that can act in the turn»). https://github.com/DeNA/HandyRL/blob/master/docs/custom_environment.md

### B.2 Смерть / elimination

- **PettingZoo AEC**: агент, умерший, должен получить ещё один «dead step»: «an agent that dies must still be given one more turn so that the user can call `last()` and see its final observation, accumulated reward and termination/truncation flag» и только потом удаляется из `agents` (`_was_dead_step`). https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/utils/env.py
- **RLlib**: terminated-флаг можно выставить агенту, который не ходил: «agent A can act in a way that terminates agent B from the episode without agent B having acted itself»; `__all__` завершает всё: «terminates all agents and ends the episode». https://github.com/ray-project/ray/blob/master/doc/source/rllib/multi-agent-envs.md
- **NMMO**: в шаге смерти агент ещё есть в `gym_obs` (`_current_agents = alive + dead_this_tick`), `terminated[agent_id] = True`; потом исчезает. Есть воскрешение в играх (`game.update`), поэтому «time of death must be marked». https://github.com/NeuralMMO/neural-mmo/blob/master/nmmo/core/env.py
- **JaxMARL (SMAX)**: фиксированный массив юнитов, мёртвые `dones[agent] = ~unit_alive`, плюс `dones["__all__"]`. https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/environments/smax/smax_env.py
- **PufferLib**: отсутствующий в `obs` агент получает нулевую obs, нулевой reward, `terminals=True`, `masks=False` («if agent not in obs: ... self.rewards[i] = 0 ... self.masks[i] = False»). То есть **финальная награда для агента, которого нет в `obs`, теряется**. https://github.com/PufferAI/PufferLib/blob/3.0/pufferlib/emulation.py (ветка 3.0; на master библиотека переписана на C/CUDA и этого файла нет)
- **Sample Factory**: `info["is_active"] = False` — данные после смерти считаются invalid и маскируются в loss; предупреждение: доля inactive >50% ухудшает обучение. https://github.com/alex-petrenko/sample-factory/blob/master/docs/07-advanced-topics/inactive-agents.md
- **Kaggle env**: у агента нет «смерти» как отдельного состояния; для Halite/Kore reward выбывшего = `step_eliminated - episode_steps - 1` (отрицательный, чем раньше вылет, тем хуже). https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/halite/halite.json
- **Generals (JAX)**: выбывший игрок остаётся в массивах, «their actions are ignored from then on»; игра идёт, пока жива хоть одна другая команда. https://github.com/strakam/generals-bots
- **OpenSpiel**: выбывания нет, `NumPlayers` фиксирован.

### B.3 Reward, пришедший, когда агент не ходит

- **RLlib**: reward-dict может содержать любого агента в любой шаг: «an action by agent A can trigger a reward for agent B, even when agent B isn't acting itself». https://github.com/ray-project/ray/blob/master/doc/source/rllib/multi-agent-envs.md
- **PettingZoo AEC**: `rewards` — мгновенные, `_cumulative_rewards` накапливает их между ходами агента; `last()` отдаёт накопленное. https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/utils/env.py
- **OpenSpiel**: `Rewards()` — вектор по всем игрокам в состоянии s', при этом `Returns() = Sum(Rewards(S_0..S_t))`. https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/spiel.h
- **Kaggle env**: reward кумулятивный; в обёртке `train()` берётся разность с предыдущим шагом `reward -= self.steps[-2][position].reward`, т.е. при нескольких `INACTIVE`-шагах подряд промежуточные приращения, по моему чтению кода, теряются (логика `advance()` + разность только с `steps[-2]`). https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/core.py (вывод по коду, не по документации)
- **HandyRL**: отдельно `reward()` (на шаге, по игрокам) и `outcome()` (терминальный итог по игрокам, -1/0/1). Для simultaneous N-игроков (Hungry Geese) outcome строится как сумма попарных сравнений рангов: `outcomes[p] += 1/(NUM_AGENTS-1)` за победу над соперником, `-= ` за поражение. https://github.com/DeNA/HandyRL/blob/master/handyrl/envs/kaggle/hungry_geese.py
- **Sample Factory / PufferLib / EnvPool**: reward — вектор на каждый шаг, никакой логики «ожидания хода» нет.

### B.4 Команды и командные награды

- **RLlib**: `with_agent_groups` — группа агентов становится одним «агентом» с Tuple-spaces. https://github.com/ray-project/ray/blob/master/doc/source/rllib/multi-agent-envs.md
- **Melting Pot**: есть `collective_reward_wrapper`. https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/utils/substrates/wrappers/collective_reward_wrapper.py
- **Generals JAX**: `teams=[0,0,1,1]`, общий reward команды +1/-1, общий обзор. https://github.com/strakam/generals-bots
- **Pommerman**: команды `[0,2]` против `[1,3]`, reward `[1,-1,1,-1]`. https://github.com/MultiAgentLearning/playground/blob/master/pommerman/forward_model.py
- **NMMO**: командные tasks, команда как группа агентов с общим Task; baseline обучается IPPO децентрализованно, «allowing flexible team sizes and compositions». https://arxiv.org/abs/2406.05071
- **PettingZoo, OpenSpiel, EnvPool, Kaggle, JaxMARL**: понятия команды в API нет, команда кодируется самим env.

### B.5 Разные observation/action spaces на агента (асимметричные роли)

- PettingZoo: `observation_space(agent)`, `action_space(agent)` per agent («we allow for different observation and action spaces between the agents»). https://github.com/Farama-Foundation/PettingZoo/blob/master/docs/api/parallel.md
- RLlib: `get_observation_space(agent_id)`, `get_action_space(agent_id)`. https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py
- JaxMARL: `observation_space(agent)`, `action_space(agent)` в базовом классе. https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/environments/multi_agent_env.py
- OpenSpiel: один `NumDistinctActions` на игру, различия — через `LegalActions(player)`. https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/spiel.h
- Kaggle: схема observation/action одна на env (`*.json`), роль по `observation.player`. https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/orbit_wars/orbit_wars.json
- **PufferLib ограничение**: размеры берутся у `possible_agents[0]` (`single_agent = self.possible_agents[0]`), значит гетерогенные агенты внутри одного env не поддерживаются. https://github.com/PufferAI/PufferLib/blob/3.0/pufferlib/emulation.py
- NMMO: один и тот же space на всех агентов (`observation_space(agent)` возвращает общий `_obs_space`). https://github.com/NeuralMMO/neural-mmo/blob/master/nmmo/core/env.py

### B.6 Truncation vs termination и финальные observation

- PettingZoo/RLlib/Kaggle-подобные API различают `terminated` и `truncated`: PettingZoo `step` возвращает `(obs, rewards, terminations, truncations, infos)`, RLlib `terminateds` + `truncateds`. https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/utils/env.py , https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py
- Lux S3 (JAX): `terminated` отдельно от `truncated`; truncation когда `steps >= (max_steps_in_match + 1) * match_count_per_episode`; при auto-reset `info["final_observation"]` и `info["final_state"]` сохраняются. https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/src/luxai_s3/env.py
- JaxMARL: единственный `dones` (+ `__all__`), truncation отдельно не выражается; авто-reset в `step`. https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/environments/multi_agent_env.py
- Sample Factory: «Multi-agent environments require auto-reset!» и «we have no use for the last observation of the previous episode», т.е. bootstrap по финальной obs не поддерживается; у нас же в CLAUDE.md truncation добавляет `γ·V(final_obs)`, значит нужна **финальная obs в явном виде**. https://github.com/alex-petrenko/sample-factory/blob/master/docs/03-customization/custom-multi-agent-environments.md
- NMMO: `truncated[agent] = agent_id in self.realm.players` в момент `HORIZON` (жив до конца = truncated, а не terminated). https://github.com/NeuralMMO/neural-mmo/blob/master/nmmo/core/env.py
- Kaggle: статус `DONE` после `episodeSteps` без различия причин. https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/core.py
- OpenSpiel: только `IsTerminal()`; `StepType.LAST`. https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/rl_environment.py

### B.7 Переменное число агентов/юнитов и паддинг

- Агентов (игроков): PettingZoo `agents` ⊆ `possible_agents`; RLlib `agents` / `possible_agents`; PufferLib паддит до `possible_agents`, mask = False для отсутствующих; EnvPool `max_num_players`.
- Юнитов внутри агента: Lux S3 — слоты `(T, N)` + `units_mask` (https://github.com/Lux-AI-Challenge/Lux-Design-S3/blob/main/src/luxai_s3/state.py); NMMO — таблицы фиксированной длины `PLAYER_N_OBS` и `ActionTargets` с масками (action `Target` указывает на строку таблицы Entity) (https://github.com/NeuralMMO/neural-mmo/blob/master/nmmo/core/env.py); µRTS/Generals — действие на каждую клетку + source-маска; Orbit Wars — список сущностей, топ-решение само режет число флотов и ограничивает токены (https://tufalabs.ai/research/orbit-wars/).
- Ни один из проверенных API не стандартизирует «список сущностей переменной длины» в `ObsType` — это всегда решает конкретный env.

### B.8 Global state для центрального критика

- PettingZoo: `state()` — «a global view of the environment appropriate for centralized training decentralized execution methods like QMIX» (опционально, `NotImplementedError` по умолчанию). https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/utils/env.py
- JaxMARL: ключ `obs["world_state"]` (SMAX, MPE) через `get_world_state`. https://github.com/FLAIROx/JaxMARL/blob/main/jaxmarl/environments/smax/smax_env.py
- OpenSpiel: сам `State` + `ObservationTensor(player)` vs `InformationStateTensor(player)`; global state — это полное состояние. https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/spiel.h
- Melting Pot: `global_observations` — «names of the dmlab2d observations to make available to all players». https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/utils/substrates/substrate.py
- Kaggle: поля с `"shared": true` (общий публичный state) и per-agent приватные; для Kaggriculture `farms` shared + `private` не shared. https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kaggriculture/kaggriculture.json
- RLlib / Sample Factory / PufferLib / NMMO / EnvPool: отдельного канала нет (в RLlib выражается агрегацией групп или самим env).

---

## Часть C. Выводы для `GameSpec` / `MultiAgentEnv`

### C.1 Must-have

1. **Acting set явно в каждом шаге**: `acting[seat]` (и/или `info["active"]`, что у нас уже есть). Нужен для sequential (cabt, PGX), mixed (Lux S2 phases, RLlib-правило «missing obs = not acting»), simultaneous (почти все). Источники: https://github.com/ray-project/ray/blob/master/doc/source/rllib/multi-agent-envs.md , https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/spiel.h , https://github.com/DeNA/HandyRL/blob/master/docs/custom_environment.md
2. **Reward для не ходящих и мёртвых агентов**: награды должны приходить любому seat в любой шаг и накапливаться до следующего хода агента (AEC `_cumulative_rewards`). PufferLib теряет финальный reward выбывшего, Kaggle-`train()` теряет промежуточные приращения — это ловушки. Нужны: Halite/Kore (вылет по времени), Hungry Geese, Generals, NMMO, Pommerman.
3. **Dead step с финальной obs и reward**: seat, выбывший на шаге t, получает на этом шаге последнюю obs/reward/`terminated`, затем перестаёт быть в acting и в сборе траектории (PettingZoo dead step, NMMO `_current_agents`). Игра продолжается для остальных. Нужно: Halite, Kore, Hungry Geese, Generals FFA, Pommerman, NMMO, Orbit Wars (elimination).
4. **Раздельные `terminated` и `truncated` по seat + финальная obs для bootstrap**: у нас truncation добавляет `γ·V(final_obs)`. Нужно: Lux S3 (truncation в конце серии), NMMO (alive к концу горизонта = truncated), любая игра с лимитом шагов (Pommerman 800, Orbit Wars 500). Sample Factory-подход «auto-reset, финальной obs нет» нам не подходит.
5. **Outcome как отдельный объект от step-reward**: HandyRL разделяет `reward()` и `outcome()`; Hungry Geese — попарный ранг, Generals — +1/-1 команде, Lux S3 — серия. `GameSpec` должен задавать outcome kind: `score | win_loss_draw | rank | series`, а из них выводятся попарные результаты для ELO/PFSP (совпадает с нашим «pairwise per-seat ratings»).
6. **Серия матчей как первый класс** (Lux S3: best-of-5 с общими параметрами, состояние карты переносится, разный reward на матч и на серию). Нужен уровень «episode = series of matches» с отдельными `match_done` и `episode_done`, и reward = `team_wins` на уровне серии. Только Lux S3 (из найденных).
7. **Команда / группа seats**: общий outcome команды, общий обзор, опционально общий reward. Нужно: Pommerman (2v2), Generals (2v2), NMMO (16x8), GRF (11 игроков, но в Kaggle 1), «team-vs-team с несколькими независимыми ботами» — это N seats, группируемых в команды в outcome, а не в API агентов. Вариант RLlib `with_agent_groups` годится только для «один бот управляет командой».
8. **Per-unit действия переменной длины с масками**. Минимум три формы вывода (слоты с маской как Lux S3, список сущностей как Orbit Wars/Halite, тензор на клетку как µRTS/Generals). `GameSpec` должен описывать действие как «структуру над сущностями + маски над каждым компонентом», а не только `Discrete/MultiDiscrete`. Особенно нужен **pointer-target** (индекс в obs-таблице как NMMO `Target`, цель-планета в Orbit Wars победителя).
9. **Entity-list / Dict observations с паддингом и масками** (Lux S3 `units_mask`, NMMO таблицы `PLAYER_N_OBS`, Orbit Wars списки). Нужны: `max_entities` на тип, маска валидности, стабильная нумерация (Lux рециклирует ID 0..15!, Halite/Kore ключи `uid`).
10. **Переменное число seat в игре** (Orbit Wars 2 или 4, Generals N, Halite 2 или 4): единая политика на 2p и 4p (победители Orbit Wars так и делали), значит seats не должны быть зашиты в формат obs; «токен-сводка игрока на слот» — приём победителя.
11. **Seat symmetry**: канонизация перспективы (все топ-решения Lux S3 «reflect to player-0 view»; Orbit Wars зеркальные карты; Kaggle даёт `observation.player`). В spec нужен флаг/метод `to_canonical_view(seat)` и обратный для действий.
12. **Асимметричные роли**: достаточно `role[seat]` + per-role spaces (как PettingZoo `observation_space(agent)`), но реального соревнования в нашем списке почти нет (см. C.2).
13. **Global state для критика**: опциональный `global_state()` (PettingZoo `state()`, JaxMARL `world_state`, Melting Pot global obs). Нужно для Lux S3 (fog: critic должен видеть всё), Pommerman Team, GRF, NMMO.
14. **Fog of war** — часть spec (Lux S3, Generals, Pommerman Team, NMMO, µRTS partial): obs каждого seat уже отфильтрована, state шире; для кода learner это значит, что recurrent/memory обязательны (Lux S3 победитель — ConvLSTM).
15. **Time limits и timeouts как часть исхода**: Kaggle статусы `TIMEOUT/ERROR/INVALID` дают reward `None` (https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/core.py), Planet Wars «Issuing any invalid order forfeits the game» (https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/planet_wars/README.md). Для RL важен **легальный** вывод (маски), для eval — политика штрафа за невалидное действие.
16. **Скорость**: JAX/векторизованные env у Lux S3 и Generals (10M+ шагов/с) vs Python у Orbit Wars (1 с на ход, 6.3M шагов на GPU-час у 200M модели): `MultiAgentEnv` не должен предполагать шаг-в-процессе на CPU, оставить место для batched/JAX backend (SP6).

### C.2 Выглядит соблазнительно, но в реальных соревнованиях не нужно

1. **Общий протокол «агенты сообщаются» (TeamRadio)** — только Pommerman TeamRadio; в остальных нет (https://github.com/MultiAgentLearning/playground).
2. **Динамическое добавление агентов (новые seat посреди игры)**: нет ни в одном соревновании из списка; `possible_agents` в PettingZoo/RLlib и так фиксирован (https://github.com/Farama-Foundation/PettingZoo/blob/master/docs/api/parallel.md). Нужно только выбывание.
3. **Асимметричные roles типа hunter vs prey как требование API**: в списке нет; ближайшее — Protect the King (NMMO minigame) и атака/защита в GRF. Поддержать через per-seat spaces, но не усложнять ядро.
4. **Chance nodes / mean-field players как в OpenSpiel**: нигде не нужны; случайность скрыта внутри `step`.
5. **Возобновляемая/непрерывная «живая» среда (MMO без эпизодов)**: NMMO 2.0 имеет `HORIZON`, эпизоды конечны.
6. **Сообщения между агентами/язык** (Werewolf, 20 questions) — LLM-среды, выходят за рамки RL над тензорами.
7. **Per-unit отдельный процесс/код как Battlecode**: бот без центрального контроллера, нельзя обучить как один policy с batched actions, полезно знать, но не нужно в `GameSpec`.
8. **Хранение полной истории всех seat в `info`**: ни один API этого не требует; RLlib/PettingZoo передают только текущий шаг.
9. **Отдельный reward на каждый юнит**: победители Lux S3/Orbit Wars использовали **win/loss зеро-сум reward для всей команды**, shaping по юнитам не был ключевым (https://zenn.dev/kurupical/articles/61dbeedf89a29d , https://tufalabs.ai/research/orbit-wars/). Для credit assignment лучше дать hook, но не обязательное поле.

### C.3 Таблица «какое соревнование требует какой фичи»

| Фича | Требуют |
|---|---|
| acting set / sequential | cabt, PGX, Lux S2 (фаза placement), RLlib-кейсы |
| elimination + dead step + отрицательный reward по времени вылета | Halite, Kore, Hungry Geese, Generals FFA/2v2, Pommerman, NMMO, Orbit Wars |
| outcome = rank | Hungry Geese, NMMO (last team), Generals FFA |
| outcome = score | Orbit Wars, Kaggriculture, Halite, Kore |
| outcome = series | Lux S3 |
| команды | Pommerman, Generals 2v2, NMMO, Lux S3 (команда = все 16 юнитов, один бот) |
| per-unit переменный action со слотами/списком | Lux S3, Orbit Wars, Halite, Kore, µRTS, Generals |
| pointer target | NMMO, Orbit Wars (победитель), µRTS/Generals (клетка) |
| fog + memory | Lux S3, Generals, Pommerman Team, NMMO |
| 2p и 4p в одном env | Orbit Wars, Halite, Kore, Generals |
| truncation + bootstrap | почти все (лимиты шагов) |
| global state | Lux S3, Pommerman Team, GRF, NMMO |

### C.4 Ограничения исследования

- Страницы Kaggle (writeups Orbit Wars 2 и 13 место, списки соревнований), CodinGame (правила контестов) и W&B недоступны для чтения без JS; данные о них из результатов поиска или сводок и помечены [частично подтверждено] / [не подтверждено].
- Победители Lux S2, Halite IV, Kore 2022 и Fall/Summer 2025 CodinGame не найдены; статус части env в `kaggle-environments` (crawl, kargo, pyxis, reinforce_tactics, werewolf, planet_wars) как отдельных соревнований не проверен.
- Для Halite III, Hungry Geese (число игроков в соревновании), Lux 2021 числа даны с пометкой [не подтверждено].
- Сводки WebFetch для PDF и длинных страниц могут содержать ошибки (например, при обработке Lux S3 в одной из сводок серия названа «best of 3», хотя в specs `best of 5`); числа выше сверены по исходникам там, где они процитированы.

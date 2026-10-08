# SP1: приёмка по спеке §3 (T8.4, шаги 1–8)

- Ветка `sp1-stabilization`. Начало: HEAD `8f935a2`, после правки документации HEAD `95df18d`.
- Дата: 2026-10-08.
- Машина: WSL2, 8 ядер (`nproc` = 8), 11 ГБ ОЗУ, без GPU. Python 3.12.3, torch 2.14.1+cpu, `.venv`.
- Шаги 9 (вопрос о приёмке) и 10 (merge/push) не выполнялись: merge и push не делались.

## Итог

**ВСЕ КРИТЕРИИ ВЫПОЛНЕНЫ (ALL PASS).** Одна правка документации (диапазон побед в README и CLAUDE.md), коммит `95df18d`; шаг 7 после неё перезапущен.

| § | Критерий | Результат |
|---|---|---|
| 3.1 | Тесты | PASS: 873 passed, 19 deselected, 232 с, exit 0, без предупреждений; дерево чистое; ruff чистый |
| 3.2 | Контрактные тесты | PASS: 93 passed за 29.6 с; у каждого пункта есть зелёный тест |
| 3.3 | «Учится» | PASS: быстрые 0.6–4.2 с; медленный 94.25 % и 92.0 % (≥ 80 %), около 1 мин |
| 3.4 | Видимость без WandB | PASS: структура run dir, логи всех процессов, 4 вида записей в metrics.jsonl, прогресс в консоли |
| 3.5 | Жизненный цикл | PASS: 10/10 тестов; ручное убийство лёрнера даёт exit 1, сообщение с путём к логу, без сирот |
| 3.6 | Скорость | PASS: шаги сред/с 4641 → 9444 → 10664 (монотонно); «до» и «после» есть в `docs/benchmarks.md` |
| 3.7 | Документация | PASS: все команды README дают exit 0; флаги совпадают с `--help`; одна правка `docs:` |
| 3.8 | GPU | PASS (проверка списка): 16 `gpu`-тестов, список совпадает с `docs/GPU_CHECKS.md`; без CUDA все 16 skipped |

---

## §3.1 Тесты: полный быстрый набор, один процесс, без зависаний, запись только в `tmp_path`

Предусловие: `git status --porcelain` пуст.

Команда: `timeout 2400 .venv/bin/python -m pytest -m "not gpu and not slow" -q -p no:cacheprovider`.

- `exit=0`. Результат: **873 passed, 19 deselected in 232.04s (0:03:52)**, wall 234 с.
- В выводе нет ни одного `warning`.

Проверка чистоты (`git status --porcelain --ignored` минус `.venv`/кэши) показала только `!! .superpowers/`. Это заранее существующий игнорируемый каталог плана SDD: тесты его не создают, `runs/`, `checkpoints/` и `*.jsonl` нет. Считаю это результатом **clean**.

`.venv/bin/ruff check .`: `All checks passed!`, exit 0.

Счёт тестов совпадает с CLAUDE.md: 873 быстрых, 3 slow, 16 GPU (873 + 3 + 16 = 892 = 873 + 19 deselected).

**PASS.**

## §3.2 Контрактные тесты

Команда: `.venv/bin/python -m pytest tests/contract tests/unit/test_league_matchmaking.py tests/integration/test_league_runs.py -v -rA`.

Результат: **93 passed in 29.61s**, exit 0. Лог был в `/tmp/sp1-contract.txt`, в конце шага 7 удалён по сценарию брифа.

| Пункт §3.2 | Тест (node id), все PASSED |
|---|---|
| Лёрнер воспроизводит log-prob и value воркера для 4 ядер: без памяти, LSTM, GRU, окно внимания | `tests/contract/test_worker_learner_consistency.py::test_learner_reproduces_worker_logprobs_and_values[none]`, `[lstm]`, `[gru]`, `[window]`; дополнительно `::test_turn_based_stateful_core_reproduces[lstm/gru/window × masks/no_masks]`, `::test_chunk_starting_right_after_a_done_reproduces_from_a_zero_state`, `::test_resumed_parked_buffers_reproduce`, `::test_heterogeneous_agents_and_checkpoint_opponents` |
| Self-play с двумя агентами | `tests/integration/test_league_runs.py::test_two_agent_self_play_reaches_budget_and_both_learners_train` (+ `tests/unit/test_league_matchmaking.py::test_self_play_two_agents_both_get_collecting_envs`) |
| Лига на трёх агентах, встречаются все пары | `tests/integration/test_league_runs.py::test_three_agent_league_all_pairs_meet_and_seats_balanced` (+ `tests/unit/test_league_matchmaking.py::test_league_three_agents_all_pairs_meet`) |
| Пошаговая игра: проигравший получает −1 на своём последнем переходе, done у всех собирающих мест | `tests/contract/test_turn_based.py::test_final_rewards_and_dones_reach_both_players`: проверяет `(1, -1.0, True)` у проигравшего и `done=True` у обоих мест (+ остальные 5 тестов `test_turn_based.py`) |
| Порядок действий при 12 компонентах `MultiDiscrete`/`Tuple` | `tests/contract/test_action_order_rollout.py::test_twelve_units_receive_their_own_head_through_rollout_loop[False]`, `[True]` (12-компонентный `MultiDiscrete` через настоящий `RolloutLoop`, с маской и без); `Tuple` проверяет `tests/unit/test_action_order.py::test_tuple_components_follow_index_order` (11 компонентов, 15/15 тестов файла passed) |
| Баланс мест ±5 % | `tests/unit/test_league_matchmaking.py::test_seat_distribution_balanced_within_5_percent` (+ `::test_four_player_arena_seats_balanced_within_5_percent`) |
| Eval модели с состоянием | `tests/contract/test_eval_stateful.py::test_recurrent_agent_matches_manual_step_loop[lstm]`, `[counter]`, `::test_manual_reference_detects_dropped_or_leaked_state` |

**PASS.**

## §3.3 «Учится»

### Быстрые тесты

Команда: `pytest tests/learning -m "not slow" -v --durations=0`. Результат: **4 passed, 1 deselected in 7.52s**.

| Тест | Длительность |
|---|---|
| `test_fast_learning.py::test_contextual_bandit_is_solved_in_seconds` | 0.61 с |
| `test_fast_learning.py::test_short_chain_is_solved_in_seconds` | 0.86 с |
| `test_fast_learning.py::test_short_chain_needs_discounting` | 4.19 с |
| `test_bc_learns.py::test_bc_imitates_masked_scripted_expert` | 1.57 с |

Все меньше 20 с.

### Медленный тест: крестики-нолики против случайного легального игрока

Команда: `time pytest tests/learning -m slow -v -s`, запущена дважды. `nproc` = 8.

| Прогон | win_rate (greedy, 400 игр) | W / D / L | По местам (W/D/L) | Обучение | pytest | real |
|---|---|---|---|---|---|---|
| 1 | **0.9425** | 377 / 21 / 2 | seat 0: 199/1/0; seat 1: 178/20/2 | 55 с | 55.86 с | 0 м 57.6 с |
| 2 | **0.920** | 368 / 23 / 9 | seat 0: 197/3/0; seat 1: 171/20/9 | 56 с | 57.17 с | 0 м 58.9 с |

Оба прогона ≥ 0.80.

Время около 1 мин, а бриф ожидал «около 3–5 минут». Спека требует «примерно за 3 минуты», так что более быстрый прогон критерию не противоречит. README говорит о 52–73 с, CLAUDE.md — о «~70 s»; наблюдение с ними согласуется.

Дополнительно команда README `pytest -m slow -v` (дословно): 3 passed (ttt + 2 torch.compile) за 73.98 с, real 1 м 16.6 с. Это совпадает с «~1.5 min» в README.

**PASS.**

## §3.4 Видимость без WandB

Команда из брифа: `colosseum train ... --set run.dir=/tmp/sp1-acceptance --set run.name=vis --set training.total_timesteps=150000`, exit 0, около 14 с.

- В консоли `Run directory: /tmp/sp1-acceptance/vis`.
- Строки прогресса:
  - `[agent_0] step 436 | 73.7% budget | 10,788 env-steps/s | ...` — через 10 с после старта;
  - `[agent_0] step 600 | 100.0% budget ...` — финальная.

  На Quick start (шаг 7) строки шли каждые ~10 с: 16:53:21, :31, :41, :51, 16:54:01.
- Run dir содержит `checkpoints`, `config.resolved.yaml`, `logs`, `metrics.jsonl`, `ratings.json`.
- `logs` содержит `learner-agent_0.log`, `main.log`, `worker-0.log`, `worker-1.log`.
- `checkpoints/agent_0` содержит `ckpt_v100 … ckpt_v600` (6 штук).
- Виды записей в `metrics.jsonl`: `Counter({'train': 60, 'episodes': 2, 'system': 2, 'ratings': 2})`, то есть все четыре.
- `ratings.json` содержит `elo`, `env_steps` (155104), `games`, `past_games`, `win_rates`, `wr_vs_past`.

**PASS.**

## §3.5 Жизненный цикл

Команда: `pytest tests/integration/test_lifecycle.py tests/integration/test_league_runs.py::test_resume_continues_versions_env_steps_and_lr -v`. Результат: **10 passed in 50.42s**.

- `test_normal_finish_exit_0_and_final_checkpoint`
- `test_killed_worker_gives_exit_1_and_points_to_its_log`
- `test_learner_crash_in_train_step_gives_exit_1`
- `test_sigterm_stops_all_descendants_within_10s`
- `test_killed_main_process_takes_all_descendants_with_it`
- `test_sigint_to_process_group_exits_130_with_final_checkpoint`
- `test_sigint_during_startup_exits_130_not_aborted`
- `test_config_error_exit_1_without_traceback`
- `test_readme_quickstart_from_repo_root`
- `test_resume_continues_versions_env_steps_and_lr`

Ручная проверка смерти **лёрнера**: `kill -9` лёрнера (pid 9132) через 30 с после старта.

- `exit=1`; главный процесс завершился через 1.47 с после kill.
- Сообщение: `learner-agent_0 died (exit -9), see /tmp/sp1-acceptance/kill/logs/learner-agent_0.log`.
- `pgrep -f multiprocessing.spawn` нашёл один pid (9107), но это была моя собственная обёртка bash: её командная строка содержит сам шаблон pgrep. Повторная проверка (`pgrep -af "multiprocessing|colosseum"`) нашла только текущий shell. Дочерних процессов не осталось.

Бюджет соблюдается: Quick start остановился на 603 888 env-steps при бюджете 600 000, прогон `vis` на 155 104 при 150 000. Перебег — это шаги, уже бывшие в работе при достижении бюджета. Финальные чекпоинты записаны (`ckpt_v2292`, `ckpt_v600`).

**PASS.**

## §3.6 Скорость

Использованы числа «после», которые записал T8.3 (`docs/benchmarks.md`, раздел «После SP1»; сырые данные `docs/benchmarks/after.json`). Бриф свежего прогона не требует (шаг 6 — только `sed` по документу), поэтому бенчмарк не перезапускался.

Команда (одна для «до» и «после»): `.venv/bin/python scripts/bench_throughput.py --workers 1 2 4 --duration 60 --warmup 15`. «После» замерено на коммите `b116b1a`.

| Воркеры | Апдейты/с | Шаги сред/с | chunks/update | Глубина очереди | Узкое место |
|---|---|---|---|---|---|
| 1 | 18.16 | 4641 | 8.00 | 0.4 / 32 | воркеры |
| 2 | 37.08 | 9444 | 8.00 | 2.1 / 32 | воркеры |
| 4 | 41.95 | 10664 | 8.00 | 24.5 / 32 | лёрнер |

- Рост 1 → 2 → 4 строго монотонный и по апдейтам/с, и по шагам сред/с. Числа в `after.json` совпадают с таблицей.
- Раздел «До» есть, с той же командой: 1.68 / 0.70 / 0.60 апдейтов/с и 213 / 90 / 77 шагов сред/с, то есть throughput падал.
- Есть контрольный замер на старой нагрузке (`after-pre-t8.1.json`): 5016 / 9442 / 12945 шагов сред/с, тоже монотонно.

С `b116b1a` до HEAD `src/` менялся только в путях запуска, остановки и ошибок: закрытие очередей при выходе (`launcher.py`, `utils/process.py`), ошибки `colosseum bc`, комментарии. Горячего пути роллаута и обучения эти правки не касаются.

**PASS.**

## §3.7 Документация работает дословно

### Quick start (дословно, из корня репо, в свежем `env -i bash --noprofile --norc`)

Бриф начинает блок с `source .venv/bin/activate`. `git clone` и `scripts/setup-dev.sh` не запускались: `.venv` уже создан этим скриптом.

Первый прогон и повтор после правки документации:

| Команда | exit (прогон 1 / повтор) | Время |
|---|---|---|
| `colosseum validate -c configs/examples/tic_tac_toe.yaml` | 0 / 0 | 1–2 с |
| `colosseum train ... --set run.name=ttt-quickstart` | 0 / 0 | 54 / 53 с |
| `NEW=…; OLD=…; colosseum eval ... -a new=$NEW -a old=$OLD --num-matches 200 --output eval.json` | 0 / 0 | 2 с |
| `colosseum train ... run.name=ttt-continued, resume_from=runs/ttt-quickstart, total_timesteps=1200000` | 0 / 0 | 54 / 52 с |
| `colosseum train -c configs/examples/tic_tac_toe_multi.yaml --set run.name=ttt-league` | 0 / 0 | 52 / 54 с |

- `eval.json` создан. Таблица содержит W/D/L, win rate с 95 % CI, score с CI и разбивку по местам.
  - Повтор: `new` против `old` (`ckpt_v2292` против `ckpt_v1400`): 166/30/4, win_rate 0.830 [0.772, 0.876], score 0.905.
  - seats 0: 98/2/0; seats 1: 68/28/4.
- Продолжение:
  - лог: `Resume [agent_0]: …/ckpt_v2292 (policy_version 2292)`, затем `env-step counter continues from 603888`;
  - первый сохранённый чекпоинт продолжения — `ckpt_v2300`, больше последнего `ckpt_v2292` у quickstart;
  - в пуле FIFO остались `ckpt_v3700…ckpt_v4543`.
- Лига: оба агента дошли до 100 % бюджета, есть `checkpoints/agent_alpha` и `checkpoints/agent_beta`.

### Распределённый режим на localhost

Команды брифа (README с подстановкой `cfg.yaml`, `A`, `B`), выполнены дважды:

- `workers exit=0` (около 9 с).
- Созданы два run dir: `tic_tac_toe-<ts>-learner-agent_0` и `tic_tac_toe-<ts>-workers-DESKTOP-UMCOKC5`, у обоих есть `logs/`.
  - У лёрнера: `learner-agent_0.log`.
  - У воркеров: `worker-0.log`, `worker-1.log`, `workers-main.log`.
  - Имя каталога воркеров `<name>-workers-<host>` совпадает с README (бриф писал короче, `-workers`).
- У лёрнера есть `checkpoints/agent_0/ckpt_v72`.

### Пример из раздела Configuration reference

`colosseum train ... --set run.name=lr-test --set algorithm.learning_rate=1e-3 --set rollout.num_workers=4 --set training.resume_from=null`:

- `colosseum validate` с теми же флагами дословно: exit 0.
- `train` нарушил бы правило окружения «≤ 2 воркера», поэтому запущен с теми же флагами плюс `--set rollout.num_workers=2 --set training.total_timesteps=60000`. Результат: exit 0.
- `config.resolved.yaml` содержит `learning_rate: 0.001`, `resume_from: null`. Значит, `1e-3` и `null` в `--set` разбираются так, как описано в README.

### Разделы с заглушками (`my_game/…`, `path/to/demos/`)

Каждый флаг README сверен с `colosseum <cmd> --help`, все существуют:

- `validate`/`train`: `-c`, `--set`;
- `eval`: `-c`, `-a`, `--num-matches`, `--output`, `--deterministic`;
- `bc`: `-c`, `--data`, `--output`, `--epochs`, `--seq-len`;
- `serve-weight-store --port`;
- `run-learner`: `-c`, `--agent`, `--traj-port`, `--weight-store`, `--set`;
- `run-workers`: `-c`, `--weight-store`, `-l`, `--set`.

Не запускались:
- `uv pip install … ".[wandb]"`: установка пакетов запрещена правилами окружения;
- `pytest -m gpu -v`: эквивалент выполнен на шаге 8.

### README и CLAUDE.md против наблюдений шагов 1–6

Совпадают:
- счёт тестов 873/3/16;
- время Quick start 52–73 с;
- «~1.5 min» для slow;
- формат сообщения о смерти процесса;
- коды выхода;
- структура run dir;
- таблица бенчмарка.

Одно расхождение исправлено:
- Было: README «wins 86–93% … (four measured runs)», CLAUDE.md «reaches 86–93% wins».
- Наблюдалось: 94.25 %, вне диапазона.
- Исправление: «86–94%» и «six measured runs».
- Коммит `95df18d docs: tic-tac-toe win-rate range includes the T8.4 acceptance runs (86–94%)`.
- После правки прошли тесты, читающие README (`tests/integration/test_lifecycle.py`, 9 passed), и шаг 7 перезапущен целиком (таблица выше): всё exit 0.

Уборка: `rm -rf runs eval.json /tmp/sp1-acceptance /tmp/sp1-*.txt`.

**PASS.**

## §3.8 GPU

- `pytest -m gpu --collect-only -q`: собрано **16** тестов.
- Сравнение с `grep -oE 'tests/[^ |`]+::[^ |`]+' docs/GPU_CHECKS.md`: **`GPU list matches`**, diff пуст.
- `pytest -m gpu -q`: **16 skipped**, 876 deselected, 0 failed. Причина пропуска — «requires a CUDA device»:
  - `tests/unit/test_algorithm_state.py` ×2;
  - `tests/unit/test_appo_metrics.py` ×6 (AMP fp16/bf16 и pin_memory);
  - `tests/unit/test_bc_trainer.py` ×2;
  - `tests/unit/test_kickstart_kl.py` ×2;
  - `tests/unit/test_state.py` ×4.
- В `docs/GPU_CHECKS.md` есть инструкция прогона: `scripts/setup-dev.sh --gpu`, проверка `torch.cuda.is_available()`, затем `.venv/bin/python -m pytest -m gpu -v`.

**Эти 16 тестов ещё нужно прогнать на машине с CUDA** командой из `docs/GPU_CHECKS.md`. По решению контроллера приёмка SP1 проверяет только полноту списка.

**PASS** (в рамках решения контроллера).

---

## Соответствие `review/repro/` тестам

Скрипты не конвертировались (решение контроллера).

Обозначения:
- «вне SP1» — находка не входит в трассировку спеки §9 или отнесена к SP2–SP6;
- «вспомогательный» — модуль поддержки без собственного сценария.

### `review/repro/env_types/`

| Скрипт | Сценарий | Покрывающий тест / статус |
|---|---|---|
| `harness.py`, `toy_envs.py` | вспомогательные: запуск воркера in-process, игрушечные среды | неприменимо. Аналоги в SP1: `tests/contract/harness.py`, `tests/helpers.py`, `tests/dataflow_helpers.py` |
| `exp_solo_ffa_turn.py` — SOLO | 1-игрок через воркер → APPO → координатор → eval | `tests/learning/test_fast_learning.py::test_contextual_bandit_is_solved_in_seconds`, `::test_short_chain_is_solved_in_seconds` (1-игроковые среды через настоящий цикл); `tests/unit/test_league_matchmaking.py::test_single_player_env_gets_solo_matches`; `tests/unit/test_eval_engine.py::test_solo_schedule`; `tests/unit/test_eval_report.py::test_solo_report_has_normal_intervals`; `tests/integration/test_eval_cli.py::test_eval_cli_rounds_an_odd_pairwise_num_matches_up` (solo-режим CLI) |
| `exp_solo_ffa_turn.py` — FFA-3 с выбыванием | выбывание игрока | вне SP1 (SP2: выбывание игроков). Ранги и исходы FFA: `tests/contract/test_seat_results.py::test_worker_reports_every_seat_of_a_four_seat_ffa`, `tests/unit/test_league_ratings.py::test_ffa_ranking_gives_correct_pairwise_results` |
| `exp_solo_ffa_turn.py` — TURN-BASED | `info["active"]`, награды и done неактивных мест | `tests/contract/test_turn_based.py` (все 6), `tests/contract/test_worker_learner_consistency.py::test_turn_based_stateful_core_reproduces` |
| `exp_spaces.py` — Dict-наблюдения | Dict obs в VectorEnv/validate | вне SP1 (SP2: Dict observations) |
| `exp_spaces.py` — покомпонентные действия | `MultiDiscrete([A]*K)` + CompositeDist + маска по юнитам | `tests/contract/test_action_order_rollout.py::test_twelve_units_receive_their_own_head_through_rollout_loop`, `tests/unit/test_action_order.py::test_twelve_masked_heads_step_and_decode`, `::test_multidiscrete_mask_segments_follow_index_order` |
| `exp_spaces.py` — масштаб совместного log-prob ratio по K | рост ratio с числом юнитов | вне SP1 (SP2: per-unit actions; в SP1 из ET-07 закрыт только порядок компонентов, `tests/unit/test_action_order.py`) |
| `exp_spaces.py` — рекуррентная модель в eval | LSTM обходится в eval | `tests/contract/test_eval_stateful.py::test_recurrent_agent_matches_manual_step_loop`, `::test_manual_reference_detects_dropped_or_leaked_state` |
| `exp_teams.py` — команды 2v2 | командные игры | вне SP1 (SP2: команды/роли) |
| `exp_teams.py` — рейтинги FFA, один агент на нескольких местах | `player_outcomes` и ELO | `tests/contract/test_seat_results.py::test_same_agent_seats_carry_no_cross_agent_signal`, `::test_coordinator_consumes_seats_without_collisions`; `tests/unit/test_league_ratings.py::test_agent_holding_two_seats_counts_every_cross_pair`, `::test_ffa_ranking_gives_correct_pairwise_results` |
| `exp_teams.py` — посадка в arena, `num_players=1`, 3-игроковая FFA | места в матчах | `tests/unit/test_league_matchmaking.py::test_seat_distribution_balanced_within_5_percent`, `::test_four_player_arena_has_four_slots`, `::test_league_single_agent_falls_back_to_solo`, `::test_single_player_env_gets_solo_matches` |
| `exp_teams.py` — solo-исход из наград | `player_outcomes([7.0])` | `tests/unit/test_review_fixes.py::test_outcomes_from_rewards`, `tests/contract/test_seat_results.py::test_worker_falls_back_to_reward_outcomes_without_env_signal` |
| `solo.yaml`, `ffa.yaml`, `ffa_league.yaml` (конфиги для CLI) | см. строки SOLO / FFA выше | как выше; FFA с выбыванием — вне SP1 (SP2) |

### `review/repro/r1_algorithms/`

| Скрипт | Сценарий | Покрывающий тест / статус |
|---|---|---|
| `t1_vtrace_ref.py` | V-trace против эталона (Espeholt eq. 1) | `tests/unit/test_vtrace_lambda.py::test_vtrace_lambda_matches_reference`, `::test_on_policy_vtrace_lambda_equals_gae`; `tests/unit/test_vtrace.py` (8 тестов) |
| `t1_vtrace_ref.py` — смещение при truncation | time-limit как терминал | `tests/contract/test_truncation.py::test_truncation_adds_the_discounted_value_of_the_final_observation` (+ остальные 5 тестов файла) |
| `t2_dists.py` — строка маски из одних False | CategoricalDist падает | R1-13. В SP1 неактивные места не идут в инференс, а активное место с пустой маской даёт понятную ошибку: `tests/contract/test_turn_based.py::test_empty_mask_on_an_acting_slot_raises`, `tests/unit/test_eval_engine.py::test_acting_seat_without_legal_action_raises`, `tests/unit/test_registry.py::test_validate_config_rejects_acting_seat_with_empty_mask` |
| `t2_dists.py` — log_prob/энтропия/градиент с маской, KL масок | NaN/inf | `tests/unit/test_action_masking.py::test_categorical_log_prob_with_mask`, `tests/unit/test_distributions.py::test_masked_categorical_entropy_gradient_survives_loss_scaling`, `tests/unit/test_kickstart_kl.py::test_masked_kl_is_finite_with_finite_grad`, `::test_masked_kl_renormalizes_over_legal_actions` |
| `t2_dists.py` — DiagGaussian с многомерным Box; неограниченный log_std | R1-16, R1-15 | вне SP1: нет в трассировке §9 |
| `t3_recurrent_e2e.py` | чанк «как у воркера» → APPO; kickstart/BC/eval на рекуррентной сети | `tests/contract/test_worker_learner_consistency.py::test_learner_reproduces_worker_logprobs_and_values[lstm/gru]`, `tests/unit/test_kickstart_kl.py::test_kickstart_lstm_masked_teacher_equal_to_student_gives_zero_kl`, `tests/unit/test_bc_trainer.py::test_stateful_bc_uses_unroll_with_resets`, `tests/contract/test_eval_stateful.py::test_recurrent_agent_matches_manual_step_loop` |
| `t4_misc.py` (1) | mse_loss в смешанной точности под AMP | GPU-тест `tests/unit/test_appo_metrics.py::test_amp_train_step_on_cuda`: ещё не прогнан, см. §3.8 |
| `t4_misc.py` (2) | NormalizeObs считает выборки много раз | `tests/unit/test_obs_normalization.py::test_train_step_counts_each_sample_once`, `::test_forward_never_updates_stats_even_in_train_mode` |
| `t4_misc.py` (3) | LR и lambda kickstart сбрасываются при resume | `tests/unit/test_algorithm_state.py::test_resume_reproduces_the_next_update_exactly`, `tests/unit/test_kickstart_kl.py::test_lambda_decays_linearly_and_round_trips`, `tests/unit/test_checkpoint_store.py::test_resumed_learner_lr_follows_restored_progress`, `tests/integration/test_league_runs.py::test_resume_continues_versions_env_steps_and_lr` |
| `t4_misc.py` (4) | доля клипа при policy lag | R1-10 («только метрика»): `tests/unit/test_policy_lag.py::test_learner_emits_policy_lag_mean_and_max`, `tests/unit/test_appo_metrics.py::test_appo_reports_diagnostic_metrics_on_policy`. Ограничение лага — вне SP1 (SP5 `max_policy_lag`) |
| `t4_misc.py` (5) | скорость рекуррентного unroll (цикл против cuDNN) | вне SP1 (R1-09, SP6: cuDNN RNN path) |
| `t4_misc.py` (6) | offline BC игнорирует маски | `tests/unit/test_bc_trainer.py::test_masks_are_applied`, `::test_masks_are_applied_for_stateful_models`, `tests/learning/test_bc_learns.py::test_bc_imitates_masked_scripted_expert` |
| `t4_misc.py` (7) | BC с Box и `action_type='discrete'` по умолчанию | `tests/unit/test_bc_trainer.py::test_continuous_bc_uses_log_prob_and_trains_log_std`, `::test_float_actions_with_discrete_space_raise`, `tests/integration/test_bc_cli.py::test_bc_cli_has_no_action_type_option` |

### `review/repro/r2_worker/`

| Скрипт | Сценарий | Покрывающий тест / статус |
|---|---|---|
| `common.py` | вспомогательный | неприменимо (аналоги: `tests/contract/harness.py`, `tests/dataflow_helpers.py`) |
| `bench.py` | throughput одного воркера, потоки torch | замер: `scripts/bench_throughput.py` + `docs/benchmarks.md` (§3.6). Тесты: `tests/unit/test_bench_throughput.py`, `tests/integration/test_worker_threads.py::test_spawned_worker_uses_rollout_torch_threads`, `tests/unit/test_threads.py` |
| `repro_actionspec.py` | 12-компонентный `MultiDiscrete` → не те юниты | `tests/unit/test_action_order.py::test_multidiscrete_components_follow_index_order`, `tests/contract/test_action_order_rollout.py::test_twelve_units_receive_their_own_head_through_rollout_loop` |
| `repro_alignment.py` | выравнивание obs/reward/done/bootstrap в чанках, границы эпизодов | `tests/contract/test_rollout_loop_characterization.py::test_transitions_are_consecutive_and_rewards_dones_align`, `tests/contract/test_open_transitions.py` (4), `tests/contract/test_truncation.py` (6) |
| `repro_commands.py` | две команды refresh в очереди, потеря чекпоинтов | `tests/unit/test_drain_commands.py::test_drain_commands_returns_newest_with_merged_checkpoints` |
| `repro_lstm.py` | форма `lstm_hidden` от воркера, которую APPO не принимает | `tests/contract/test_rollout_state.py::test_chunks_carry_initial_state_with_batch_one`, `tests/contract/test_worker_learner_consistency.py::test_learner_reproduces_worker_logprobs_and_values[lstm]` |
| `repro_nan.py` — пустая маска | воркер падает на маске из одних False | закрыто для неактивных мест (инференс только у активных); активное место без легального хода — явная ошибка: `tests/contract/test_turn_based.py::test_empty_mask_on_an_acting_slot_raises` |
| `repro_nan.py` — NaN-наблюдение | воркер падает на NaN | вне SP1 (R2-15 нет в §9). Смерть воркера даёт exit 1 и путь к логу: `tests/integration/test_lifecycle.py::test_killed_worker_gives_exit_1_and_points_to_its_log` |
| `repro_reassign_waste.py` | частичные буферы выбрасываются при переназначении | `tests/contract/test_parking.py::test_reassignment_parks_buffers_without_losing_transitions` (+ остальные 3 теста файла) |
| `repro_subproc_exc.py` | исключение среды в subprocess-ребёнке приходит как `EOFError` без контекста | вне SP1 (R2-16a нет в §9; код `_worker_loop` не менялся). Падение воркера при этом ловит lifecycle (exit 1 + путь к логу) |
| `repro_vecenv_infoleak.py` | info прошлого эпизода протекает после auto-reset | `tests/unit/test_vec_env_reset_info.py::test_auto_reset_info_is_the_reset_info` |
| `repro_weightlag.py` | воркер берёт старые веса из очереди maxsize=2 | `tests/unit/test_ipc_latest.py::test_worker_gets_newest_weights_after_a_burst_of_publishes`, `::test_push_weights_leaves_only_the_newest_payload` |
| `subproc_rss.py` | тяжёлые subprocess-дети (~240 МБ RSS, импорт torch) | вне SP1 (R2-16c; `_create_env` по-прежнему в `launcher.py`). Число потоков у детей: `tests/integration/test_worker_threads.py::test_subprocess_env_children_use_one_thread` |

### `review/repro/r3_distributed/`

| Скрипт | Сценарий | Покрывающий тест / статус |
|---|---|---|
| `scripts/bench_serialization.py` | стоимость сериализации torch.save+lz4 против numpy | скорость — вне SP1 (SP5: wire format). Корректность нового numpy-формата: `tests/unit/test_serialization.py`, `tests/unit/test_payloads.py` |
| `scripts/probe_maxmsg.py` | лимит gRPC 64 МиБ на веса | вне SP1 (R3-12, SP5). Лимит настраивается через `transport.grpc_max_message_mb`; отдельного теста нет |
| `scripts/probes.py` — `version_regression`, `weight_store_down` | рестарт лёрнера / недоступный store | вне SP1 (SP5: отказоустойчивость; README «run-learner cannot resume») |
| `scripts/probes.py` — `backpressure`, `send_latency` | `serve-trajectory`, поток на чанк | `serve-trajectory` удалён: `tests/unit/test_launcher_lifecycle.py::test_serve_trajectory_command_is_gone`; устойчивость к мёртвому лёрнеру: `tests/integration/test_distributed.py::test_grpc_trajectory_sink_tolerates_dead_learner`. Постоянные потоки и задержка — вне SP1 (SP5) |
| `scripts/probes.py` — `local_weight_queue_staleness` | локально воркер грузит старые веса | `tests/unit/test_ipc_latest.py::test_worker_gets_newest_weights_after_a_burst_of_publishes` |
| `multi_crash.yaml` | падение лёрнера одного агента вешает лигу | `tests/integration/test_budget_stop.py::test_a_dead_learner_stops_the_run`, `tests/integration/test_lifecycle.py::test_learner_crash_in_train_step_gives_exit_1`, ручная проверка §3.5; плохой `algorithm_class` до запуска: `tests/unit/test_entry_point_validation.py::test_train_rejects_invalid_config_without_creating_a_run_dir` |

### `review/repro/r4_league/`

| Скрипт | Сценарий | Покрывающий тест / статус |
|---|---|---|
| `v1_matchmaking.py` A | лига из 3 агентов, матчи только для первого | `tests/unit/test_league_matchmaking.py::test_league_three_agents_all_pairs_meet`, `::test_owner_rotates_with_global_env_index_and_round`, `tests/integration/test_league_runs.py::test_three_agent_league_all_pairs_meet_and_seats_balanced` |
| `v1_matchmaking.py` B | self-play с 2 агентами | `tests/unit/test_league_matchmaking.py::test_self_play_two_agents_both_get_collecting_envs`, `tests/integration/test_league_runs.py::test_two_agent_self_play_reaches_budget_and_both_learners_train` |
| `v1_matchmaking.py` C | обучаемая политика всегда на месте 0 | `tests/unit/test_league_matchmaking.py::test_seat_distribution_balanced_within_5_percent`, `::test_checkpoint_opponents_spread_over_seats` |
| `v1_matchmaking.py` D | 4p arena, коллизия ключей исходов | `tests/unit/test_league_matchmaking.py::test_four_player_arena_has_four_slots`, `tests/contract/test_seat_results.py::test_worker_reports_every_seat_of_a_four_seat_ffa`, `::test_coordinator_consumes_seats_without_collisions` |
| `v1_matchmaking.py` E | в self-play рейтингов нет | `tests/unit/test_league_ratings.py::test_same_agent_pairs_do_not_touch_elo_but_update_wr_vs_past`, `::test_past_win_rate_window`, `tests/integration/test_metrics_outputs.py::test_metrics_jsonl_ratings_json_and_console` |
| `v1_matchmaking.py` F | PFSP по win rate за всё время (нестационарность) | частично: `tests/unit/test_league_matchmaking.py::test_pfsp_prefers_hard_opponents`, `tests/unit/test_review_fixes.py::test_pfsp_uses_win_rates`. Окна и снимки — вне SP1 (R4-09 частично; SP3/SP4) |
| `v1_matchmaking.py` G | отсутствующий чекпоинт → `latest`, но `collect=False` | `tests/unit/test_checkpoint_store.py::test_missing_checkpoint_falls_back_to_latest_and_collects` |
| `v1_matchmaking.py` H | в win rate попадают абсолютные N-игроковые исходы | `tests/unit/test_league_ratings.py::test_ffa_ranking_gives_correct_pairwise_results`, `::test_ties_are_order_independent`, `::test_win_rate_tracker_stores_fractional_sums` |
| `v2_ckpt_elo_cfg.py` A | второй запуск в том же каталоге чекпоинтов | run dir на каждый запуск: `tests/unit/test_run_dir_logging.py::test_explicit_name_refuses_existing_run`; `tests/unit/test_checkpoint_store.py::test_duplicate_id_is_replaced_not_duplicated`, `::test_final_checkpoint_roundtrip_and_version_continuation` |
| `v2_ckpt_elo_cfg.py` B | зависимость ELO от порядка, эффективный K в N-игроковой | `tests/unit/test_league_ratings.py::test_elo_update_pairs_is_order_independent`, `::test_eight_player_ffa_winner_moves_like_a_two_player_win`. Неограниченный дрейф и настраиваемый K — вне SP1 (SP4: OpenSkill/BT) |
| `v2_ckpt_elo_cfg.py` C | per-agent override заменяет секцию, лишние ключи молча принимаются | `tests/unit/test_config_overrides.py::test_agent_override_deep_merges_onto_global_section`, `::test_typos_are_rejected`, `::test_every_config_model_forbids_extra_keys` |
| `v3_eval.py` A | ничьи и CI у обратной записи | `tests/unit/test_eval_report.py::test_pair_rows_are_consistent_and_contain_their_estimates`, `::test_wilson_interval_contains_point_and_handles_edges` |
| `v3_eval.py` B, C | преимущество первого хода, ротация мест | `tests/unit/test_eval_engine.py::test_pairwise_schedule_balances_seats`, `tests/unit/test_eval_report.py::test_evaluate_end_to_end_per_seat_breakdown` |
| `v3_eval.py` D | LSTM обходится в eval/kickstart | `tests/contract/test_eval_stateful.py::test_recurrent_agent_matches_manual_step_loop`, `tests/unit/test_kickstart_kl.py::test_kickstart_lstm_masked_teacher_equal_to_student_gives_zero_kl` |
| `v3_eval.py` E | разные архитектуры в CLI eval | `tests/integration/test_eval_cli.py::test_eval_cli_pairs_pt_and_heterogeneous_checkpoint`, `::test_load_eval_model_builds_from_meta_networks`. Скриптовые агенты в eval — вне SP1 (R4-12 частично; SP3) |

### `review/repro/r5_config/`

| Скрипт | Сценарий | Покрывающий тест / статус |
|---|---|---|
| `t_config.py` 1 | загрузка всех примеров конфигов | `tests/unit/test_examples.py::test_every_example_config_validates`, `tests/unit/test_registry.py::test_example_configs_build_and_validate` |
| `t_config.py` 2 | опечатки в ключах | `tests/unit/test_config_overrides.py::test_typos_are_rejected`, `::test_every_config_model_forbids_extra_keys` |
| `t_config.py` 3 | семантика per-agent override | `tests/unit/test_config_overrides.py::test_agent_override_deep_merges_onto_global_section`, `::test_partial_networks_override_merges_core_kwargs`, `::test_unknown_agent_id_raises` |
| `t_config.py` 4 | семантика `--set`, `agents.*` при `agents: null` | `tests/unit/test_config_overrides.py::test_parse_override_value`, `::test_parse_override_value_quoting`, `::test_apply_overrides_sets_nested_values_and_creates_agent_sections`, `::test_load_config_applies_overrides_before_validation` |
| `t_stale.py` | старый каталог чекпоинтов от другой игры/архитектуры | `tests/unit/test_run_dir_logging.py::test_explicit_name_refuses_existing_run`, `tests/unit/test_checkpoint_store.py::test_resume_rejects_architecture_mismatch_before_spawning`, `::test_check_model_state_reports_architecture_mismatch` |
| `badnet/nets.py` | вспомогательный: сети с неверными размерами | неприменимо (аналоги в тестах `test_registry.py`) |
| `sc/sitecustomize.py` | вспомогательный: включение логов в дочерних процессах | неприменимо: заменено логами по процессам (`tests/unit/test_run_dir_logging.py::test_setup_process_logging_file_info_console_warning`, `tests/integration/test_run_dir_outputs.py::test_run_dir_has_resolved_config_and_process_logs`) |
| `bad_latent.yaml`, `bad_value.yaml`, `bad_actions.yaml` | неверный размер головы, значения, действия | `tests/unit/test_registry.py::test_validate_config_reports_head_dim_mismatch`, `::test_validate_config_reports_value_shape`, `::test_validate_config_reports_action_shape_mismatch` |
| `bad_import.yaml` | опечатка в имени класса | `tests/unit/test_registry.py::test_validate_config_reports_bad_env_and_bad_class`, `::test_wrong_classes_are_rejected_before_construction` |
| `bad_kwargs.yaml` | сети от другой игры (chase-энкодер на ttt) | `tests/unit/test_registry.py::test_validate_config_reports_head_dim_mismatch`, `::test_validate_config_reports_unroll_shape` (пробный шаг/unroll на настоящей среде) |
| `bad_rnn.yaml` | старые ключи `recurrent_*` | `tests/unit/test_registry.py::test_network_config_rejects_removed_recurrent_keys` |
| `np3.yaml` | `num_players` не совпадает со средой | `tests/unit/test_config_overrides.py::test_validate_rejects_num_players_mismatch` |
| `typo.yaml` | опечатка в ключе | `tests/unit/test_config_overrides.py::test_typos_are_rejected`, `::test_cli_validate_reports_typo_with_exit_1` |
| `sp_one_agent.yaml`, `sp_two_agents.yaml` | self-play с 1 и 2 агентами | `tests/integration/test_pipelines.py::test_full_pipeline`, `tests/integration/test_league_runs.py::test_two_agent_self_play_reaches_budget_and_both_learners_train` |

### `review/repro/r6_empirical/`

| Скрипт | Сценарий | Покрывающий тест / статус |
|---|---|---|
| `scripts/ttt_eval.py` | ttt против случайного легального игрока | `tests/learning/ttt_eval.py` (`play_vs_random`) в `tests/learning/test_ttt_slow.py::test_tic_tac_toe_beats_random_80_percent`. Счётчик нелегальных ходов неприменим: маски |
| `scripts/gen_ttt_expert.py` | данные BC от скриптового эксперта | `tests/learning/test_bc_learns.py::test_bc_imitates_masked_scripted_expert`, `tests/integration/test_bc_cli.py::test_bc_cli_trains_and_saves_loadable_weights` |
| `scripts/chase_eval.py` | качество chase против случайного | неприменимо / вне SP1: качество обучения chase не критерий. Пример покрыт `tests/unit/test_examples.py::test_chase_accepts_its_own_sampled_actions_and_scalars`, `tests/unit/test_action_order.py::test_chase_example_policy_matches_its_action_space` |
| `scripts/tput_parse.py` | разбор throughput из логов | заменён `scripts/bench_throughput.py` (по `metrics.jsonl`) и `tests/unit/test_bench_throughput.py` |
| `scripts/train_logged.py` | обёртка для логов дочерних процессов | неприменимо: заменено логами по процессам (`tests/integration/test_run_dir_outputs.py::test_run_dir_has_resolved_config_and_process_logs`) |
| `nplayer/r6probe/envs.py` + `ffa3.yaml`, `solo.yaml`, `ttt_active.yaml` | среды с N≠2 и пошаговая ttt | 3+ игроков: `tests/unit/test_league_matchmaking.py::test_four_player_arena_has_four_slots`, `tests/contract/test_seat_results.py::test_worker_reports_every_seat_of_a_four_seat_ffa`, `tests/unit/test_eval_report.py::test_three_player_pair_uses_seat_groups`. Solo: см. env_types SOLO. Пошаговая ttt: `tests/unit/test_examples.py::test_tic_tac_toe_reports_active_player_and_legal_moves` и `tests/contract/test_turn_based.py` |
| логи `robust/*.log` (не скрипты) | kill worker, SIGINT, SIGTERM, resume, from-BC | `tests/integration/test_lifecycle.py` (все 9), `tests/integration/test_league_runs.py::test_resume_continues_versions_env_steps_and_lr`, `tests/unit/test_checkpoint_store.py::test_resolve_resume_checkpoint_dir_run_dir_and_pt` |

Файлы без сценария не скрипты и в таблицу не входят: `*.txt`, `*.log`, `env_types/design_draft.md`, `__init__.py`.

---

## Отклонения и замечания

1. **Правка документации** (коммит `95df18d`, единственное изменение):
   - README «86–93% … (four measured runs)» → «86–94% … (six measured runs)»;
   - CLAUDE.md «86–93%» → «86–94%».

   Причина — наблюдённые 94.25 %. Шаг 7 после правки перезапущен полностью, всё exit 0.
2. **Время медленного теста** около 1 мин вместо ожидаемых брифом «3–5 минут». Спека («примерно за 3 минуты») и README («52–73 s») этому не противоречат.
3. **Бенчмарк не перезапускался.** Использованы числа T8.3 с коммита `b116b1a`. Последующие изменения `src/` касаются только путей запуска, остановки и ошибок.
4. **Пример Configuration reference** (`rollout.num_workers=4`):
   - `validate` выполнен дословно;
   - `train` запущен с дополнительными `--set rollout.num_workers=2 --set training.total_timesteps=60000` по правилу окружения «≤ 2 воркера».
5. **Не запускались:**
   - `git clone` и `scripts/setup-dev.sh`: бриф начинает блок с `source .venv/bin/activate`;
   - `uv pip install … [wandb]`: установка пакетов запрещена;
   - команды с заглушками `my_game/…`: флаги сверены с `--help`.
6. **Косметика, не критерий:** run dir `run-workers` содержит пустой `checkpoints/`. README говорит, что там пишутся только логи; утверждение README это не нарушает.
7. **Совпадение pgrep на шаге 5:** `pgrep -f "multiprocessing.spawn"` из брифа совпадает с командной строкой самого вызывающего shell. Найденный pid 9107 — обёртка bash. Повторная проверка подтвердила, что дочерних процессов нет.
8. **GPU:** 16 тестов `-m gpu` ещё нужно прогнать на машине с CUDA по `docs/GPU_CHECKS.md`.

## Гигиена

- После всех шагов и уборки `git status --porcelain` пуст.
- `--ignored`: только заранее существующий `.superpowers/`.
- `runs/`, `eval.json`, `/tmp/sp1-acceptance` удалены.
- `ps`: процессов python/colosseum нет, кроме собственного shell проверки.

---

## Решения контроллера во время SP1 (все записи «Ruling:» из журнала, по порядку)

Каждая запись: что решено — почему — чем грозит, если решение неверно.

- Preflight: Ruling: no __init__.py under tests/; helpers imported only as `from helpers import`; unique support-module/class names — T0.2 basename guard + single module object — if wrong: some cross-dir imports need moving support modules to tests/ (cheap)
- Preflight: Ruling: T6.2 wrappers keyword-only, log_dir added to existing kwargs — T2.1 made targets kw-only; positional args would TypeError every child — if wrong: none (strictly safer)
- Preflight: Ruling: benchmark migrated in T6.2 (run dir/Launcher signature) and T6.3 (metrics.jsonl instead of WandBLogger patch) — overview assigned it but task text omitted it; T8.3 needs comparable numbers — if wrong: "после" numbers not comparable, redo T8.3 measure
- Preflight: Ruling: T4.5 keeps CompositeDist.cat — replacing class would drop Part A method used by default unroll — if wrong: none
- Preflight: Ruling: T7.2 tests build ckpt path as base_dir/agent/ckpt_id — save() returns id — if wrong: none
- Preflight: Ruling: T2.1 RED uses monkeypatch OMP_NUM_THREADS=3 so test can fail — conftest exports 1 — if wrong: none
- Preflight: Ruling: briefs carry ALL+task rulings automatically (brief.sh appends rulings.md sections) — brief.sh copied only task text, overview duties were lost — if wrong: none
- Preflight: Ruling: execution order T5.1-T5.3, T6.1-T6.5, then T5.4, T7.1, T7.2, T8.x — T5.4 depends on T6.5 — if wrong: none
- Preflight: Ruling: counter attribute name `_env_step_counter` (T2.5's) wins; T5.3/T6.2 adapt — implemented-first name, least churn — if wrong: cosmetic rename
- Preflight: Ruling: T2.6 keeps stats["episodes"]; greps whole repo before removing _extract_action_masks/_apply_command (move mask helper into eval.py) — avoid breaking eval/tests — if wrong: none
- Preflight: Ruling: T2.5 rewrites exact-version pipeline assertion as structural check — budget stop/final ckpt change exact versions — if wrong: weaker test (still checks FIFO+monotonic+interval)
- Preflight: Ruling: T5.1 deletes only setup_matchmaker line, keeps/adapts WorkerCommand contract test; T5.3 updates ckpt["state_dict"]->["model_state"] in that test — preserve coverage — if wrong: none
- Preflight: Ruling: T6.4 edits existing test_wandb_logger.py keeping T0.1 subprocess test — avoid file clash/lost coverage — if wrong: none
- Preflight: Ruling: T4.1 test kit builds on Part A helpers via `from helpers import` — overview forbids duplicates; double module load — if wrong: none
- Preflight: Ruling: T4.3 dedupes APPO unroll (_evaluate vs compute_loss) and makes Distribution.cat preserve masks — avoid duplicated logic and NaN KL — if wrong: small refactor redo
- Preflight: Ruling: T2.2 consolidates numpy<->torch helpers in core/ipc.py — three copies otherwise — if wrong: none
- Preflight: Ruling: T5.3 removes superseded learner helpers and continues train_step/consumed_samples on resume — counters restarting at 0 break LR/budget continuation — if wrong: none
- Preflight: Ruling: GPU-marked tests added to T4.3 (kickstart CUDA), T4.4 (BC CUDA), T4.6 (AMP fp16/bf16, pin_memory, device transfer, state_dict on CUDA) — spec §3.8 coverage gap — if wrong: a few extra skipped tests on CPU
- Preflight: Ruling: review/repro scripts are mapped to covering tests in T8.4 (no separate conversion) — spec says "где применимо"; contract tests already cover most scenarios — if wrong: a repro scenario stays untested; easy follow-up
- Task T0.4: implementer DONE_WITH_CONCERNS (6b7cf39). Ruling: 4-worker benchmark run is allowed — the ≤2-worker limit binds tests, not the benchmark that spec §3.6 requires for 1/2/4 — if wrong: none
- Task T0.4: Ruling: fix in T0.4 now — pin benchmark workload explicitly in _make_config (env/net classes + algorithm/rollout values), add chunks_per_update column + learner-bound note, add run-validity check (non-zero exit), doc setup line + commit JSON — spec §3.6 needs comparable before/after numbers — if wrong: small script rework later
- Task T0.4: Ruling: T6.3 sources env steps from the global env-step counter (system records) and uses record timestamps — chunk-count formula is invalid once envs emit `active` — if wrong: after-numbers mis-scaled, re-measure
- Task T0.4: Ruling: every task that changes the config schema updates scripts/bench_throughput.py::_make_config and greps scripts/; T1.4 adds a guard unit test that _make_config builds — pinned keys (recurrent_type, gae_lambda, checkpoint.dir) would otherwise silently break the after-measurement — if wrong: none
- Task T0.5: Ruling: strengthen the boundary test now (+ cheap: assert ckpt seat appears in results; wrap loop.close() in try/finally in wrapper) — T2.6/T3.x rework this path and need a real pin — if wrong: none
- Tasks T1.1+T1.2: batched dispatch (BASE cd37dd9) — Ruling: batch two small independent new modules (state utils, PolicyModel) into one implementer + one review with per-task verdicts — saves a dispatch cycle; separate commits keep them separable — if wrong: one combined fix round
- Task T1.3: Ruling: T1.6 adds 3 core tests (mixed-len batched==per-row for WindowAttention; causality hook; native nn.LSTM/GRU reference with num_layers=2 and non-zero s0); T1.4 adds WindowAttentionCore num_layers>=1 guard — cheap, kills surviving mutants, baseline for SP6 fast paths — if wrong: none
- Task T1.4: Ruling: fix now — validate state with two batch sizes (2 and 3); plus cheap minors: wrap raw exceptions (ActionSpec.from_space, env.reset, initial_state, sample) in ConfigError; Core.reset_state empty-container guard; issubclass checks before construction; negative tests for mask-size/state/action-shape/unroll-shape messages — validator's purpose is catching exactly these — if wrong: none
- Task T2.1: Ruling: fix now — warn when interop threads != requested after the except; probe+assert interop==1 in spawned worker test; also monkeypatch cpu_count=8 in auto-threads test (CI runners have 4 vCPU) — silent regression risk from later tasks (T2.2/T6.2) adding torch work before the call — if wrong: none
- Task T2.2: Ruling: cancel weight-queue join only when stop_event is set (workers stopping), otherwise let feeder flush; add regression test "learner exits, live reader drains without hang"; validate payloads in TrajectoryServicer/WeightStore servicer (INVALID_ARGUMENT / drop) + test; bounded decompression (max_length cap + eof check) and npz header size cap + test; plus cheap minors: resume-state contract test, dedupe tree walkers (state.py on top of ipc.py, single _is_namedtuple), module.__dict__ lookup, archive index validation — correctness/safety of core IPC — if wrong: none
- Task T2.6: Ruling: fix all three now (+ cheap minors: chunk independence after buffer reuse test, park(full)->RuntimeError test, row-after-done is episode start check, double _last_weight_sync) — T3.1–T3.4 rewrite these paths and need real guards — if wrong: none
- Controller: Ruling: benchmark "bound" classification via chunks_per_update is meaningless after T2.4 (always == batch_chunks); T6.3/T8.3 must classify bottleneck via learner queue depth (system records) — if wrong: mislabeled column only
- Task T4.1: implementer DONE_WITH_CONCERNS (38591cf); not-gpu 483 passed; Ruling: test-kit name mapping added to rulings for T4.2–T4.6, T7.1–T7.2 — kit built on existing helpers per ALL ruling — if wrong: none; review dispatched
- Task T4.2: parked — normalizer stats updated before loss forward (ratio≠1 at zero lag on early steps) — Ruling: keep spec's "update once per train step, before epochs" (spec §5 block 4 explicit); updating after epochs would give exact parity — revisit in SP6 — if wrong: slightly noisier ratios early in training
- Task T4.3: Ruling: T4.6 must build on T4.3's APPO structure (_evaluate, _autocast(), full_batch, _select_chunks, compute_loss(chunks, batch=)) and NOT replace compute_loss/train_step wholesale as its brief says — keep T4.6's additions (grad_norm, explained_variance, rho stats, lr key, consumed_samples, state_dict) — brief predates T4.3's dedup ruling; wholesale replacement would reintroduce duplicate unroll, extra host copy and lose autocast — if wrong: none
- Task T4.4: Ruling: fix both now + cheap minors: mask width validation (composite too-wide silently accepted), dones warning only for stateful models, error message branching NaN vs illegal action, force done at end of each add_data call, --seq-len IntRange(min=1) — warm start is BC's purpose; silent mask-layout acceptance is a correctness bug — if wrong: none
- Task T4.5: Ruling: fix both now — compare per-head category counts / mask sizes and non-composite n/dim; call validate_config at distributed learner+workers entry and bc/eval CLI (T7.2 keeps it); + cheap minors (explicit gymnasium-order assertion, docstrings on sort_keys/pairs, size/kind mismatch test, assert chunk.action_masks natural order) — removing silent layout misalignment is the task's purpose — if wrong: none
- Task T4.6: Ruling: fix now (distinctive scaler state before save, compare full scaler state_dict; CPU-runnable scaler restore test if torch supports CPU GradScaler) + cheap minors: accumulate metrics on device with one host sync per train_step, average only finite grad_norm + skipped-update count, warn on scaler/kickstart presence mismatch at resume, single-copy deep_cpu_copy, bf16 test must not require a scaler — resume correctness is a spec criterion — if wrong: none
- Ruling: distributed learner resume (run_distributed_learner never passes resume_state) is parked to SP5 (distribution) — out of SP1 scope, local resume is the SP1 deliverable — if wrong: distributed users cannot resume until SP5
- Ruling: explicit resume (resolve_resume, run-dir source) with any unreadable/malformed checkpoint meta in the selected agent dir raises ConfigError naming the path; CheckpointManager scan (coordinator start-up) skips malformed meta (incl. non-object JSON, non-numeric timestamp) with a warning, never AttributeError — explicit user intent fails loud, background index degrades gracefully — if wrong: a user with one corrupt old ckpt must delete it to resume
- Ruling: agent ids must not contain '.' (agent-id validator stricter than checkpoint path component: [A-Za-z0-9_][A-Za-z0-9_-]*) so every agent is addressable by `--set agents.<id>.…` — one unambiguous override grammar beats a quoting syntax — if wrong: users with dotted ids rename them (fails loudly at load)
- Ruling: local-mode learners must be seeded from training.seed with a per-agent stream distinct from worker streams (learner initial weights reproducible) — added to T6.1 fix round since seeding.py is T6.1's — if wrong: minor extra scope
- Ruling: `review/` (incl. repro configs) is a frozen snapshot of the pre-SP1 review and is never edited by SP1 tasks — it documents the old code — if wrong: none (repro configs target old code by design)
- Ruling: distributed `run-workers` role dir includes a sanitized hostname (`<name>-workers-<host>`) so several worker machines can share one run.name on a shared FS — multi-machine is the owner's core use case — if wrong: cosmetic dir naming
- Ruling: resolved config records the BASE effective run name (without the role suffix) — role suffix is derived from the role, so re-running a resolved config reproduces the same layout — if wrong: cosmetic
- Ruling: learner final-snapshot put timeout = SHUTDOWN_GRACE_SEC - 2.0 (5 s), main waits SHUTDOWN_GRACE_SEC (7 s) — spec's 'children gone within 10 s' binds; one constant — if wrong: very slow final train step on huge CPU models misses final snapshot (periodic ckpts remain)
- Ruling: agent ids equal to a global metric namespace (`ratings`, `system`, `episodes`, `train`) are rejected by check_agent_id (ConfigError) — namespaces in WandB/jsonl must be unambiguous — if wrong: users rename such an agent
- Ruling: distributed roles (run-learner/run-workers) have no MetricsHub/WandB — parked to SP5 (distribution: hub aggregates metrics) — if wrong: distributed users only get per-process logs until SP5
- Ruling: `run-learner` returns 130/143 after SIGINT/SIGTERM like `train`/`run-workers`; CLI main maps KeyboardInterrupt before handlers are installed to exit 130 with a one-line message, no traceback — one exit-code contract for all commands (D10) — if wrong: none
- Ruling: adopt pthread_sigmask(SIG_BLOCK,{SIGINT}) around proc.start() + child-side SIG_IGN then SIG_UNBLOCK (CPython resource_tracker pattern) — no lost Ctrl+C, descendants end with SIG_IGN + normal mask — if wrong: revert to SIG_IGN swap
- Ruling: accept the resume-test redesign — my T5.4 LR ruling assumed a shared budget; with a larger second budget LR legitimately rises — the interrupted-then-continued scenario is the real resume use case — if wrong: none (exact schedule checks added)
- Ruling: controller verified the 20-line test-only fix diff directly instead of a separate re-review — trivial, fully read, all three minors addressed — if wrong: final whole-branch review is the net
- Ruling: controller verified the ~115-line fix diff directly (fork_rng/flags restore read in full; new test asserts reviewed) instead of a separate re-review — all 4 items addressed — if wrong: final whole-branch review is the net
- Ruling: T8.3 benchmark doc must state the workload change (turn-based TTT now infers only for the acting seat since T8.1) and compare "до"/"после" on worker scaling shape and learner updates/s, not claim a like-for-like env-steps/s speedup; optionally also report a run at the T8.1 parent commit for the same workload if cheap — honesty over a flattering number — if wrong: none
- Ruling: fix the pre-existing validate_config empty-row gap now (reuse core/seat_info acting_flags/extract_masks/check_masks) — a user env valid for training must pass `colosseum validate` — if wrong: none
- Ruling: controller read the registry.py fix diff in full (shared seat helpers; no weakening; finite-logprob check is a strengthening) — no separate re-review — if wrong: final whole-branch review is the net
- Ruling: replace chain with a combination-lock chain (state-dependent key, wrong action steps back) + gamma=0 negative control; and fix the warnings source now (Launcher teardown closes its queues; tests release launchers deterministically) with a 3x full-suite zero-warning check — evidence of learning and a pristine suite are acceptance criteria — if wrong: none
- Ruling: final fix wave = Important semaphore fix (+pinned test update, feeder-thread regression test, bounded join) + minor 1 (abandoned-reader path, same mechanism) + minor 4 (bc data errors -> Config error) + minor 2 as docstring "unused until SP5" markers (no deletion at the last step) + nits 305/348; park minor 3 (launcher split, SP4/SP6 refactor), minor 5 (harmless), minor 6 (SP5) — fix what is user-visible or cheap and safe; avoid refactors at the end of SP1 — if wrong: launcher stays large until a later SP
- Ruling: park both residuals (no second fix wave per process) and surface them to the user in the acceptance report — narrow windows, loud not silent, 3-line fixes for SP2's first commit — if wrong: rare hang at exit needing a third Ctrl-C; one traceback on an unreadable BC file

## Остаточные пункты после финального ревью (отложены, не блокируют слияние)

- Второй Ctrl+C во время `_release_command_queues` (после восстановления обработчиков сигналов) может пропустить запасной `cancel_join_thread`; если команда мёртвого воркера больше 64 КиБ осталась непрочитанной, выход может зависнуть до третьего Ctrl+C. Исправление — 3 строки (`except BaseException` → `cancel_join_thread` → `raise`).
- `offline_bc`: `torch.load` с `OSError`/`PermissionError` (нечитаемый файл данных) всё ещё печатает traceback вместо одной строки `Config error:`.
- launcher.py (1064+ строк) не разделён; дублирование разворачивания маски в `_reset_mask_row`; финальный чекпоинт распределённого learner может потеряться, если drainer остановился раньше (SP5).

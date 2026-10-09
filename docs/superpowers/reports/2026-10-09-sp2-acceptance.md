# SP2: приёмка по спеке §3 (T8.7)

- Ветка `sp2-game-model`. Начало приёмки: HEAD `b97dfec`. Правки документации во время приёмки: `1311fbb`, `07ae1a5`, `6c6f863`. Отчёт — следующий коммит после `6c6f863`.
- Дата: 2026-10-09.
- Машина: WSL2, 8 ядер (`nproc` = 8), 11 ГБ ОЗУ, без GPU. Python 3.12.3, torch 2.14.1+cpu, `.venv`.
- Шаг 14 (merge) не выполнялся: слияние в `main` ждёт явного одобрения владельца. Сообщение владельцу (шаг 13, вторая половина) отправляет контролёр после финального ревью ветки.

## Итог

**ВСЕ КРИТЕРИИ ВЫПОЛНЕНЫ (ALL PASS).** Три правки документации во время приёмки (`docs:`), шаг 7 после них перезапущен. Владельцу остаются вопросы: проверка team_tag «лучший из двух прогонов», `ratio_mode` при K=128, шумный выбор `unit_trace` (раздел «Открытые пункты»).

| § | Критерий | Результат |
|---|---|---|
| 3.1 | Тесты | PASS: 1243 passed, 27 deselected, 342.7 с, exit 0, предупреждений 0; ruff чистый; дерево чистое; ≤ 2 воркеров во всех тестах; счёт 1243 / 9 / 18 совпадает с README и CLAUDE.md (после правки `1311fbb`) |
| 3.2 | Контрактные тесты | PASS: `tests/contract` 157 passed за 8.2 с; V-trace 33 passed; юнит-файлы с контрактными пунктами 174 passed; у каждого пункта есть зелёный тест |
| 3.3 | «Учится» | PASS: быстрые 6 тестов, 0.3–4.4 с; медленные 7/7 выше порогов (coin_grid 5.63× ≥ 2×; tic_tac_toe 0.917; unit_harvest 1.000; team_tag 0.920 с первого прогона; tron 0.970 / 0.958; predator_prey 1.000 / 0.990; coop_buttons 7.39 / 7.39 ≥ 4.40), обучение 43–149 с, весь файл 11 мин 34 с |
| 3.4 | Эксперимент по юнитам | PASS: 12 прогонов (K ∈ {8, 128} × `ratio_mode` × `unit_trace`) в `docs/benchmarks.md` и JSON (12 строк); дефолт `unit_trace` = `joint` (ruling T8.4), закреплён тестом `test_appo_v2.py` |
| 3.5 | Скорость | PASS: 4848 / 9298 / 13702 шагов сред/с при 1 / 2 / 4 воркерах = 104 % / 98 % / 128 % от SP1, рост монотонный; код горячего пути с замера не менялся |
| 3.6 | Эталонные примеры | PASS: `validate` space_miners и chase — exit 0; smoke-`train` обоих и 19 контрактных тестов — 21 passed, 0 skipped |
| 3.7 | Документация | PASS: Quick start и три команды «Other game structures» дословно в чистом shell — всё exit 0; флаги README = `--help`; ENV_GUIDE покрывает все типы игр; три правки `docs:` |
| 3.8 | Распределённый режим | PASS: 7 тестов distributed/grpc passed; localhost smoke: `workers exit=0`, у лёрнера `ckpt_v78`; асимметричная игра — `exit=1`, `Config error:` с упоминанием SP5 |
| 3.9 | GPU | PASS (проверка списка): 18 `gpu`-тестов, список совпадает с `docs/GPU_CHECKS.md`; без CUDA 18 skipped, 0 failed; **ждёт прогона владельцем на CUDA** |
| 3.10 | Остатки SP1 | PASS: первый коммит после спеки и плана — `7e28f2c` (T0.1); тесты всех трёх пунктов блока 0 есть и зелёные |

---

## §3.1 Тесты: полный быстрый набор, ноль предупреждений, запись только в `tmp_path`, ruff

Предусловие: `git status --porcelain` пуст (HEAD `b97dfec`).

Команда (вывод в рабочий каталог задачи вместо `/tmp`): `timeout 3600 .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw -p no:cacheprovider`.

- `exit=0`. Результат: **1243 passed, 27 deselected in 342.74s (0:05:42)**.
- `grep -E "warnings summary|[0-9]+ warnings?( |$)"` по выводу: ничего — **no warnings**.
- `git status --porcelain --ignored` без `.venv`/кэшей/`.superpowers/`: **clean**. Всего игнорируемых — `.pytest_cache/`, `.ruff_cache/`, `.superpowers/`, `.venv/`; `runs/`, `checkpoints/`, `*.jsonl` нет.
- `.venv/bin/ruff check .`: `All checks passed!`.
- Сбор по маркерам: `not gpu and not slow` — 1243/1270; `slow` — 9/1270; `gpu` — 18/1270 (1243 + 9 + 18 = 1270).
  - Было в README и CLAUDE.md: 1237 (до FIX-2, который добавил 6 тестов). Исправлено на 1243 в `1311fbb`; 9 slow и 18 GPU уже совпадали.
- `grep -rln "num_workers.*[3-9]\|num_workers=[3-9]" tests/` нашёл 6 файлов. Каждое попадание прочитано, все ложные:
  - `num_workers: 1` или `2`, а цифра 3–9 стоит дальше в строке (`envs_per_worker: 4`, `chunk_length: 8`): `tests/game_helpers.py:1045`, `tests/integration/test_sp2_pipelines.py` (8 строк), `tests/integration/test_sp2_metrics_outputs.py:56`, `tests/integration/test_sp2_distributed_e2e.py:48`;
  - `SubprocessVectorEnv(_units, num_envs=3, num_workers=2)` — 2 процесса: `tests/integration/test_game_subproc_vector_env.py:74`;
  - `"rollout.num_workers": 8` — только разбор `--set` в конфиг, процессы не запускаются: `tests/unit/test_sp2_config_overrides.py:168`.
  - Итог: ни один тест не запускает больше 2 воркер-процессов.

**PASS.**

## §3.2 Контрактные тесты

Команды:
- `.venv/bin/python -m pytest tests/contract -v -rA` → **157 passed in 8.18s**;
- `.venv/bin/python -m pytest tests/unit -v -k "vtrace"` → **33 passed, 973 deselected in 15.05s**;
- дополнительно (часть пунктов раздела 6 после оверлея T7.3 лежит в `tests/unit`, а не в `tests/contract`): `pytest tests/unit/test_match_runner.py tests/unit/test_episode_tracker.py tests/unit/test_sp2_matchmaker.py tests/unit/test_sp2_eval_report.py tests/unit/test_team_outcomes.py tests/unit/test_sp2_coordinator.py tests/unit/test_sp2_eval_schedule.py tests/integration/test_sp2_eval_cli.py -v` → **174 passed in 2.89s**.

Все перечисленные ниже тесты — PASSED в этих прогонах.

| Пункт | Тест (node id) |
|---|---|
| Лёрнер воспроизводит совместные и per-unit log-prob воркера при нулевом лаге, 4 ядра, чанки с `boot`, `pad`, обрывом и выбыванием | `tests/contract/test_learner_reproduces_worker.py::test_learner_reproduces_worker_log_probs_at_zero_lag[discrete-none]`, `[discrete-lstm]`, `[discrete-gru]`, `[discrete-attention]`, `[units-none]`, `[units-lstm]`, `[units-gru]`, `[units-attention]` (тест требует слоты T/B/R/P в чанках, терминальный ACT выбывшего места и BOOT обрыва для мёртвого сокомандника); негативный контроль `::test_mutated_chunks_are_detected[none/lstm/gru/attention]`; per-unit loss `::test_per_unit_policy_loss_and_mean_valid_entropy_under_random_unit_masks[none-raw/none-normalized/lstm-raw/lstm-normalized]`, `::test_per_unit_policy_loss_off_policy_under_random_unit_masks[...]` (4 варианта) |
| V-trace с `boot` совпадает с эталоном: нулевой лаг = GAE(λ) на `act`; `boot` обрывает трассу; `terminal` даёт 0; NaN на `pad` не доходит до таргетов | `tests/unit/test_vtrace_slots.py::test_matches_the_slot_reference[*]` (24 варианта), `::test_zero_lag_equals_gae_over_act_slots[1.0/0.95/0.5/0.0]`, `::test_boot_cuts_the_trace_and_terminal_bootstraps_zero`, `::test_nan_in_unused_slots_never_reaches_the_targets`, `::test_random_patterns_respect_the_slot_invariants`; `tests/unit/test_appo_v2.py::test_unit_trace_sets_the_scalar_log_rho_of_vtrace` |
| Ни одна награда не теряется, кроме мест, ни разу не ходивших в эпизоде (они считаются в `dropped_reward_episodes`) | `tests/contract/test_chunk_v2_rules.py::test_reward_accounting_matches_the_env_except_documented_drops`, `::test_a_seat_that_never_acts_drops_its_reward_and_is_counted`, `::test_rewards_before_the_first_act_are_carried_into_it` |
| Bootstrap из `final_obs` считает сеть лёрнера | `tests/contract/test_learner_reproduces_worker.py::test_bootstrap_uses_the_learners_current_values` |
| `uint8` доходит до модели; `global_state` доходит только до value | `tests/contract/test_learner_reproduces_worker.py::test_uint8_observations_reach_the_model_on_both_sides`, `::test_global_state_changes_values_but_not_log_probs[none/lstm/gru/attention]` (R16); `tests/contract/test_chunk_v2_rules.py::test_global_state_is_recorded_for_acts_and_the_final_boot` |
| Каждая ошибка контракта среды блока 1, включая `max_idle_steps` | через `MatchRunner` (R14): `tests/unit/test_match_runner.py::test_episode_tracker_errors_surface_through_the_runner_with_its_context[...]` — 19 вариантов: `reset ends the episode`, `reset gives rewards`, `reset terminates seats`, `empty seat in acting`, `eliminated seat in acting`, `acting seat without observation`, `observation for an empty seat`, `observation does not fit the role's space`, `mask does not fit the role's space`, `reward for an empty seat`, `reward after elimination`, `terminated again`, `terminated for an empty seat`, `episode_over with acting seats`, `truncation without final_obs of a live seat`, `truncation without the final global_state the role declares`, `acting seat without the declared global_state`, `outcome not by the layout's teams`, `idle overflow`; `::test_env_contract_errors_carry_the_worker_and_env_context`; на уровне трекера `tests/unit/test_episode_tracker.py::test_reset_rules[*]`, `::test_step_result_rules[*]`, `::test_malformed_fields_are_contract_errors[*]`, `::test_global_state_rules`, `::test_max_idle_steps`, `::test_reset_without_acting_seats_counts_as_the_first_idle_step`; доставка `env.max_idle_steps` до воркера — `tests/unit/test_sp2_launcher_checkpoints.py::test_worker_main_passes_env_max_idle_steps_and_lineups_to_the_rollout_worker` (зелёный в §3.1) |
| Жизненный цикл места: награды ждущим, награда в шаге выбывания, мёртвый сокомандник получает награду в конце, пустые места, никогда не ходившее место | `tests/contract/test_chunk_v2_rules.py::test_rewards_before_the_first_act_are_carried_into_it`, `::test_elimination_at_s_minus_2_is_terminal_with_its_step_reward_then_a_pad`, `::test_a_seat_eliminated_in_the_truncation_step_is_terminal_and_gets_no_boot` (R15), `::test_dead_teammate_keeps_its_open_act_until_the_final_reward`, `::test_a_seat_that_never_acts_drops_its_reward_and_is_counted`; `tests/unit/test_episode_tracker.py::test_empty_seats_get_nothing`, `::test_reward_in_the_elimination_step_is_allowed_and_later_ones_are_not`, `::test_dead_teammate_keeps_rewards_and_stays_live`; `tests/unit/test_match_runner.py::test_rewards_of_the_elimination_step_come_before_on_terminated` |
| Структура чанка по правилам записи (маски и наблюдения `boot`/`pad`) | `tests/contract/test_chunk_v2_rules.py::test_mid_episode_boundary_writes_a_boot_and_continues_in_the_same_buffer`, `::test_episode_end_by_rules_marks_the_open_act_terminal_and_pads`, `::test_chunks_span_episodes_and_end_with_a_boot_mid_episode`, `::test_truncation_with_the_open_act_at_s_minus_2_writes_the_final_boot_and_seals`, `::test_truncation_with_the_open_act_at_s_minus_3_boots_then_pads_at_the_next_act` |
| Граница чанка в середине эпизода при припаркованном буфере продолжает эпизод в сброшенном буфере места (LSTM) | `tests/contract/test_chunk_v2_rules.py::test_mid_episode_boundary_with_a_parked_buffer_continues_in_the_seat_buffer[lstm]`, `[gru]`; парковка без потерь — `tests/contract/test_rollout_loop_lineups.py::test_lineup_changes_park_buffers_without_losing_or_duplicating_acts` |
| Инварианты матчмейкинга: роли, однородность команд при `self`, доли вариантов ±5 %, перестановки, каждый агент владеет средами, асимметрия при `mode: self_play` | `tests/unit/test_sp2_matchmaker.py::test_2v2_teams_are_homogeneous_with_teammates_self`, `::test_layout_weights_are_followed_within_5_percent`, `::test_seats_are_balanced_within_5_percent[*]`, `::test_permute_seats_keeps_team_and_role_structure`, `::test_one_hunter_vs_three_prey_under_self_play_mode`, `::test_layouts_are_restricted_to_those_with_a_seat_of_the_owners_roles`, `::test_seat_of_a_role_the_core_does_not_play_goes_to_any_latest_player_of_it`; `tests/unit/test_sp2_coordinator.py::test_owner_rotates_with_global_env_index_and_round`, `::test_asymmetric_agents_both_own_envs_and_fill_each_others_teams` |
| `MatchResult` по командам; отчёт eval для каждого типа исхода | `tests/unit/test_match_runner.py::test_events_follow_the_contract_order_and_the_result_is_per_team`; `tests/unit/test_sp2_eval_report.py::test_wdl_pairs_with_sides_and_reversed_rows`, `::test_rank_layout_reports_mean_rank_first_places_and_higher_table`, `::test_score_layout_reports_compositions_and_cross_play`, `::test_one_agent_in_a_wdl_layout_reports_per_seat`, `::test_one_agent_in_a_rank_layout_reports_per_seat`; `tests/contract/test_sp2_eval_engine.py::test_ffa_layouts_with_eliminations_report_ranks` |
| Демо-игры идут через настоящий `MatchRunner` без ошибок контракта | `tests/contract/test_demo_coin_grid.py::test_random_matches_run_through_the_match_runner` и то же в `test_demo_coop_buttons.py`, `test_demo_predator_prey.py`, `test_demo_team_tag.py`, `test_demo_tron.py`, `test_demo_unit_harvest.py`; `tests/contract/test_reference_examples.py::test_chase_random_matches`, `::test_space_miners_random_matches` |

Отличие от брифа: бриф ищет тесты T5.1, T3.3, T1.5 в `tests/contract`, а они лежат в `tests/unit` (`test_sp2_matchmaker.py`, `test_match_runner.py`, `test_episode_tracker.py`). Поэтому эти файлы прогнаны третьей командой.

**PASS.**

## §3.3 «Учится»

Быстрые: `.venv/bin/python -m pytest tests/learning -m "not slow" -v --durations=0` → **6 passed, 7 deselected in 11.47s**.

| Тест | Время |
|---|---|
| `tests/learning/test_sp2_fast_learning.py::test_short_chain_needs_discounting` | 4.42 с |
| `tests/learning/test_sp2_units_coop_learning.py::test_coop_bandit_two_agents_learn_their_joint_answer` | 3.17 с |
| `tests/learning/test_sp2_bc_learns.py::test_bc_imitates_a_masked_scripted_expert` | 1.29 с |
| `tests/learning/test_sp2_fast_learning.py::test_short_chain_is_solved_in_seconds` | 1.15 с |
| `tests/learning/test_sp2_fast_learning.py::test_contextual_bandit_is_solved_in_seconds` | 0.92 с |
| `tests/learning/test_sp2_units_coop_learning.py::test_units_bandit_is_solved_with_per_unit_ratios` | 0.26 с |

Все меньше 20 с. `nproc` = 8; `uptime` перед медленными: load average 0.97 / 1.58 / 2.36.

Медленные: `.venv/bin/python -m pytest tests/learning/test_demo_learning_slow.py -m slow -v -s` → **7 passed in 693.86s (0:11:33)**, exit 0. Числа и сравнение с T8.3 — в разделе «Замеры «учится» (T8.3)», подраздел «Прогон приёмки (T8.7, шаг 3)». Все значения не ниже порогов. Ни одно не ниже заметно, чем в T8.3 (seed 0). team_tag прошёл с первого прогона, повтор не понадобился.

Cross-play `coop_buttons`: перед строкой `[coop_buttons]` тест напечатал отчёт eval (`== layout coop2 (score, 100 matches)`, `coop_a+coop_b n=100 mean_score 7.390 [7.208, 7.572] (cross-play)`, средние возвраты по ролям). Строка `[coop_buttons]` печатает JSON `cross_play` с записью `coop_a+coop_b` (n = 2441).

**PASS.**

## §3.4 Эксперимент по юнитам

- `sed -n '/## Эксперимент по юнитам/,/## Размер/p' docs/benchmarks.md`: раздел «Эксперимент по юнитам (SP2, критерий 4)». В нём таблица из 12 строк seed 0 (K ∈ {8, 128} × `ratio_mode` ∈ {joint, per_unit} × `unit_trace` ∈ {joint, geo_mean, none}) и 3 строки seed 1 (K=128 `per_unit`). Колонки: `win_vs_random`, `share_vs_scripted`, `clip_fraction`, `clip_fraction_joint`, `ess` и другие диагностики. Есть выводы и решение.
- `docs/benchmarks/units-experiment.json`: `12 joint` (12 строк, `recommendation` = `joint`).
- Решение в отчёте: раздел «Решения T8.3–T8.4», запись «Ruling: `unit_trace: auto` с `Units` = **`joint`**», плюс строка журнала «Ruling (T8.4, coded pre-registered rule) …». `grep -n "Ruling: unit_trace"` из брифа ничего не находит: в черновике после «Ruling:» стоит обратная кавычка. На суть это не влияет.
- Дефолт в коде совпадает с решением. `resolve_modes` в `src/colosseum/algorithms/appo.py` даёт `unit_trace` = `joint` для `auto`. Закрепляющий тест (R11, P17): `tests/unit/test_appo_v2.py::test_modes_resolve_auto_and_collapse_for_one_decider`, строка 59: `assert resolve_modes(AlgorithmConfig(), units) == ("per_unit", "joint", "mean_valid")`. Тест зелёный в §3.1. `test_unit_trace_default.py` не создавался (R11).

**PASS.**

## §3.5 Скорость

- `git log --oneline -1 -- docs/benchmarks/after-sp2.json` → `f5e8a4b docs: SP2 units experiment, throughput after SP2, global_state size; unit_trace default ruling` (замер на коде `a4f1ad1`).
- `git diff --stat f5e8a4b..HEAD -- src/colosseum/{worker,learner,algorithms,envs,networks,core}`: 2 файла, только docstring: `core/errors.py` (текст `EnvContractError`) и `core/specs.py` (docstring модуля). Остальные изменения `src/` с замера: `cli.py` и `utils/process.py` (FIX-2) — обработка Ctrl-C при запуске, горячий путь не затронут. Перезамер не нужен.
- «После SP2» (`scripts/bench_throughput.py --workers 1 2 4 --duration 60 --warmup 15`, tic-tac-toe):

| Воркеры | Шаги сред/с | Доля от SP1 | Порог 90 % |
|---|---|---|---|
| 1 | 4848 | 104 % (4641) | ≥ 4177 |
| 2 | 9298 | 98 % (9444) | ≥ 8500 |
| 4 | 13702 | 128 % (10664) | ≥ 9598 |

Рост 4848 → 9298 → 13702 строго монотонный. Отклонение замера T8.4: load average перед замером был 1.86 при требовании < 1. Это может только занизить числа.

**PASS.**

## §3.6 Эталонные примеры

- `.venv/bin/colosseum validate -c configs/examples/space_miners.yaml` → `Config is valid.`, `exit=0`.
- `.venv/bin/colosseum validate -c configs/examples/chase.yaml` → `Config is valid.`, `exit=0`.
- `pytest tests/integration/test_reference_examples_smoke.py tests/contract/test_reference_examples.py -v -rs` → **21 passed in 15.03s**, 0 skipped (Box2D установлен):
  - smoke-обучение: `tests/integration/test_reference_examples_smoke.py::test_chase_trains_briefly`, `::test_space_miners_trains_briefly`;
  - 19 контрактных тестов `tests/contract/test_reference_examples.py` (дерево действий chase, маски и исход space_miners, пресеты, ошибки параметров, `validate`, случайные партии через `MatchRunner`).

**PASS.**

## §3.7 Документация работает дословно

### Quick start и «Other game structures» (дословно, из корня репо, в свежем `env -i HOME=$HOME PATH=/usr/bin:/bin bash --noprofile --norc`)

`git clone` и `scripts/setup-dev.sh` не запускались: `.venv` уже создан этим скриптом. Блок брифа начинается с `source .venv/bin/activate`. Время каждой команды измерялось обёрткой.

| Команда | exit (прогон 1 / прогон 2) | Время |
|---|---|---|
| `colosseum validate -c configs/examples/tic_tac_toe.yaml` | 0 / 0 | 1.7 с |
| `colosseum train ... --set run.name=ttt-quickstart` | 0 / 0 | 57.4 / 52.4 с |
| `NEW=…; OLD=…; colosseum eval ... -a new=$NEW -a old=$OLD --num-matches 200 --output eval.json` | 0 / 0 | 2.6 / 2.5 с |
| `colosseum train ... run.name=ttt-continued, resume_from=runs/ttt-quickstart, total_timesteps=1200000` | 0 / 0 | 54.0 / 52.7 с |
| `colosseum train -c configs/examples/tron.yaml --set run.name=tron-doc --set training.total_timesteps=60000` | 0 / 0 | 21.1 / 21.4 с |
| то же `predator_prey` | 0 / 0 | 16.4 / 16.8 с |
| то же `coop_buttons` | 0 / 0 | 14.6 / 14.4 с |
| `colosseum eval -c configs/examples/tron.yaml -a a=<новый> -a b=<старый> --layout 4p --num-matches 40` | 0 / 0 | 2.7 / 2.6 с |

Строк `FAILED` нет.

- Время quickstart (57 и 52 с) README описывал как «about a minute (59–64 s over three seeds)». Цифры вне диапазона, поэтому README исправлен (см. ниже). Доля побед greedy против случайного игрока проверена медленным тестом шага 3 на том же конфиге: 0.917, в диапазоне README «86–94%».
- `eval.json` создан (ключи `agents`, `num_matches`, `deterministic`, `ci_level`, `layouts`). Отчёт крестиков-ноликов содержит W/D/L, win rate и score с 95 % Wilson CI и разбивку по сторонам.
  - Прогон 2: `new` (`ckpt_v2412`) против `old` (`ckpt_v1600`): 63/131/6, win_rate 0.315 [0.255, 0.382], score 0.642.
  - Прогон 1: 95/101/4, по сторонам `as team 0: W=95 D=4 L=1`, `as team 1: W=0 D=97 L=3`.
- Отчёт tron `4p` содержит mean rank с CI, долю первых мест и попарную матрицу.
  - Прогон 2: a 1.869 [1.671, 2.067], первые места 0.412; b 3.131, 0.075.
  - Прогон 1: `a above b: 0.900`.
- Продолжение:
  - лог: `Resume [agent_0]: …/ckpt_v2304 (policy_version 2304)`, `env-step counter continues from 600832` (прогон 2: `ckpt_v2412`, 606496);
  - первый сохранённый чекпоинт продолжения — `ckpt_v2400` (прогон 2: `ckpt_v2500`). Он больше последнего чекпоинта quickstart (`ckpt_v2304` / `ckpt_v2412`).
  - В пуле FIFO к концу остались `ckpt_v3800…ckpt_v4644` (прогон 2: `ckpt_v3900…ckpt_v4752`), поэтому первую версию видно только в `main.log`.

### Разделы с заглушками (`my_game/…`, `path/to/demos/`, `cfg.yaml`)

Каждый флаг README и раздела «Commands» CLAUDE.md сверен с `colosseum <cmd> --help`, все существуют:
- `validate`/`train`: `-c`, `--set`;
- `eval`: `-c`, `-a`, `--layout`, `--num-matches`, `--num-envs`, `--output`, `--deterministic`, `--seed`;
- `bc`: `-c`, `--agent`, `--data`, `--output`, `--epochs`, `--batch-size`, `--lr`, `--seq-len`;
- `serve-weight-store`: `--port`;
- `run-learner`: `-c`, `--agent`, `--traj-port`, `--weight-store`, `--set`;
- `run-workers`: `-c`, `--weight-store`, `-l`, `--set`.

### `docs/ENV_GUIDE.md`

Первая таблица (строки 9–19) перечисляет каждый тип игры спеки с демо-средой и конфигом:
- соло — `coin_grid`;
- 1v1 пошаговая — `tic_tac_toe`;
- бот с юнитами — `unit_harvest`;
- команда на команду — `team_tag`;
- FFA с выбыванием — `tron`;
- 1 vs N, асимметрия — `predator_prey`;
- кооператив — `coop_buttons`;
- плюс эталоны `space_miners` и `composite_action`.

Одновременные ходы описаны в §3 (строка 134: «Одновременные ходы: все живые места…»; демо — `unit_harvest`, `tron`). Разное число игроков — в §2 (строка 74, `GameSpec.symmetric([2, 3, 4])`, `tron`).

### README и CLAUDE.md против наблюдений шагов 1–6

Совпадают:
- дефолт `unit_trace` (`auto` = `joint` для любых действий; README, раздел Configuration reference; CLAUDE.md, раздел Algorithm);
- `ratio_mode` (`auto` = `per_unit` с `Units`);
- таблица скорости 4848 / 9298 / 13702;
- ограничения распределённого режима (только игры, где каждый агент играет все роли; без лиги, рейтингов, `metrics.jsonl`; совпадает с отказом на шаге 8);
- GPU не запускался;
- 9 slow и 18 GPU тестов;
- пороги медленных тестов;
- team_tag «лучший из двух».

Расхождения, исправленные коммитами `docs:`:
1. `1311fbb`:
   - счёт быстрых тестов 1237 → 1243 в README («Tests») и CLAUDE.md («Implementation Status»);
   - README «Process lifecycle»: новый пункт — Ctrl-C всегда даёт 130 и `Interrupted`/`Received SIGINT` без traceback, в том числе при запуске; потерянный при импорте Ctrl-C: `train` и распределённые роли останавливаются как обычно, `eval`/`bc`/`validate` доделывают работу и выходят с 130; `serve-weight-store` выходит с 0 (FIX-2);
   - `docs/ENV_GUIDE.md`: счётчик `dropped_reward_episodes` «нигде не выводится» → итог виден один раз, в последней строке INFO воркера `Worker <i>: finished. {...}` в `logs/worker-<i>.log`, в `metrics.jsonl` его нет (замечание повторного ревью T8.6; проверено по `src/colosseum/worker/rollout_worker.py:170`).
2. `07ae1a5`: время quickstart «59–64 s over three seeds» → «54–64 s in five measured runs» (после прогона 1 шага 7 и шага 3).
3. `6c6f863`: после повторного прогона шага 7 (52 с) → «52–64 s over the six measured runs».
   - Шесть прогонов: три сида T8.3, шаг 3, прогоны 1 и 2 шага 7.
   - Третий прогон шага 7 не делался: правка только расширяет диапазон на число, полученное в прогоне 2, который уже шёл на тексте `07ae1a5`, отличающемся только этим числом.

Уборка: `rm -rf runs eval.json` (после каждого прогона).

**PASS.**

## §3.8 Распределённый режим

- `.venv/bin/python -m pytest tests/integration -v -k "distributed or grpc"` → **7 passed, 87 deselected in 7.21s**:
  - `tests/integration/test_sp2_distributed_e2e.py::test_distributed_grpc_pipeline`;
  - `tests/integration/test_sp2_grpc.py::test_weight_store_roundtrip_and_adapters`, `::test_trajectory_transport_carries_chunk_v2_trees`, `::test_servicer_rejects_a_chunk_without_obs`, `::test_trajectory_sink_tolerates_a_dead_learner`, `::test_trajectory_sink_warns_when_the_learner_rejects_a_chunk`, `::test_weight_store_rejects_non_array_weights`;
  - юнит-тесты ролей (`tests/unit/test_sp2_distributed_roles.py`) зелёные в §3.1.
- Localhost smoke — команды брифа; `run.dir` = рабочий каталог задачи вместо `/tmp/sp2-acceptance`:
  - `run-workers` (tic-tac-toe, `total_timesteps=20000`): **`workers exit=0`** за 9 с;
  - `kill $LR $WS` → лёрнер 143 (SIGTERM, как положено);
  - созданы два run dir:
    - `tic_tac_toe-20261009-184404-learner-agent_0`: `config.resolved.yaml`, `logs/learner-agent_0.log`, `checkpoints/agent_0/ckpt_v78`;
    - `tic_tac_toe-20261009-184414-workers-DESKTOP-UMCOKC5`: `config.resolved.yaml`, `logs/worker-0.log`, `worker-1.log`, `workers-main.log` (и пустой `checkpoints/`, как в SP1).
- Асимметричная игра при остановленном weight store: `colosseum run-workers -c configs/examples/predator_prey.yaml … -l hunter=localhost:50052` → **`asymmetric exit=1`**, одна строка:
  `Config error: distributed mode supports only games where every agent plays every role of the enabled layouts; agent 'hunter' does not play ['prey']. Asymmetric agents and leagues across machines come with SP5 (train such configs with 'train' on one machine)`.
- Уборка: каталог прогона удалён.

**PASS.**

## §3.9 GPU

- `pytest -m gpu --collect-only -q`: **18** тестов. Сравнение с `grep -oE 'tests/[^ |`]+::[^ |`]+' docs/GPU_CHECKS.md | sort` — **`GPU list matches`**, diff пуст. Состав:
  - `tests/unit/test_appo_v2.py` ×10: AMP с Dict + `uint8` + `global_state` fp16/bf16, AMP с `Units` × {none, lstm} × {fp16, bf16}, GradScaler, `state_dict`, `pin_memory` × 2;
  - `tests/unit/test_kickstart_v2.py` ×2;
  - `tests/unit/test_sp2_bc_trainer.py` ×2;
  - `tests/unit/test_state.py` ×4.
- `pytest -m gpu -q -rs`: **18 skipped, 1252 deselected**, 0 failed, причина — «requires a CUDA device».
- `docs/GPU_CHECKS.md` содержит инструкцию: `scripts/setup-dev.sh --gpu`, затем `.venv/bin/python -m pytest -m gpu -v`.

**Эти 18 тестов владелец ещё должен прогнать на машине с CUDA** (`docs/GPU_CHECKS.md`). Приёмка проверяет, что CUDA-пути помечены и перечислены (формулировка критерия 9).

**PASS** (в рамках критерия 9).

## §3.10 Остатки SP1

`git log --oneline --reverse main..sp2-game-model | head -8`:
- `32ff269`, `9fcf519`, `7b9a438`, `f5d6b23`, `f914d0c` — спека и план (`docs:`);
- **`7e28f2c fix: SP1 residuals: detach command queues on a second Ctrl+C, one-line bc data errors, queue helpers in core/ipc`** (T0.1) — первый кодовый коммит ветки;
- затем `cb55b51`, `52cba9e` (T1.1, T1.2).

Пункты блока 0 и их тесты (все зелёные в §3.1):
1. Второй Ctrl+C во время освобождения очередей команд:
   - `tests/unit/test_ipc_queue_helpers.py::test_interrupt_while_draining_detaches_every_queue_and_reraises`;
   - `::test_interrupt_while_joining_keeps_joined_queues_and_detaches_the_rest`;
   - `::test_release_reads_back_an_unread_large_command_and_joins_the_feeder`.
2. `colosseum bc` с нечитаемым файлом данных — одна строка `Config error:`:
   - `tests/integration/test_sp2_bc_cli.py::test_a_data_file_that_cannot_be_opened_is_a_one_line_config_error` (PermissionError из `torch.load`);
   - `::test_bad_data_is_a_one_line_config_error[unreadable]`;
   - `tests/unit/test_sp2_bc_trainer.py::test_unreadable_files_are_data_errors`.
   - SP1-тест T0.1 (`tests/integration/test_bc_cli.py`) перенесён сюда в T7.3.
3. Хелперы очередей в `core/ipc.py` под публичными именами (поправка R6; бриф грепает `_QueueReader`, реальное имя — `QueueReader`, P17):
   - `src/colosseum/core/ipc.py:237 class QueueReader`, `:356 def release_command_queues`, `:420 def queue_depths`;
   - лаунчер импортирует их: `src/colosseum/launcher.py:34 from colosseum.core.ipc import QueueReader, SharedCounter, queue_depths, release_command_queues`;
   - тест: `tests/unit/test_ipc_queue_helpers.py::test_helpers_moved_out_of_the_launcher` (старых `_QueueReader`/`_release_command_queues`/`_queue_depths` в лаунчере нет), `::test_queue_reader_drains_a_plain_queue_synchronously`, `::test_queue_depths_reports_sizes_and_unknown`;
   - дублирование `registry._reset_mask_row` исчезло вместе со старым registry в T7.3 (`grep -rn _reset_mask_row src` пуст).

Пункт SP5 (финальный чекпоинт распределённого лёрнера) по спеке вне блока 0, он в «Открытых пунктах».

**PASS.**

---

## Замеры «учится» (T8.3)

Машина: 8 ядер, 11 ГБ ОЗУ, без GPU (WSL2). Каждый тест: `colosseum train` на конфиге примера, 2 воркера, затем greedy против случайных легальных игроков через `play_lineups` (`tests/learning/test_demo_learning_slow.py`). Обучение запускалось строго последовательно, по одному прогону. Значения — итоговые, на конфигах после калибровки (см. «Решения»).

| Тест | Порог | seed 0 | seed 1 | seed 2 | Обучение, с (seed 0) | Шаги сред/с | `total_timesteps` |
|---|---|---|---|---|---|---|---|
| coin_grid | счёт ≥ 2 × случайный | 10.55 / 2.35 = 4.50× | 11.66 / 2.35 = 4.97× | 14.97 / 2.35 = 6.38× | 48 | 6 900 | 300 000 |
| tic_tac_toe | ≥ 0.80 побед | 0.863 | 0.930 | 0.935 | 64 | 10 100 | 600 000 |
| unit_harvest | ≥ 0.80 побед | 1.000 | 1.000 | 1.000 | 164 | 1 620 | 260 000 |
| team_tag (size 6, view_radius 3; лучший из двух прогонов) | ≥ 0.80 побед команды | 0.690 ✗; повтор 0.935 | 0.935 | 0.890 | 208; повтор 178 | 2 000–2 360 | 400 000 |
| tron 2p / 4p | ≥ 0.80 / ≥ 0.50 первых мест | 0.945 / 0.930 | 0.975 / 0.930 | 0.980 / 0.963 | 142 | 2 900 | 400 000 |
| predator_prey охотник / жертвы | ≥ 0.70 / ≥ 0.70 | 1.000 / 0.965 | 1.000 / 0.995 | 1.000 / 0.985 | 94 | 4 480 | 400 000 |
| coop_buttons A / B | ≥ max(5 × случайный, 0.6 × оракул) | 7.39 / 7.39 (≥ 4.40) | 7.39 / 7.39 (≥ 4.40) | 7.39 / 7.39 (≥ 4.40) | 91 | 4 590 | 400 000 |

Пояснения:
- «Обучение» — время всей команды `colosseum train` (запуск, обучение, финальные чекпойнты). На seeds 1–2 разброс: tic_tac_toe 59–64 с, unit_harvest 143–169 с, team_tag 175–188 с, tron 127–131 с, predator_prey 81–82 с, coop_buttons 78–80 с, coin_grid 42–46 с. Весь файл медленных тестов — 12–16 минут.
- **Обучение асинхронное: seed не воспроизводит прогон** (порядок чанков и версии весов зависят от планировщика ОС). Поэтому team_tag, где запас мал, проверен на 9 прогонах выбранного конфига: 0.690, 0.935, 0.890, 0.910, 0.835, 0.950, 0.935 (seeds 0–5 и повтор seed 0), затем при проверке теста с повтором 0.760 и 0.945. 7 из 9 проходят; провалившиеся прогоны ушли в пассивную политику. Порог в таком прогоне не выполнен. Поэтому тест team_tag принимает лучший из двух независимых прогонов (см. «Решения» и «Отклонения»).
- predator_prey: случайный охотник против случайных жертв выигрывает 0.550 (баланс игры не нарушен).
- coop_buttons: случайная пара набирает 0.00, оракул 7.34 (`measure_baselines(episodes=300)`), порог 4.40. Обе однородные команды дают ровно 7.39 на всех seeds: игра детерминирована по seed сброса, а greedy-политика, проходящая к кнопкам кратчайшим путём, даёт один и тот же счёт на 100 фиксированных eval-seeds (смешанная команда A+B — тоже 7.39 [7.21, 7.57], возвраты по местам различаются: 8.17 / 8.12). В `ratings.json` есть перекрёстная запись `coop_a+coop_b` (n = 2499 / 2267 / 2616 на seeds 0 / 1 / 2).

Быстрые тесты: `tests/learning/test_sp2_units_coop_learning.py`.
- Units-бандит: 1.2 с, решён за 10 обновлений (необученный greedy 0.53, случайная игра 0.24, `deciders_valid_mean` 4.6). Тест показывает, что действие `Units` обучается целиком с `ratio_mode: per_unit`. Распределение заслуги по юнитам он **не** проверяет: joint тоже его решает, головы юнитов общие.
- Кооперативный бандит: 3.3 с, решён за 130 обновлений (случайная игра 0.08).

### Прогон приёмки (T8.7, шаг 3)

`.venv/bin/python -m pytest tests/learning/test_demo_learning_slow.py -m slow -v -s`, HEAD `1311fbb` (код = `b97dfec`). Load average перед запуском 0.97. Результат: **7 passed in 693.86s (11 мин 34 с)**. Обучение шло последовательно, других нагрузок на машине не было.

| Тест | Порог | T8.7 | T8.3, seed 0 | Обучение, с | Шаги сред/с |
|---|---|---|---|---|---|
| coin_grid | счёт ≥ 2 × случайный | 13.210 / 2.345 = 5.63× | 4.50× | 43 | 8 207 |
| tic_tac_toe | ≥ 0.80 побед | 0.917 | 0.863 | 54 | 11 890 |
| unit_harvest | ≥ 0.80 побед | 1.000 | 1.000 | 135 | 1 981 |
| team_tag | ≥ 0.80 побед команды | 0.920 (первый прогон, повтора нет) | 0.690 ✗; повтор 0.935 | 149 | 2 787 |
| tron 2p / 4p | ≥ 0.80 / ≥ 0.50 первых мест | 0.970 / 0.958 | 0.945 / 0.930 | 127 | 3 253 |
| predator_prey охотник / жертвы | ≥ 0.70 / ≥ 0.70 | 1.000 / 0.990 (случайный против случайных 0.550) | 1.000 / 0.965 | 80 | 5 204 |
| coop_buttons A / B | ≥ 4.404 (max(5 × 0.000, 0.6 × 7.340)) | 7.390 / 7.390 | 7.39 / 7.39 | 79 | 5 213 |

- Все значения не ниже порогов. Ни одно не ниже значения T8.3 (seed 0).
- Обучение каждого теста — 43–149 с, меньше 3 минут.
- Файл шёл 11.6 мин, немного быстрее, чем «12–16 минут» в T8.3. Шаги сред/с выше, чем в T8.3 (машина была свободна).
- `cross_play` в `ratings.json` прогона coop_buttons:
  - `coop_a+coop_a` — mean 4.81, n 3729;
  - `coop_a+coop_b` — mean 3.53, n 2441;
  - `coop_b+coop_b` — mean 5.10, n 3862.
- Отчёт eval смешанной команды: `coop_a+coop_b n=100 mean_score 7.390 [7.208, 7.572] (cross-play)`.

## Эксперимент по юнитам и скорость (T8.4)

Подробности, полная таблица и сырые данные — в `docs/benchmarks.md` (разделы «После SP2», «Эксперимент по юнитам (SP2, критерий 4)», «Размер `global_state` (team_tag)») и `docs/benchmarks/*.json`.

**Критерий 4 (эксперимент по юнитам) — выполнен.** `scripts/units_experiment.py`: `unit_harvest` при K=8 и K=128, `ratio_mode` × `unit_trace` (12 прогонов), 340 000 шагов сред на прогон, seed 0. Прогоны шли по одному: один прогон занимает ≈ 7 ядер. 75 минут. Все 12 прогонов завершились с кодом 0. Отдельно три строки K=128 `per_unit` повторены с seed 1 (26 минут), потому что прогоны одной конфигурации расходятся сильнее порога правила.

| K=128, `ratio_mode: per_unit` | `share_vs_scripted` seed 0 / seed 1 / среднее | `win_vs_random` seed 0 / seed 1 | `ess` | `clip_fraction` / `clip_fraction_joint` |
|---|---|---|---|---|
| `unit_trace: joint` | 0.236 / 0.176 / 0.206 | 0.79 / 0.79 | 0.53 / 0.40 | 0.13–0.19 / 0.68–0.78 |
| `unit_trace: geo_mean` | 0.044 / 0.143 / 0.093 (отдельный прогон на 340k шагов: 0.135) | 0.20 / 0.93 | 1.00 | 0.08–0.09 / 0.66–0.68 |
| `unit_trace: none` | 0.160 / 0.219 / 0.190 | 0.99 / 0.97 | 1.00 | 0.05–0.08 / 0.55–0.62 |

- K=8 насыщен: все шесть прогонов выигрывают у случайного игрока 100 % и играют вничью со скриптовым ботом (`share_vs_scripted` = 0.5 везде). По качеству режимы при K=8 не различаются.
- `ratio_mode` при K=128 (seed 0): `joint` лучше `per_unit` в среднем на 0.13 по `share_vs_scripted` (0.268 / 0.242 / 0.318 против 0.236 / 0.044 / 0.160); без сорвавшегося прогона `per_unit`/`geo_mean` (0.044) разница ≈ 0.09–0.10. Дефолт не меняется — открытый пункт для владельца (см. «Отклонения»).

**Медленный тест unit_harvest с новым дефолтом** (`unit_trace` = `joint`) перепроверен ревьюером на `f5e8a4b`: 1.000 побед, обучение 134 с.

**Критерий 5 (скорость) — выполнен.** `scripts/bench_throughput.py --workers 1 2 4 --duration 60 --warmup 15`, коммит `a4f1ad1`: 4848 / 9298 / 13702 шагов сред/с = 104 % / 98 % / 128 % от SP1 (4641 / 9444 / 10664; порог 90 %), рост 1 → 2 → 4 монотонный. Отклонение: load average перед замером 1.86 при требовании < 1 (это может только занизить числа).

**`global_state` (team_tag, после калибровки T8.3).** Payload чанка: 12 192 байт с `global_state` против 7 584 без (×1.61), `train_step` 14.1 против 10.9 мс (+29 %, critic encoder).

---

## Отклонения и замечания

Отклонения от спеки и плана, записанные в T8.3–T8.4 (без изменений):

- team_tag: критерий 3 (≥ 0.80 побед команды) проверяется как «лучший из двух независимых прогонов» (решение контролёра, см. «Решения»). Одиночный прогон выбранного конфига проходит в 7 случаях из 9 (0.690 и 0.760 не прошли). Порог не снижен.
  - Владельцу: подтвердить такую проверку или выбрать другую — ещё меньшую игру, иной критерий, исправление пассивного равновесия self-play в SP3.
  - В прогоне приёмки (шаг 3) team_tag прошёл с первого прогона: 0.920.
- Эксперимент по юнитам (T8.4): при K=128 `ratio_mode: joint` дал `share_vs_scripted` выше `per_unit` в среднем на 0.13 (seed 0: 0.268 / 0.242 / 0.318 против 0.236 / 0.044 / 0.160; без сорвавшегося прогона `per_unit`/`geo_mean` с 0.044 разница ≈ 0.09–0.10). Дефолт спеки (`ratio_mode: auto` = `per_unit` с `Units`) по плану не меняется.
  - Владельцу: оставить `per_unit` или проверить `joint` на нескольких сидах и реальной игре (один сид, разброс между прогонами до 0.09).
- Изменённые бюджеты и размеры сред (только через решения, пороги не менялись):
  - unit_harvest: 400 000 → 260 000 шагов;
  - team_tag: поле 7 → 6, view_radius 2 → 3, LR 1e-3 → 2e-3, энтропия 0.01 → 0.003, 500 000 → 400 000 шагов;
  - эксперимент по юнитам: один прогон за раз, 340 000 шагов, дополнительный seed 1.
  - Подробности — в разделе «Решения T8.3–T8.4».
- Скорость: перед замером T8.4 load average был 1.86, а не < 1 (это может только занизить числа).

Задачи контролёра вне плана (после T8.4):

- **FIX-1** (`a4f1ad1`, отчёт `.superpowers/sdd/2026-10-08-sp2-game-model/task-FIX-1-report.md`) — нестабильный `tests/integration/test_sp2_game_runs.py::test_resume_continues_versions_and_checks_the_role_signature`.
  - Причина: первый прогон перебирал бюджет в 1500 шагов (счётчик шагов сбрасывается раз в ~0.5 с, плюс задержка остановки), иногда до ≥ 3000. Продолжение с фиксированным бюджетом 3000 тогда сразу останавливалось без шага обучения.
  - Исправлен только тест: бюджет продолжения = реальные `env_steps` первого прогона + 2000. Добавлена проверка, что счётчик шагов продолжается.
  - Ошибки в продукте нет. Перебор бюджета записан как открытый пункт SP5.
- **FIX-2** (`b97dfec`, отчёт `task-FIX-2-report.md`) — нестабильный `tests/integration/test_sp2_lifecycle.py::test_sigint_during_startup_exits_130_not_aborted` (exit −2).
  - Причина: CPython 3.12 помечает KeyboardInterrupt, вышедший из строкового `exec` (так `dataclasses` при `import torch` создаёт методы), как «необработанный». Под `python -m` процесс тогда завершается сигналом вместо нашего `sys.exit(130)`.
  - Исправлены ещё два режима:
    - Ctrl-C, потерянный в weakref-callback importlib, игнорировался;
    - Ctrl-C во время старта потока-наблюдателя давал traceback и exit 1.
  - Теперь Ctrl-C при запуске всегда даёт 130. `eval`/`bc`/`validate` при потерянном Ctrl-C доделывают работу и выходят с 130 и `Interrupted`. README это описывает (`1311fbb`).

Отклонения T8.7 от брифа:

1. Вывод команд и временные файлы — в рабочем каталоге задачи (`/home/viv/.claude/jobs/0db917f9/tmp/t87/`), а не в `/tmp` (указание контролёра). Это касается и `run.dir` распределённого smoke-теста шага 8. Добавлены флаги `-p no:cacheprovider --color=no`.
2. Шаг 2: тесты пунктов T5.1/T3.3/T1.5 лежат в `tests/unit`, а не в `tests/contract`, поэтому они прогнаны отдельной командой (174 passed).
3. Шаг 4: `grep -n "Ruling: unit_trace"` пуст из-за обратной кавычки в тексте решения; само решение в отчёте есть.
4. Шаг 7: две правки диапазона времени quickstart (`07ae1a5`, `6c6f863`). Третий прогон шага 7 не делался, обоснование в §3.7.
5. Шаг 10: публичные имена `QueueReader`, `release_command_queues`, `queue_depths` вместо `_QueueReader` (R6, P17).
6. Шаг 13: коммит и push отчёта сделаны. Сообщение владельцу и шаг 14 (merge) не выполнялись по указанию контролёра: их делает контролёр после финального ревью ветки.

## Гигиена

- После шагов 1–11 и уборки `git status --porcelain` пуст.
- `git status --porcelain --ignored` без `.venv`/кэшей/`.superpowers/`: **clean**.
- `runs/`, `eval.json` и каталог распределённого smoke удалены.
- `ps -eo pid,args | grep -E "colosseum|multiprocessing"`: **no leftover processes**.
- В `/tmp` остались файлы `sp2-slow-seed*.txt`, `sp2names`, `sp2r`. Их создали более ранние задачи (T8.3), а не эта приёмка; они не трогались.

---

## Решения контроллера во время SP2 (все записи «Ruling:» из журнала, по порядку)

Каждая запись: что решено — почему — чем грозит, если решение неверно. Сначала решения этапа плана (PR-1..PR-5) и предполётной проверки (P1–P22). Затем все строки «Ruling» журнала контролёра SP2 (`.superpowers/sdd/2026-10-08-sp2-game-model/progress.md`) по порядку, дословно, на английском. Если строка не называет свою задачу, в квадратных скобках добавлен префикс задачи. В конце — решения T8.3–T8.4 из черновика.

### Решения этапа плана (PR-1..PR-5, `00-overview.md` и журнал)

- PR-1 — K == 1: все режимы схлопываются в `joint` (поправка R9); явный `unit_trace: none` соблюдается — почему: точное выполнение «при K = 1 все режимы дают один результат» (спека, блок 6) и поведение SP1 для игр без юнитов — цена ошибки: агент `Units(1, …)` с `per_unit` учится с совместной функцией потерь (одна запись INFO в логе).
- PR-2 — ядро команды всегда занимает одно место, правилу `teammates` следуют только остальные места команды (прочтение части C спеки, блок 7, п. 4) — почему: владелец всегда собирает данные в своей команде, это нужно ротации владельцев и критерию `coop_buttons` — цена ошибки: при `mixed` место ядра никогда не отдаётся другому агенту.
- PR-3 — в распределённом режиме вариант партии фиксируется на всё время жизни среды воркера — почему: до SP5 в распределённом режиме нет координатора — цена ошибки: распределённые прогоны игр с несколькими вариантами видят смесь вариантов, выбранную при старте воркеров.
- PR-4 — `put_row` оставлен рядом с `tree_assign` (поправка R4) — почему: обе функции уже выполнены и проверены, объединять их не стоит переделки — цена ошибки: небольшой дублирующий хелпер.
- PR-5 (P21) — файлы `models.py` примеров — самостоятельные шаблоны, повтор маленьких классов Policy/Value между примерами допустим — почему: пользователь копирует одну папку примера — цена ошибки: немного дублирования в `examples/`.

### Решения предполётной проверки (P1–P22, `constraints.md`)

- P1 (T4.2) — R9 реализуется точно; тест эквивалентности при одном решающем перебирает только `unit_trace` ∈ {`joint`, `geo_mean`} — почему: явный `none` соблюдается и вне политики даёт другой результат — цена ошибки (оценка контроллера): тест не ловит расхождение `none` при K = 1, которое и так задумано.
- P2 (T4.2) — R10 и R12 не вписаны в текст задачи, реализуются по `constraints.md` — почему: поправки обязательны и сильнее текста частей — цена ошибки (оценка контроллера): нет; без этого пропали бы диагностики (`log_rho_joint_abs_p95`, тест нулевого лага) и GPU-тесты-преемники.
- P3 (T3.3, T3.4, T4.2, T4.3) — добавить тесты R14, R15, R9/R10/R12, R16; ожидаемые числа «N passed» меняются — почему: поправки добавляют покрытие — цена ошибки (оценка контроллера): нет (критерий — зелёный полный набор).
- P4 (T5.3) — R13: ячейки по ролям получают `length_mean` и W/D/L по типу соперника (`latest`/`past`/`arena`) с проверками — почему: поправка R13 — цена ошибки (оценка контроллера): немного больше полей в записях `episodes`.
- P5 (T5.4) — тест R14 через monkeypatch `rollout_worker_process` (`max_idle_steps=123` доходит до воркера); R8: docstring `__main__` сохраняет строку про Docker/K8s — почему: проверка доставки настройки без запуска процессов; оверлей не должен терять строку — цена ошибки (оценка контроллера): нет.
- P6 (T6.2) — удалить пустую строку теста (03:5209) и всегда истинный assert (03:5217) — почему: тест не должен ничего не проверять — цена ошибки (оценка контроллера): нет.
- P7 (T6.3) — GPU-тест R12 «BC на деревьях на CUDA» (`trainer_for` получает `device`); запрещённое маской действие эксперта — `DataError`, так что `colosseum bc` печатает одну строку `Config error:` — почему: R12 и правило «ошибки данных — одна строка» SP1 — цена ошибки (оценка контроллера): плохие данные BC печатали бы traceback.
- P8 (T6.4) — скрипт копирования удаляет мёртвый `_CHUNK_STEP_ARRAYS` SP1 из модуля сериализации sp2 — почему: мёртвый код плоского чанка — цена ошибки (оценка контроллера): нет.
- P9 (T7.1) — перенести тест потока seed лёрнера SP1 без изменений; применимые тесты `test_performance.py` SP1 (флаг torch.compile, CPU pin_memory) — в `tests/unit/test_sp2_performance.py`, иначе пропуск записывается в ворота покрытия T7.3 — почему: без молчаливой потери покрытия SP1 — цена ошибки (оценка контроллера): незамеченная потеря теста SP1.
- P10 (T7.2) — `past_games["agent_0"] >= 0` заменить явной проверкой ключа и значения; `TicTacToePolicy` требует `action_spec` (без молчаливого запасного `CategoricalDist`) — почему: всегда истинный assert и молчаливый fallback запрещены — цена ошибки (оценка контроллера): нет.
- P11 (T7.3) — в воротах покрытия колонка «GPU-преемник» (R12); строки `test_ttt_slow.py`+`ttt_eval.py` и `test_examples.py` — «отложено до T8.3 / T8.5 (принятый пробел)» (R17); R18 (`"layout": "2p"` в бенчмарке) — почему: поправки R12, R17, R18 — цена ошибки (оценка контроллера): между T7.3 и T8.3/T8.5 медленный тест крестиков-ноликов и тест примеров отсутствовали.
- P12 (T8.1, T8.2) — R5 `context="demo, "`; R1 вложенная форма рейтингов; ожидание RED «ошибки сбора» неверно — smoke-тесты собираются и падают при запуске — почему: поправки R1/R5 и фактическое поведение pytest — цена ошибки (оценка контроллера): нет.
- P13 (T8.3) — R1; быстрые тесты T8.3 в `tests/learning/test_sp2_units_coop_learning.py` (не перезаписывать `test_sp2_fast_learning.py` T7.2); `baseline > 0` перед делением; расширять `tests/cli_runner.py`, а не дублировать его в `demo_learning` — почему: уникальные имена файлов тестов, без деления на 0, без дубликатов — цена ошибки (оценка контроллера): нет.
- P14 (T8.4) — R11 точно: без `test_unit_trace_default.py`, проверка `grep -n "geo_mean" tests/unit/test_appo_v2.py`, явный `git add`, `diagnostics()` бросает `KeyError` на отсутствующую метрику (без заполнения NaN); раздел отчёта — «Эксперимент по юнитам и скорость (T8.4)» — почему: один закрепляющий тест дефолта; без молчаливых NaN — цена ошибки (оценка контроллера): нет.
- P15 (T8.5) — grep запрещённого наследия `git grep -n <pattern> -- ':!review' ':!research' ':!docs/superpowers'`; число пропущенных тестов — шесть; перенос в unit-тест: отказ chase на многоэлементное действие, «validate каждого конфига примеров», проверка секции `run`, тест манифестов деплоя SP1; неизвестный `preset` space_miners — `ValueError` — почему: без потери тестов SP1 примеров — цена ошибки (оценка контроллера): нет. (Файл назван `tests/unit/test_example_configs.py`, см. решение T8.5 ниже.)
- P16 (T8.6) — grep `/eval.json` («.gitignore ok»), ссылка на «Step 5.1»; строки README по R1; таблица `docs/GPU_CHECKS.md` перестраивается из `pytest -m gpu --collect-only -q` (R12) — почему: документация описывает реальное состояние — цена ошибки (оценка контроллера): расхождение списка GPU-тестов с документом.
- P17 (T8.7) — grep публичных `QueueReader`/`release_command_queues`/`queue_depths`; PR-1..PR-4 и все решения журнала — в отчёт; ссылка на тест авторазрешения в `test_appo_v2.py` (R11); имена разделов как в P14 — почему: отчёт должен ссылаться на реальные имена — цена ошибки (оценка контроллера): нет. Выполнено в этом отчёте (§3.4, §3.10, этот раздел).
- P18 (T5.3, T5.4) — `FakeAlgorithm` один раз в `tests/game_helpers.py` (без дословной копии) — почему: общий набор тестовых средств — цена ошибки (оценка контроллера): нет.
- P19 (T3.4) — `RolloutLoop` бросает `ValueError`, если `len(lineups) != num_envs` (без молчаливого отбрасывания) — почему: никаких молчаливых fallback — цена ошибки (оценка контроллера): нет.
- P20 (T5.2) — `RatingBook.update` бросает `ValueError` с именем неизвестного варианта — почему: никаких молчаливых fallback — цена ошибки (оценка контроллера): нет.
- P21 (T8.1, T8.2, T8.5) — см. PR-5.
- P22 (часть 01, заметка 10; часть 04, заметки 7 и 13) — заменены поправками R3, R1, R11 — почему: поправки обязательны — цена ошибки (оценка контроллера): нет.

### Записи журнала контролёра (дословно)

1. [T0.1, предполётная проверка] Ruling: pre-flight scan of intra-part consistency runs in parallel with T0.1 — T0.1 touches only old code (SP1 residuals) and was replayed by Part A's author; cross-part conflicts were already resolved by amendments R1–R18 — cost if wrong: a scan finding forces a T0.1 rework.
2. [T1.x] Ruling: start T1.x while the pre-flight scan is still running — Part A was replayed task by task by its author; scan findings will be applied to the tasks they name before those tasks run — cost if wrong: a scan finding against an already-executed Part A task needs a follow-up fix commit.
3. [предполётная проверка] Ruling: P1–P20 — apply the scan's proposed fixes in the named tasks (R-amendments not yet written into part text, test/file-name clashes, no-op asserts, silent fallbacks → errors, missing SP1 test successors) — spec is binding; each fix is local — cost if wrong: small rework in the named task.
4. [предполётная проверка] Ruling: PR-5 (P21) — example models.py files are standalone templates; small repeated Policy/Value classes across examples are accepted — users copy one example dir — cost if wrong: minor duplication in examples/.
5. [предполётная проверка] Ruling: P3/P1 — "N passed" counts in briefs are approximate when amendments add tests — counts were computed before the amendments — cost if wrong: none (full-suite green is the gate).
6. Ruling: T1.2 Units.__eq__/__hash__ include component order (deviation from brief) — gymnasium Dict eq ignores key order but mask layout depends on it; matches ActionSpec.signature — cost if wrong: two spaces differing only in Dict order compare unequal.
7. Ruling: T1.5 reviewer Minor 1 (final step skips empty-seat/global_state checks) treated as a spec gap — spec block 1 says data for an EMPTY seat is an EnvContractError without exception — fix round 1 with Minor 2 (non-int seat keys / action_masks=None -> EnvContractError, global constraint) — cost if wrong: one extra fix round
8. Ruling: T1.6 Important 1 is plan-mandated code (probe close() masks EnvContractError) — fix it: the brief's own interface line requires EnvContractError for a non-MultiAgentEnv env_fn — cost if wrong: none
9. Ruling: T1.7 plan-mandated duplicate unknown-agent check in agent_roles/get_agent_config — fix with a shared private helper (no behavior change) — cost if wrong: none
10. Ruling: T2.1 .gitignore `dist/` -> `/dist/` (outside the brief) accepted — the unanchored rule hid the new networks/dist package (now and after T7.3) — cost if wrong: none found (no other dist dirs).
11. Ruling: T2.1 Minor 1 (param/shape errors in make_distribution don't name the action group, brief promised it) scheduled into T2.2 (which edits make_distribution): wrap each group's construction and re-raise ValueError naming the group; explicit log_std shape check — cost if wrong: small extra scope in T2.2.
12. Ruling: T3.2 plan-mandated gap (write_pad does not enforce 'PAD only after an episode end') — fix; also fold cheap guards (B-BOOT only with 1 free slot, park checks agent+spec owner, write_act at cursor 0 requires begin) — they turn silent worker/learner divergences in T3.4 into errors — cost if wrong: a legit write pattern rejected (none known under spec block 4 rules)
13. Ruling: T3.3 deviations accepted — (1) seat returns/eliminated_step/team result/episode length from EpisodeTracker (single source of truth, same values); (2) EpisodeEnd.final_obs holds live seats only (matches R15/spec block 1); (4) checked model lookup — cost if wrong: none found. Deviation (3) ActRecord.obs as a batch view rejected (aliasing; fix round 1).
14. Ruling: T3.4 Minors 1–2 fixed now (not deferred): MatchRunner rejects collect=True on a checkpoint network (spec block 7: checkpoint seats never collect; wrong policy_version otherwise) and RolloutLoop closes its vec env when the constructor fails — cheap, prevents silent invariant break / leaked subprocesses — cost if wrong: one extra small round
15. Ruling: T4.1 Minor 1 (slot-structure precondition unchecked on the learner path) scheduled into T4.4: collect_batch validates each decoded chunk's slot structure (ACT never last; non-terminal ACT followed by ACT or BOOT; PAD only after a slot ending an episode; non-reset BOOT only last) with a cheap numpy check -> clear error naming the agent — protects against a broken producer/distributed peer silently corrupting targets — cost if wrong: small CPU cost per chunk.
16. Ruling: T4.2 rho_clip_frac/c_clip_frac count ~1e-7 float-noise ratios as clipped at bar 1.0 (≈5% at zero lag) — keep SP1 definition (parity; zero-lag test uses bars 1.5 like SP1) — cost if wrong: slightly inflated clip diagnostics in the T8.4 experiment; revisit there if misleading
17. Ruling: T4.2 Minor 1 (per_unit policy loss / mean_valid entropy masking not pinned with random unit masks) scheduled into T4.3 as an extra test — the per-unit loss is the core of criterion 4 — cost if wrong: a little extra scope in T4.3.
18. Ruling: T4.3 test-only Minors 1–4 done now (cheap; per-unit clip is the core of criterion 4): value_loss via compute_loss in the bootstrap test, an off-policy per-decider ratio/clip pin, an asserted elimination precondition, policy-loss discrimination vs mean-over-K and pooled mean — cost if wrong: one extra small round
19. Ruling: T4.4 Minors 1–2 fixed now: validate flag consistency (terminal ACT ⇒ reset_after; open ACT ⇒ not reset_after; PAD ⇒ reset_after; non-ACT ⇒ not terminal) and build the slot-letters string only on failure (hot path; speed criterion 5) — cost if wrong: one extra small round
20. Ruling: T5.1 under teammates: mixed, an opposing team's non-core seats may draw the owner's latest ("latest другого обучаемого агента" is relative to that team's core) — spec-literal; mixed teams are about robustness to arbitrary partners; same-entity pairs are skipped in ratings — cost if wrong: with 2 agents ~half of mixed arena matches contain the owner on both sides (less pure arena signal).
21. Ruling: T5.2 plan-mandated silent fallback (win_rate/elo return priors for an unknown layout) — fix: raise ValueError like update (P20, no silent fallbacks); fold cheap test pins (decisive latest-first past case; worked 2v1v1 / duplicated-agent ELO values; mixed latest+ckpt team) — cost if wrong: none
22. Ruling: T6.2 validate_config reports an unsupported role space as ConfigError (env_spec keeps EnvContractError for env authors' runtime path) — Part A contract note 2 asks validate_config to convert TypeError/ValueError from spec construction to ConfigError; the CLI prints one line either way — cost if wrong: error class differs between validate and train for the same mistake
23. Ruling: T6.1 fix all three Importants — (1) signature check must cover every listed role (real bug); (2) rank solo mode per-seat report (plan-mandated gap vs spec block 8 'SP1 solo mode: per-seat return and outcome'); (3) keep SP1's checkpoint-architecture validation by reusing core.validation._check_model once per distinct (networks, roles), wrapped as ConfigError naming the path (plan-mandated drop of an SP1 guarantee) — cost if wrong: one fix round
24. Ruling: T6.4 plan-mandated grpc-less failures (fixture imports grpc; gRPC imports before scope check) — fix: grpc is an optional extra, the CI suite must skip cleanly — cost if wrong: none
25. Ruling: T7.2 integration test overrides (total_timesteps 6000, checkpoint.interval 5, match_refresh 0.25 s) to make past_games > 0 robust — assertions only strengthened; remaining timing dependence weakest on very fast machines — cost if wrong: rare flake on a much faster CI box.
26. Ruling: T8.2 minors fixed now in one small round (incl. T8.1 minor 1): constructor checks for all six demo games (predator_prey reset could hang; team_tag view_radius=0 crash; coin_grid/tron/unit_harvest degenerate kwargs), stronger cross-play assertion (coop_a+coop_b present), a fast team_tag train smoke (only fast path with uint8 global_state + critic encoder) — cost if wrong: one extra small round
27. Ruling: T8.5 SP1 test_examples port named tests/unit/test_example_configs.py (P15 named test_reference_examples.py, which collides with the task's contract test basename) — basenames must be unique — cost if wrong: none
28. Ruling: T8.5 minors fixed now in one small round: port SP1 test_attention_example_builds_a_stateful_model (closes R17 gap), space_miners tie ranks follow the framework convention (min rank 1/1), outcome-path tests (win bonus, tie-break, tie), Round 2/Final Round random-match coverage, push-mask docstring correction — cost if wrong: one extra small round
29. [T8.5] Ruling: tron keeps averaged ranks for simultaneous crashes (3.5/3.5) while space_miners uses min-rank ties — both are allowed by the contract (fractional ranks; SP1 ruling T5.2), pairwise results are identical (equal ranks = draw); averaged ranks express a shared placement in elimination games — cost if wrong: mean-rank numbers differ slightly between conventions in eval reports.
30. Ruling (T8.3, implementer, measured): unit_harvest example budget 400k → 260k env steps (400k trained 262 s > 3-min target; 260k wins 1.000 on seeds 0–2) — cost if wrong: less headroom on slower machines.
31. Ruling (T8.3, implementer, measured): team_tag budget 500k → 400k, lr 1e-3 → 2e-3, entropy 0.01 → 0.003 (old settings at 400k: 0.70/0.56/0.59; new: 0.915/0.840/0.850 vs 0.80 threshold) — thresholds unchanged — cost if wrong: worst seed margin 0.04 may flake on a slower machine; fallback 500k (~217 s).
32. Ruling: T8.3 Important (team_tag acceptance rests on 3 non-reproducible async samples, worst 0.84 vs 0.80, after a passing 500k config was cut for time) — fix round: measure the current config on seeds 3–5 AND the original 500k/lr1e-3/ent0.01 config on seeds 0–2 (fresh runs), pick the config with the better worst-case margin (time ≤ ~4 min accepted as 'примерно ≤3 мин'), update the ruling line with all numbers — robustness of an acceptance test outweighs ~40 s — cost if wrong: ~40 s longer slow suite
33. [T8.3, fix round 2] Ruling: team_tag slow test keeps the ≥0.80 WIN-rate threshold but allows ONE independent retraining when the first run misses (best of two runs, both runs printed and recorded) — learning outcome is run-to-run stochastic (async training; ~1/7 passive-draw runs), a single retry gives ~98% pass rate without changing the criterion; open item for SP3 (PFSP over snapshots / anti-passivity) — cost if wrong: a real regression that halves pass probability could hide behind the retry; +~3 min when the retry triggers.
34. [FIX-1] Ruling: fix the flaky resume integration test as a small dedicated task FIX-1 before T8.4 (criterion 1 requires a reliably green fast suite) — cost if wrong: none.
35. [FIX-1] Ruling: local budget overshoot of ~1 s worth of steps is harmless for real budgets (LR progress clamped; SP1 budget rule is about counting) — record for CLAUDE.md parked items (SP5 "one budget semantics"): a resume needs a budget above the checkpoint's env_steps to train; optional warning when a resume starts at/past the budget — cost if wrong: tiny-budget runs exceed budget by up to 100%+.
36. Ruling (T8.4, coded pre-registered rule): unit_trace auto with Units = joint (K=128 per_unit score share vs scripted bot, 2-seed mean: joint 0.206, geo_mean 0.093, none 0.190; K=8 saturated at 0.5) — cost if wrong: run-to-run noise (0.044 vs 0.135 same config) exceeds the rule's 0.05 margin; joint vs none undecided by data; geo_mean consistently worst.
37. Ruling: T8.6 dropped-reward Important fixed in the docs only (describe the real behaviour: dropped silently at default log level, counted in the worker's dropped_reward_episodes stat); surfacing the counter (warn-once or a metric) parked as a final-review candidate — docs must describe the real state, a code change is outside a docs task — cost if wrong: users keep losing rewards of never-acting seats without a visible signal until the final wave/SP3.

### Решения (rulings) T8.3–T8.4 (из черновика)

Пороги критерия 3 не менялись. Калибровка шла по T8.3 Step 9 и правилу контролёра (fix round 1) и затронула только конфиги примеров. Худший случай считается по всем прогонам: seed не воспроизводит асинхронный прогон.

- Ruling: unit_harvest `training.total_timesteps` 400 000 → 260 000.
  - Почему: потолок 170 с обучения (floor(170 × 1545 / 10 000) × 10 000). На 400k было 1.000 за 262 с, на 260k 1.000 / 1.000 / 1.000 на seeds 0–2 за 164 / 169 / 143 с.
  - Цена ошибки: пример обучается меньше, чем мог бы. Запас над порогом большой (1.000 против 0.80), поэтому разброс между прогонами здесь не опасен.
- Ruling: team_tag — `env.kwargs` size 7 → 6 и view_radius 2 → 3, `learning_rate` 1e-3 → 2e-3, `entropy_coeff` 0.01 → 0.003, `total_timesteps` 500 000 → 400 000. Обучение 175–208 с.
  - Почему (все прогоны; ✗ — ниже 0.80):
    - исходный конфиг (7/2, 500k, 1e-3, 0.01): 0.960, затем 0.440 ✗ / 0.825 / 0.655 ✗ (seeds 0, 0, 1, 2), 192–217 с;
    - 400k, 1e-3, 0.01: 0.700 ✗ / 0.555 ✗ / 0.585 ✗;
    - 400k, LR 2e-3: 0.820 / 0.850 / 0.565 ✗;
    - 400k, LR 2e-3, энтропия 0.003: 0.915 / 0.840 / 0.850, затем 0.370 ✗ / 0.760 ✗ / 0.850 (seeds 3–5);
    - + view_radius 3: 0.765 ✗ / 0.925 / 0.935 / 0.920 / 0.665 ✗;
    - + size 6 (выбран): 0.690 ✗ / 0.935 / 0.890 / 0.910 / 0.835 / 0.950, повтор seed 0 — 0.935, проверка теста с повтором — 0.760 ✗ и 0.945;
    - то же на 500k: 0.945 / 0.790 ✗ / 0.840 / 0.815 / 0.930 / 0.590 ✗, 212–280 с (бюджет не помогает).
  - Как проигрывают: только ничьими. В проваленном прогоне 131 победа, 69 ничьих, 0 поражений: часть прогонов self-play сходится к осторожной политике и доигрывает до лимита в 30 шагов.
  - Опорные значения: скриптовый chase против случайной команды выигрывает 0.906 на 7/2 и 0.926 на 6/3; случайная команда против случайной — 0.276 / 0.310.
  - Цена ошибки: изменённый пример team_tag легче, чем игра по умолчанию. Главное: порог выполняется не в каждом прогоне. Падают 2 прогона из 9 (худший 0.690), поэтому одиночный прогон медленного теста может упасть без изменений в коде (смягчено повтором, см. следующее решение). Риск — разброс между прогонами, а не скорость машины: бюджет считается в шагах сред.
- Ruling (контролёр, fix round 2): медленный тест team_tag засчитывает **лучший из двух независимых прогонов**, порог прежний — ≥ 0.80 побед.
  - Как работает: если первый прогон (`training.seed = LEARNING_SEED`) ниже 0.80, тест один раз обучает заново — новый каталог прогона, `training.seed = LEARNING_SEED + 1000` — и проходит, если второй прогон достигает 0.80. Числа обоих прогонов печатаются в выводе pytest. Повтор есть только у team_tag, у остальных тестов его нет.
  - Почему: примерно 1–2 прогона из 7–9 сходятся к пассивной политике на ничьи (0 поражений). Асинхронное обучение не воспроизводится по seed. Один повтор даёт ≈ 1 − (1/7)² ≈ 98% при оценке 6/7 и ≈ 1 − (2/9)² ≈ 95% по всем 9 прогонам.
  - Цена ошибки: за повтором может спрятаться настоящая регрессия, которая вдвое снижает вероятность прохождения одного прогона (с ~0.8 до ~0.4 всё ещё даёт ~64% прохождений теста).
  - Открытый пункт для SP3: PFSP по снимкам и/или меры против пассивности в self-play (разнообразие соперников, штраф за ничью), чтобы снова требовать прохождения с одного прогона.

- Ruling: `unit_trace: auto` с `Units` = **`joint`** (было `geo_mean`, дефолт спеки до эксперимента). Теперь `auto` = `joint` для любых действий.
  - Почему: правило T8.4 (`recommend()`: другой trace побеждает, если при K=128 `share_vs_scripted` с `per_unit` выше `geo_mean` на ≥ 0.05, а при K=8 ниже не более чем на 0.05) выбирает `joint` и по seed 0 (`joint` 0.236, `geo_mean` 0.044, `none` 0.160), и по среднему двух сидов (0.206 / 0.093 / 0.190). При K=8 везде 0.500 (насыщение). `geo_mean` — последний на обоих сидах. `joint` — запасной вариант спеки (риск «Дефолт `unit_trace`»).
  - Цена ошибки: `none` может быть не хуже или немного лучше. По одному seed 1 правило выбрало бы `none` (0.219 против 0.176), и против случайного игрока `none` стабильнее (0.99 / 0.97 против 0.79 / 0.79). Разница `joint`–`none` меньше разброса между прогонами (одна и та же конфигурация давала 0.044 и 0.135). Исправление — одна строка в `resolve_modes` (`src/colosseum/algorithms/appo.py`); проверка — `tests/unit/test_appo_v2.py::test_modes_resolve_auto_and_collapse_for_one_decider`.
- Ruling: эксперимент по юнитам шёл по одному прогону (`--parallel 1`), а не по два, как в плане (Step 3); бюджет — 340 000 шагов сред (пилот: 709 шагов сред/с при K=128 × 480 с).
  - Почему: один прогон загружает ≈ 7 из 8 ядер. Два одновременных прогона меняли бы policy lag, а от него зависят диагностики ρ. Серийно всё равно ≈ 75 минут.
  - Цена ошибки: нет (только время).
- Ruling: дополнительный seed 1 для трёх строк K=128 `per_unit` (файл `docs/benchmarks/units-experiment-seed1.json`). Правило плана требовало повтора только у границы ±0.02, но повтор одной и той же конфигурации разошёлся на 0.09 (больше порога 0.05). Строки K=8 не повторялись: там все режимы упираются в потолок, поэтому `recommendation` в этом файле — `null` с `recommendation_note` (правилу нужны доли при K=8). Там же, в `separate_runs`, записан отдельный прогон на 340k шагов той же конфигурации `per_unit`/`geo_mean` с seed 0 (0.135 / `win_vs_random` 0.99).
  - Цена ошибки: нет (больше данных; seed 0 и среднее двух сидов дают одно и то же решение).

---

## Открытые пункты (отложены, не блокируют слияние)

Совпадают с «Parked during SP2» в CLAUDE.md. Остальные пункты, отложенные в SP1 (SP4/SP5/SP6, Tier 2, мелкие), перечислены там же, в «Open items parked during SP1 and SP2», и не изменились.

1. **SP3:** медленный тест team_tag засчитывает лучший из двух прогонов. Примерно 2 из 9 одиночных прогонов сходятся к пассивной политике на ничьи. Нужен PFSP по снимкам или меры против пассивности, чтобы снова требовать прохождения с одного прогона.
2. **Владелец:** `ratio_mode` при K=128 на unit_harvest — `joint` обошёл `per_unit` на ≈ 0.09–0.13 по `share_vs_scripted` (seed 0, шумно). Дефолт спеки (`auto` = `per_unit` с `Units`) оставлен до решения владельца.
3. **Владелец:** `unit_trace: auto` с `Units` = `joint` выбран на шумных данных. Разброс между прогонами больше порога правила 0.05, `joint` против `none` данными не решён, `geo_mean` стабильно хуже.
4. **SP5:** перебор бюджета и resume.
   - Локальный прогон перебирает `total_timesteps` примерно на секунду шагов сред.
   - Продолжению нужен бюджет выше `env_steps` чекпоинта, иначе оно не обучается. Возможно предупреждение, когда resume стартует на бюджете или за ним.
   - Решать вместе с «one budget semantics» SP5.
5. **SP5:** с заданным `training.seed` распределённые воркеры на разных машинах получают одинаковые seed сред (решение PR-3).
6. **Мелкое:** `colosseum validate` печатает контекст своих проверок `space.contains` в порядке «layout, episode step, seat» вместо общего «seat, episode step, layout».

Вопросы владельцу (их задаёт контролёр):
- Принять проверку team_tag «лучший из двух независимых прогонов» (порог 0.80 прежний) или выбрать другую?
- `ratio_mode` при K=128: оставить `per_unit` (спека) или проверить `joint` на нескольких сидах и реальной игре?
- `unit_trace` = `joint`: принять выбор, сделанный на шумных данных (`none` не хуже в пределах разброса)?
- Прогнать 18 GPU-тестов на машине с CUDA (`docs/GPU_CHECKS.md`).

Кандидаты финального ревью ветки (журнал контролёра отложил их на финальную волну исправлений; в CLAUDE.md их нет, потому что они ещё не перенесены за SP2):
- счётчик `dropped_reward_episodes` виден только в последней строке лога воркера. Нужен сигнал при потере награды: предупреждение один раз или метрика (решение T8.6);
- FIX-2:
  - `serve-weight-store` выходит с 0 на Ctrl-C, но с 130 после Ctrl-C, потерянного при запуске;
  - autouse-фикстура в `test_process_lifecycle.py` для `take_lost_interrupt()`;
  - два новых интеграционных теста заменяют `PYTHONPATH`.

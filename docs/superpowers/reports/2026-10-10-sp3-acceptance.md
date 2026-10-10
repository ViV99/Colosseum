# SP3: приёмка по спеке §3 (T6.7)

- Ветка `sp3-league`. Начало приёмки: HEAD `6fa1406` (код финальный: задачи T0.1–T6.6, финальное ревью ветки и его единственная волна исправлений уже выполнены). Правки документации во время приёмки: `f2f54ec` (`docs:`). Отчёт — следующий коммит после `f2f54ec`.
- Дата: 2026-10-10.
- Машина: WSL2, 8 ядер (`nproc` = 8), 11 ГБ ОЗУ, без GPU. Python 3.12.3, torch 2.14.1+cpu, `.venv`.
- Записи финального ревью и волны исправлений: [`2026-10-10-sp3-review/`](2026-10-10-sp3-review/README.md).
- Порядок изменён решением контроллера (журнал, строка 181): финальное ревью ветки и волна исправлений прошли **до** приёмки, поэтому все замеры приёмки (быстрый набор, slow-набор, 6 сидов team_tag, скорость) выполнены один раз на финальном коде.
- Артефакты прогонов лежат в git-ignored рабочем каталоге SDD (`.superpowers/sdd/2026-10-10-sp3-league/t67-runs/`), не в `/tmp`. В репозиторий попали `docs/benchmarks/team-tag-acceptance.jsonl` и перезамер скорости `docs/benchmarks/sp3-throughput/`.
- Шаги 14–15 (статус «принято» и merge) не выполнялись: ждут явного одобрения владельца.

## Итог

**ВСЕ КРИТЕРИИ ВЫПОЛНЕНЫ (ALL PASS).** Одна правка документации во время приёмки (`f2f54ec`: три мелочи повторного ревью волны, время BC в README, перезамер скорости). Вопросы владельцу — в конце: принять SP3 и слить ветку; подтвердить два решения волны исправлений, уточняющие спеку; GPU-проверки (22 теста) ещё ждут машины с CUDA.

| § | Критерий | Результат |
|---|---|---|
| 3.1 | Тесты | PASS: 1689 passed, 32 deselected, 398.95 с, exit 0, предупреждений 0; ruff чистый; дерево чистое; ≤ 2 воркеров во всех тестах; счёт 1689 / 10 / 22 совпадает с README и CLAUDE.md |
| 3.2 | Контрактные тесты | PASS: 40 файлов `test_sp3_*` + `tests/contract` — 614 passed, 4 skipped (CUDA), 54.9 с; у каждого пункта критерия 2 есть зелёный тест (таблица в §3.2) |
| 3.3 | Конвейер | PASS: медленный тест (один конфиг, BC с `--seed`): против `random` win 1.000 (порог 0.95), против `greedy` score 0.492 (порог 0.40), против `bc_net` score 0.483 (порог 0.45); смоук на tic-tac-toe 6.1 с; конвейер против обучения с нуля (T6.4): 0.483 против 0.492 — выигрыша нет, оба на потолке игры |
| 3.4 | team_tag с одного прогона | PASS: медленный тест одним прогоном — win 0.940; 6 новых сидов (10–15) — 0.925 / 0.905 / 0.960 / 0.910 / 0.970 / 0.980, минимум 0.905 ≥ 0.80, среднее 0.942; доля якоря 0.2 (ruling T6.3), запасной вариант (штраф за ничью) не нужен |
| 3.5 | Скорость | PASS (перезамер после волны): ветка 4911 / 9244 / 14107 шагов сред/с при 1 / 2 / 4 воркерах = 106.8 / 99.0 / 98.2 % от `main` (4598 / 9339 / 14365), рост монотонный |
| 3.6 | Совместимость с SP2 | PASS: 11/11 копий конфигов SP2 проходят `validate` (с предупреждениями о старых ручках); 136 тестов совместимости зелёные; чекпоинт SP2 как frozen, `init`, resume, учитель — node ids в §3.6 |
| 3.7 | Дефолты APPO | PASS: 18 прогонов, правило применено: `ratio_mode: auto` = `per_unit` (+0.037 < 2·SE 0.092), `unit_trace: auto` = `joint` (−0.005); дефолт в коде закреплён тестом |
| 3.8 | Документация | PASS: конвейер README (один конфиг) и Quick start дословно в чистом shell — все команды exit 0, результаты совпадают с README; флаги README / LEAGUE_GUIDE = `--help`; LEAGUE_GUIDE содержит все рецепты блока 11; ENV_GUIDE §10 «`infos` для скриптовых ботов» с ценой `subprocess`; одна правка `docs:` (время BC) |
| 3.9 | Распределённый режим | PASS: 50 тестов distributed/grpc passed; localhost smoke: `workers exit=0`, финальный `ckpt_v78`, одно предупреждение «reduced to latest only»; team_tag (scripted `random`) и `unit_harvest_league` — `exit=1`, `Config error:` с SP5 и именами скриптовых агентов |
| 3.10 | Slow-набор | PASS: 10 passed за 910.0 с (15 мин 10 с): 7 демо-игр (team_tag одним прогоном), конвейер unit_harvest, 2 проверки torch.compile |

---

## §3.1 Тесты: полный быстрый набор, ноль предупреждений, запись только в `tmp_path`, ruff

Предусловие: `git status --porcelain` пуст (HEAD `6fa1406`).

Команда (вывод в рабочий каталог задачи вместо `/tmp`): `timeout 3600 .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw -p no:cacheprovider`.

- `exit=0`. Результат: **1689 passed, 32 deselected in 398.95s (0:06:38)**.
- `grep -E "warnings summary|[0-9]+ warnings?( |$)"` по выводу: ничего — **no warnings**.
- `git status --porcelain --ignored` без `.venv`/кэшей/`.superpowers/`: после прогона в дереве только мои правки документации, сделанные во время прогона (`CLAUDE.md`, `docs/ENV_GUIDE.md`, `docs/LEAGUE_GUIDE.md`, новая папка `docs/superpowers/reports/2026-10-10-sp3-review/`); тесты ничего не оставили — нет `runs/`, `checkpoints/`, `data/`, `bc.pt`, `*.jsonl`.
- `.venv/bin/ruff check .`: `All checks passed!`.
- Сбор по маркерам: `not gpu and not slow` — 1689/1721; `slow` — 10/1721; `gpu` — 22/1721 (1689 + 10 + 22 = 1721). README («full fast suite (CI), 1689 tests», «10 tests», «22 tests») и CLAUDE.md («**1689** fast tests, **10** slow, **22** GPU-only») совпадают.
- `grep -rnE "num_workers[\"']?[:=] ?[\"']?[3-9]|--workers [3-9]" tests/`: одно попадание, `tests/unit/test_sp2_config_overrides.py:168` (`"rollout.num_workers": 8`) — только разбор `--set` в конфиг, процессы не запускаются. Ни один тест не запускает больше 2 воркер-процессов.

**PASS.**

## §3.2 Контрактные тесты

Команда: `.venv/bin/python -m pytest $(ls tests/*/test_sp3_*.py) tests/contract -v -rA -m "not slow"` (40 файлов `test_sp3_*` в `tests/unit`, `tests/contract`, `tests/integration`, `tests/learning` плюс весь `tests/contract`) → **614 passed, 4 skipped, 1 deselected in 54.89s**, exit 0. 4 skipped — CUDA-тесты `tests/unit/test_sp3_gpu_warmstart.py` (нет GPU); 1 deselected — медленный тест конвейера. Все перечисленные ниже тесты — PASSED в этом прогоне.

| Пункт (спека, критерий 2 / раздел 6) | Тесты (node ids) |
|---|---|
| Скриптовый бот за местом получает `infos` своего места; `reset` с `rng`; нелегальное действие и исключение в боте — ошибки с контекстом (T1.3) | `tests/unit/test_sp3_match_runner_players.py::test_a_scripted_seat_gets_obs_mask_and_infos_and_is_reset_every_episode`, `::test_bot_rngs_follow_the_episode_seed_seat_and_agent`, `::test_a_bot_that_breaks_the_rules_is_a_player_error_with_context[IllegalBot-…]` / `[CrashingBot-…]` / `[WrongShapeBot-…]` (контекст `worker 0, env 0, seat 1, episode step 1, layout 2p: agent 'bot': …`), `::test_a_bot_whose_reset_raises_fails_at_the_episode_start`, `::test_a_bot_editing_obs_or_mask_in_place_changes_neither_the_gate_nor_the_record`; `tests/unit/test_sp3_players.py::test_bot_rng_depends_on_episode_seed_seat_and_agent`; `tests/unit/test_sp3_validate_players.py::test_a_broken_scripted_agent_fails_validate[…]` |
| Frozen другой архитектуры играет и никогда не собирает данные; `collect` только у latest обучаемых мест (T1.3, T1.4) | `tests/unit/test_sp3_match_runner_players.py::test_a_frozen_player_of_another_architecture_is_batched_under_fixed`, `::test_a_scripted_player_never_collects_even_under_latest`, `::test_only_latest_seats_may_collect_and_fixed_seats_need_their_player`, `::test_a_missing_snapshot_of_a_scripted_player_is_an_error_not_a_collecting_seat`; `tests/contract/test_sp3_fixed_players_wiring.py::test_the_rollout_loop_serves_fixed_players_that_never_collect`; `tests/integration/test_sp3_fixed_players_worker.py::test_a_spawned_worker_seats_scripted_and_frozen_players_that_never_collect`; `tests/unit/test_sp3_players.py::test_load_fixed_players_loads_bots_and_frozen_weights_of_their_own_architecture` |
| Воркер выгружает удалённые снимки (T2.2) | `tests/contract/test_sp3_snapshot_eviction.py::test_the_worker_unloads_an_evicted_snapshot_once_no_lineup_uses_it`, `::test_the_launcher_sends_evictions_to_every_worker_and_retries_a_full_queue`, `::test_drain_commands_merges_evictions`, `::test_the_coordinator_hands_out_each_eviction_once`, `::test_without_match_refresh_the_coordinator_keeps_no_evictions` |
| Фактические доли категорий сходятся к заданным, включая перераспределение пустых категорий, расписания и override на агента (T3.2) | `tests/unit/test_sp3_matchmaker.py::test_drawn_category_shares_match_the_configured_mix`, `::test_shares_of_empty_categories_are_spread_proportionally`, `::test_shares_and_anchor_weights_follow_the_env_step_schedules`, `::test_a_per_agent_override_changes_only_that_owners_mix`, `::test_a_per_agent_layout_override_restricts_only_that_owner`; `tests/unit/test_sp3_coordinator_league.py::test_share_schedules_follow_the_coordinators_env_steps` |
| В асимметричной игре соперником бывает снимок другого агента (T3.2) | `tests/unit/test_sp3_matchmaker.py::test_asymmetric_games_draw_the_other_agents_snapshots` |
| Составы своего класса матчмейкера проверяются (T3.2, T3.3) | `tests/unit/test_sp3_coordinator_league.py::test_lineups_of_a_custom_matchmaker_are_checked`, `::test_a_custom_matchmaker_class_builds_the_lineups_and_gets_every_result`, `::test_a_bad_matchmaker_class_is_a_config_error`, `::test_the_coordinator_and_validate_report_a_failing_matchmaker_init_alike`; `tests/unit/test_sp3_validate_mix.py::test_a_broken_custom_matchmaker_fails_validate`, `::test_a_valid_custom_matchmaker_passes` |
| Хранение снимков: `keep_last` / `keep_every` / финальный; `trainer_state.pt` только в окне `keep_last`; перенос пула при resume из папки запуска (T2.1) | `tests/unit/test_sp3_snapshot_retention.py::test_keep_last_keep_every_and_the_trainer_state_window`, `::test_a_final_snapshot_is_never_evicted_and_keep_every_0_is_fifo`, `::test_the_snapshot_just_saved_is_kept_even_when_it_is_older`, `::test_a_run_dir_resume_imports_the_pool_and_a_checkpoint_dir_resume_does_not`, `::test_import_snapshots_links_model_and_meta_reads_the_source_only_and_applies_retention`, `::test_import_snapshots_checks_the_role_signature`, `::test_the_coordinator_stores_snapshots_by_the_configured_rules` |
| `init` в режимах strict и partial (T4.2) | `tests/unit/test_sp3_init.py::test_strict_pt_loads_every_tensor`, `::test_strict_mismatch_is_a_config_error_with_a_hint`, `::test_partial_loads_the_matching_tensors_and_reports_the_rest`, `::test_partial_without_any_match_is_a_config_error`, `::test_the_learner_starts_from_the_init_weights_at_version_zero`, `::test_resume_takes_precedence_over_init` |
| При прогреве критика политика на воркерах побитово не меняется (T4.3; уточнено волной, решение C-I2) | `tests/unit/test_sp3_critic_warmup.py::test_warmup_trains_only_the_value_path_and_keeps_the_workers_policy_bit_identical` (нормализатор `global_state` критика обновляется, все остальные тензоры, включая нормализаторы наблюдений, побитово прежние), `::test_the_warmup_counter_is_trainer_state`, `::test_the_lr_schedule_runs_as_usual_during_the_warmup` |
| Действие учителя-скрипта лежит в чанке и даёт лосс (воспроизводится лёрнером) (T4.5) | `tests/contract/test_sp3_dagger_worker.py::test_act_slots_of_collecting_seats_carry_the_teachers_action`, `::test_the_learner_reproduces_the_worker_with_labels_in_the_chunks`, `::test_the_teacher_sees_the_seats_infos`, `::test_the_teacher_flag_stops_the_queries`; `tests/unit/test_sp3_dagger_loss.py::test_compute_labels_is_the_joint_nll_over_labeled_act_slots`, `::test_appo_dagger_loss_with_units_is_the_bc_nll_over_labeled_act_slots`, `::test_a_student_learns_the_teachers_action`; `tests/integration/test_sp3_dagger_run.py::test_a_scripted_teacher_labels_the_students_decisions` |
| Данные `record` проходят через `bc` (T5.2) | `tests/integration/test_sp3_bc_record.py::test_bc_trains_on_a_record_dir`, `::test_bc_takes_several_data_paths`, `::test_a_record_split_into_several_part_files_loads_whole`, `::test_bc_reads_only_the_agents_roles_of_a_record`, `::test_bc_seed_makes_the_weights_reproducible`; `tests/integration/test_sp3_record.py::test_units_actions_and_uint8_observations_load_into_the_bc_trainer`; `tests/integration/test_sp3_one_config.py::test_record_bc_validate_and_eval_run_from_one_config_before_bc_pt_exists` |
| Совместимость с SP2 (критерий 6; T0.1) | см. §3.6 |

**PASS.**

## §3.3 Конвейер (главный критерий) и §3.10 slow-набор

`nproc` = 8; `uptime` перед slow-набором: load average 0.97 / 2.06 / 1.70.

Быстрые тесты SP3 «учится» и смоук конвейера: `.venv/bin/python -m pytest tests/learning tests/integration -m "not slow" -k "sp3" -v --durations=0` → **29 passed, 109 deselected in 31.63s**. Самые долгие: `tests/integration/test_sp3_sp2_resume.py::test_training_resumes_from_the_sp2_run_dir` 6.17 с, **`tests/integration/test_sp3_pipeline_smoke.py::test_tic_tac_toe_pipeline_smoke` 6.13 с** (record → bc → train с `init`, прогревом критика и kickstart → eval, всё из одного конфига), `tests/integration/test_sp3_anchor_runs.py::test_a_scripted_anchor_plays_its_share_and_shows_in_metrics_and_ratings` 6.11 с, `tests/integration/test_sp3_dagger_run.py::test_a_scripted_teacher_labels_the_students_decisions` 5.98 с, `tests/learning/test_sp3_fast_learning.py::test_scripted_kickstart_teacher_moves_the_student_to_its_action` 3.97 с, `::test_random_bot_anchor_is_beaten_while_only_latest_seats_collect` 0.33 с. Все меньше 20 с.

Slow-набор: `.venv/bin/python -m pytest -m slow -v -s -p no:cacheprovider` → **10 passed, 1711 deselected in 910.01s (0:15:10)**, exit 0.

| Тест | Напечатано (приёмка) | T6.4 (предпроверка, тот же сид) | SP2 при приёмке | Порог |
|---|---|---|---|---|
| coin_grid | обучение 41 с, 8126 шагов/с; score 9.410, random 2.345, **4.01×** | 5.37× | 5.63× (T8.3 по сидам 4.50 / 4.97 / 6.38×) | ≥ 2× |
| tic_tac_toe | 62 с; **win 0.927** | 0.938 | 0.917 | ≥ 0.80 |
| unit_harvest | 142 с; **win 1.000** | 1.000 | 1.000 | ≥ 0.80 |
| team_tag (один прогон) | 170 с, 2431 шагов/с; **win 0.940**, ничьи 0.060, средняя длина 7.985 | 0.910 | 0.920 (лучший из двух) | ≥ 0.80 |
| tron | 131 с; **2p 0.965, первое место 4p 0.932** | 0.985 / 0.955 | 0.970 / 0.958 | см. тест |
| predator_prey | 90 с; **охотник 1.000, жертва 0.990** (случайные между собой 0.550) | 1.000 / 0.995 | 1.000 / 0.990 | ≥ 0.70 |
| coop_buttons | 79 с; **coop_a 7.390, coop_b 7.390** (порог 4.404, оракул 7.340); cross-play a+a 5.08, a+b 3.07, b+b 5.53 | 7.390 / 7.390 | 7.39 / 7.39 | ≥ 4.404 |
| **конвейер unit_harvest** | seed 0; record 17 / bc 9 / train 102 / eval 19 с; 30 000 решений; 1686 шагов/с; **против `random` W/D/L 60/0/0, win 1.000; против `greedy` 0/59/1, score 0.492; против `bc_net` 0/58/2, score 0.483** | 1.000 / 0.483 / 0.483 | — | 0.95 / 0.40 / 0.45 |
| `test_vtrace_slots_torch_compile`, `test_appo_torch_compile` | PASSED | PASSED | PASSED | — |

**Медленный тест конвейера с одного конфига и BC с сидом.** Волна исправлений (пункты 1 и 15) перевела тест на один конфиг (`bc_net: {kind: frozen, path: bc.pt}`, `init.from: bc_net`, `kickstart.teacher: greedy` до появления `bc.pt`) и на `colosseum bc --seed`. Пороги T6.4 калибровались на BC без сида. Результат выше всех трёх порогов: 1.000 ≥ 0.95; 0.492 ≥ 0.40; 0.483 ≥ 0.45 (запас 0.033; оценка вероятности срыва этой проверки — ruling T6.4, дополнение). Значения в пределах калибровки T6.4 (score против `greedy` 0.475–0.500, против `bc_net` 0.483–0.500).

**Проверка «значение заметно ниже, чем раньше»: coin_grid.** 4.01× заметно ниже предпроверки T6.4 (5.37×) и чуть ниже минимума SP2 по сидам (4.50×), но вдвое выше порога. Разбор:
- код пути обучения coin_grid с T6.4 (`9a25799`) не менялся по сути: `git diff 9a25799..HEAD` по `algorithms`, `networks`, `worker`, `learner` — только ветка прогрева критика в `APPO._update_normalizers` (без прогрева вызов тот же: `update_normalizers(obs, gs)`), `kickstart_label_frac`, удалённый псевдоним `ModelPool` и assert в `_teacher_label` (DAgger); `coin_grid.yaml` переписан в форму v3 ещё в T6.6 (смесь соперников для одной команды не используется);
- два повторных прогона того же теста на том же коде сразу после slow-набора: **5.313×** и **3.761×** (`t67-runs/coin_grid_rerun.txt`).

Вывод: разброс между прогонами одного кода (асинхронное обучение, лаг политики) — 3.8–5.4×; регрессии нет. Пункт записан в «Открытые пункты» (порог 2× держится с запасом, но при сравнении «до/после» одного прогона мало).

**Конвейер против обучения с нуля** (`docs/benchmarks.md`, раздел «Конвейер против обучения с нуля», T6.4; замер не повторялся):

| вариант | сид | score против бота | win против бота | ничьи с ботом | win против random | score против bc_net | record, с | bc, с | обучение, с |
|---|---|---|---|---|---|---|---|---|---|
| конвейер | 0 | 0.483 | 0.000 | 0.967 | 1.000 | 0.492 | 16.6 | 9.2 | 101.4 |
| с нуля | 0 | 0.475 | 0.000 | 0.950 | 1.000 | 0.442 | – | – | 91.1 |
| конвейер | 1 | 0.500 | 0.000 | 1.000 | 1.000 | 0.500 | 17.0 | 9.1 | 102.6 |
| с нуля | 1 | 0.500 | 0.000 | 1.000 | 1.000 | 0.500 | – | – | 91.9 |
| конвейер | 2 | 0.467 | 0.000 | 0.933 | 1.000 | 0.467 | 17.1 | 9.1 | 105.4 |
| с нуля | 2 | 0.500 | 0.000 | 1.000 | 1.000 | 0.500 | – | – | 93.1 |

Честный вывод (ruling T6.4): средний score против бота 0.483 у конвейера и 0.492 с нуля — разница в пределах шума. Обе политики упираются в потолок игры: почти одни ничьи с ботом, ни одной победы над ним, 100 % побед над `random`. Конвейер стоит ≈ 129 с против ≈ 92 с. На `unit_harvest` с этим бюджетом тёплый старт не окупается: медленный тест проверяет, что все шаги работают вместе, а не выигрыш в качестве. Корректность `init`, прогрева и kickstart держится на быстрых и контрактных тестах (T4.2–T4.5, смоук T6.2). Ценность тёплого старта надо показывать на игре, где обучение с нуля за бюджет не доходит до бота (кандидат для SP4/SP6).

**PASS** (критерии 3 и 10).

## §3.4 team_tag с одного прогона

- Медленный тест — один прогон (рецепт блока 9: якорь `RandomBot`, `opponents: {latest: 0.6, snapshots: 0.2, rivals: 0, anchors: 0.2}`): win **0.940**, ничьи 0.060, средняя длина 7.985 (§3.3).
- 6 новых сидов того же рецепта (доля 0.2 выбрана в T6.3 на сидах 0–5). Предусловия: slow-набор закончился; load average перед запуском 0.82. Команда: `.venv/bin/python scripts/team_tag_anchors.py --anchors 0.2 --seeds 10 11 12 13 14 15 --jsonl docs/benchmarks/team-tag-acceptance.jsonl --run-dir <t67-runs>/team-tag-runs` (без `--draw-reward`: T6.3 не пошла по запасному пути); 17 мин 30 с, exit 0.

| сид | win | draw | loss | средняя длина | обучение, с | шагов сред/с |
|---|---|---|---|---|---|---|
| 10 | 0.925 | 0.075 | 0.000 | 8.99 | 170.5 | 2409 |
| 11 | **0.905** | 0.095 | 0.000 | 9.28 | 174.3 | 2358 |
| 12 | 0.960 | 0.040 | 0.000 | 7.44 | 175.5 | 2345 |
| 13 | 0.910 | 0.090 | 0.000 | 10.48 | 172.6 | 2386 |
| 14 | 0.970 | 0.030 | 0.000 | 6.54 | 180.9 | 2268 |
| 15 | 0.980 | 0.020 | 0.000 | 7.42 | 162.8 | 2539 |

Вывод скрипта: `anchors 0.2: min 0.905, mean 0.942, n 6`. Каждый сид ≥ 0.80. Пассивной политики нет: ничьих 2–9.5 %, партии 6.5–10.5 шага (у пассивных провалов брейнсторма — 30–35 % ничьих и партии до лимита в 30 шагов). На сидах 0–5 у той же доли (T6.3) было min 0.945 / mean 0.965; новые сиды чуть ниже, с запасом над порогом.

Оговорка (рекомендация ревью B): оценка идёт против той же случайной команды, против которой агент тренируется как против якоря (доля розыгрыша 0.2, фактическая доля партий ≈ 0.36). Результат показывает, что пассивный режим ушёл; обобщение за пределы якоря он не показывает.

**PASS.**

## §3.5 Скорость

- `git log --oneline -1 -- docs/benchmarks/sp3-throughput` → `98bd048` (замер T6.6 на коде `aea876e`).
- `git diff --stat aea876e..HEAD` по `worker`, `learner`, `algorithms`, `envs`, `networks`, `core`, `coordinator`, `league`: 12 файлов (волна исправлений: `appo.py`, `checkpoint_manager.py`, `coordinator.py`, `config.py`, `registry.py`, `validation.py`, `league/{base,mixture,pfsp}.py`, `networks/model.py`, `match_runner.py`, `rollout_loop.py`). По брифу — перезамер T6.6 шаг 3.
- Перезамер: `main` (`c16fe5e`) из временного `git worktree` в рабочем каталоге SDD с `PYTHONPATH=<worktree>/src` (путь импорта проверен перед каждым прогоном) и ветка `6fa1406`; `scripts/bench_throughput.py --workers 1 2 4 --duration 60 --warmup 15`; очерёдность main-1 → ветка-1 → main-2 → ветка-2; load average перед прогонами 0.89 / 8.22 / 8.75 / 9.03 (след предыдущего прогона серии, посторонних процессов нет). Worktree удалён после замера.

| Воркеры | main (1 / 2 / среднее) | ветка (1 / 2 / среднее) | ветка / main | порог 95 % |
|---|---|---|---|---|
| 1 | 4659 / 4536 / 4598 | 4872 / 4949 / 4911 | 106.8 % | ≥ 4368 |
| 2 | 9264 / 9415 / 9339 | 9220 / 9269 / 9244 | 99.0 % | ≥ 8872 |
| 4 | 14311 / 14419 / 14365 | 13977 / 14237 / 14107 | 98.2 % | ≥ 13647 |

Рост 4911 → 9244 → 14107 монотонный (у каждого из четырёх прогонов тоже). Первый замер T6.6 (до волны): 102.7 / 100.2 / 99.8 %. Таблица и JSON обновлены (`docs/benchmarks.md` «После SP3», `docs/benchmarks/sp3-throughput/`, коммит `f2f54ec`).

**PASS.**

## §3.6 Совместимость с SP2

Команды (P14: grep также по `SP2_CHECKPOINT`, `SP2_TTT_TINY`, `SP2_CONFIGS`, `copy_sp2_run`):
- `.venv/bin/python -m pytest $(grep -rlE "fixtures.{0,6}sp2|SP2_CHECKPOINT|SP2_TTT_TINY|SP2_CONFIGS|copy_sp2_run" tests --include="test_*.py") -v` — 9 файлов (`test_sp3_sp2_resume.py`, `test_sp3_example_configs_v3.py`, `test_sp3_sp2_compat.py`, `test_sp3_validate_players.py`, `test_sp3_neural_teacher.py`, `test_sp3_eval_players.py`, `test_sp3_init.py`, `test_sp3_players.py`, `test_sp3_distributed_guards.py`) → **136 passed in 8.87s**.
- `colosseum validate -c` для каждой из 11 копий `tests/fixtures/sp2/configs/*.yaml` (chase, coin_grid, coop_buttons, predator_prey, space_miners, team_tag, tic_tac_toe, tic_tac_toe_attention, tic_tac_toe_multi, tron, unit_harvest): **11 × OK**. Во всех 11 выводах есть предупреждение о переводе старых ручек (допустимо по критерию), например для `tic_tac_toe_multi`: `SP2 matchmaking knobs ['mode', 'self_play_ratio', 'latest_prob', 'pfsp_exponent'] translated to opponents={'latest': 0.0, 'snapshots': 0.0, 'rivals': 1.0, 'anchors': 0.0}, pfsp={'weighting': 'hard', 'exponent': 1.0}; write these in the config …`.

| Пункт | Тесты |
|---|---|
| Каждая копия конфига SP2 проходит `validate` без правок; фикстура покрывает все примеры | `tests/unit/test_sp3_sp2_compat.py::test_sp2_config_copies_validate_without_edits[*]` (11), `::test_the_fixture_has_a_copy_of_every_sp2_example_config`; `tests/unit/test_sp3_distributed_guards.py::test_sp2_configs_resolved_configs_and_the_sp2_kickstart_form_start` |
| Чекпоинт SP2 — настоящий чекпоинт кода SP2 | `tests/unit/test_sp3_sp2_compat.py::test_the_checkpoint_fixture_is_an_sp2_checkpoint_of_the_tiny_config` |
| Чекпоинт SP2 как frozen-агент | `tests/unit/test_sp3_players.py::test_the_sp2_checkpoint_fixture_loads_as_a_frozen_agent`; `tests/unit/test_sp3_validate_players.py::test_the_sp2_checkpoint_validates_as_a_frozen_agent`; `tests/unit/test_sp3_eval_players.py::test_eval_cli_plays_the_sp2_checkpoint_as_a_frozen_agent_by_name`; `tests/unit/test_sp3_sp2_compat.py::test_eval_plays_the_sp2_checkpoint_dir_and_leaves_it_untouched` |
| Чекпоинт SP2 как `init` | `tests/unit/test_sp3_init.py::test_the_sp2_checkpoint_fixture_works_as_init` |
| Чекпоинт SP2 для resume | `tests/unit/test_sp3_sp2_compat.py::test_resume_from_the_sp2_checkpoint_dir_restores_version_trainer_state_and_env_steps`; `tests/integration/test_sp3_sp2_resume.py::test_training_resumes_from_the_sp2_run_dir` (пул снимков SP2 переносится, источник не меняется) |
| Чекпоинт SP2 как учитель kickstart (P18) | `tests/unit/test_sp3_neural_teacher.py::test_the_sp2_checkpoint_fixture_works_as_a_kickstart_teacher` |
| Старые ручки переводятся, также частично заданные; resolved-конфиг SP3 перечитывается без предупреждений | `tests/unit/test_sp3_sp2_compat.py::test_the_resolved_form_of_an_sp2_config_rereads_without_warnings[*]`; `tests/unit/test_sp3_matchmaking_config.py::test_translate_sp2_knobs[*]`, `::test_loading_sp2_knobs_translates_once_and_never_stores_them`, `::test_sp2_knobs_together_with_new_keys_are_a_config_error`, `::test_set_accepts_the_sp2_knob_paths`; `tests/unit/test_sp3_warmstart_config.py::test_training_kickstart_knobs_are_translated_once_and_never_stored`, `::test_partially_set_knobs_keep_the_other_defaults`, `::test_knobs_together_with_a_kickstart_section_are_a_config_error`; `tests/unit/test_sp3_snapshot_retention.py::test_checkpoint_config_defaults_and_the_pool_size_translation` (последние три файла — в прогоне §3.2) |
| Живые примеры v3 ведут себя как перевод своих копий SP2 | `tests/unit/test_sp3_example_configs_v3.py::test_example_configs_behave_like_the_translation_of_their_sp2_copies[*]`, `::test_example_configs_use_no_sp2_knob[*]` |

**PASS.**

## §3.7 Дефолты APPO при `Units`

- `docs/benchmarks.md`, раздел «Дефолты APPO при `Units` (SP3, блок 10)»: таблица из 18 строк (K ∈ {8, 128} × три ячейки × сиды 0, 1, 2) и таблица решений.
- `docs/benchmarks/units-defaults-sp3.json`: `18 {"ratio_mode": [false, "per_unit"], "unit_trace": [false, "joint"]}` — 18 строк, два решения, оба «не переключать».
- Решения (правило блока 10, задано до замера): `ratio_mode` — при K = 128 `joint` лучше `per_unit` на +0.037 (0.186 → 0.223), меньше 2·SE = 0.092 → **оставить `per_unit`**; `unit_trace` — `none` хуже `joint` на 0.005 (2·SE = 0.055) → **оставить `joint`**; K = 8 насыщен (все прогоны 0.500). «Оставить» значит «разница не обнаружима на 3 сидах», а не «дефолт доказанно лучше» (ruling T6.5).
- Дефолт в коде: `tests/unit/test_appo_v2.py::test_modes_resolve_auto_and_collapse_for_one_decider` PASSED (строка 60: `resolve_modes(AlgorithmConfig(), units) == ("per_unit", "joint", "mean_valid")`); правило решения — `tests/unit/test_sp3_units_defaults_rule.py` 5 passed.

**PASS.** Открытые вопросы SP2 о `ratio_mode` и `unit_trace` закрыты этим правилом.

## §3.8 Документация

### Конвейер README и Quick start (дословно, из корня репо, в `env -i HOME=$HOME PATH=/usr/bin:/bin bash --noprofile --norc`)

Блок README «Competition pipeline» после волны исправлений — форма с одним конфигом: «скопируйте `configs/examples/unit_harvest_league.yaml` в `harvest.yaml` и назовите в нём выход BC до его появления». Правка конфига сделана скриптом ровно по фрагменту YAML из README (`agents.main: {init: {from: bc_net, critic_warmup_steps: 30}, kickstart: {teacher: greedy}}`, `agents.bc_net: {kind: frozen, path: bc.pt}`); затем команды README без изменений:

| Команда | exit | Время, с | README |
|---|---|---|---|
| `colosseum record -c harvest.yaml --player greedy --num-matches 300 --output data/greedy --seed 0` | 0 | 16.8 | ≈ 17 |
| `colosseum bc -c harvest.yaml --agent main --data data/greedy --output bc.pt --epochs 5 --seed 0` | 0 | 16.4 | было «≈ 9» → исправлено (см. ниже) |
| `colosseum train -c harvest.yaml --set run.name=harvest-league` | 0 | 124.0 | ≈ 126 |
| `colosseum eval -c harvest.yaml -a trained=$NEW -a greedy -a random -a bc_net --num-matches 100 --deterministic` (`$NEW` = `ckpt_v986`) | 0 | 40.5 | — |
| `colosseum validate -c configs/examples/tic_tac_toe.yaml` | 0 | 1.7 | — |
| `colosseum train -c configs/examples/tic_tac_toe.yaml --set run.name=ttt-quickstart` | 0 | 57.2 | 52–64 |

- `record`: 30 000 решений, 600 seat-episodes, 300 партий. `bc`: финальная NLL 0.0002. `train`: `init from frozen agent 'bc_net' (bc.pt) (strict): 20 tensors`. Предупреждений (`WARNING`) нет ни одного, в том числе о переводе старых ручек.
- `eval` (раскладка 2p, wdl): trained против `random` W/D/L 100/0/0 (win 1.000); trained против `greedy` 0/100/0 (score 0.500); trained против `bc_net` 0/100/0 (score 0.500); строки trained против greedy / random / bc есть. README: «beats `random` in 100% of games and scores 0.50 against `greedy` … all 100 games were draws» — совпадает.
- **Несовпадение и правка `docs:`** (`f2f54ec`): README говорил «BC about 9 s». Из CLI с потоками torch по умолчанию (8) `bc` идёт 16.4 с (эпоха ≈ 2.65 с); тот же `bc` с `OMP_NUM_THREADS=1` — 9.6 с. 9 с из `docs/benchmarks.md` замерены внутри процесса теста (один поток). README теперь: «BC about 16 s (about 10 s with `OMP_NUM_THREADS=1` …)», обучение «124–126 s». Ревью C (minor 3) приняло число 17 с за опечатку, хотя для CLI оно было ближе к правде. Замедление BC от 8 потоков записано в открытые пункты.
- Уборка: `rm -rf runs data bc.pt harvest.yaml`.

### Флаги, рецепты, ENV_GUIDE

- Флаги: скрипт сравнил каждый флаг после `colosseum <команда>` в README, `docs/LEAGUE_GUIDE.md`, `docs/ENV_GUIDE.md` и CLAUDE.md с `colosseum <команда> --help` (train, validate, eval, record, bc, serve-weight-store, run-learner, run-workers): **0 расхождений**. `record`, `bc`, `eval` принимают `--set` (волна, п. 1), `bc` — `--seed` (п. 15).
- `grep -n "^## " docs/LEAGUE_GUIDE.md`: 1 Понятия; 2 Один агент в self-play; 3 Скриптовый бот и якоря; 4 Замороженный агент (прошлый сабмит); 5 Лига из нескольких агентов; 6 Асимметричная игра; 7 Кооператив со скриптовым напарником; 8 Конвейер: `record` → `bc` → `init` → kickstart; 9 Расписания долей; 10 Override на агента; 11 Свой матчмейкер; 12 Хранение снимков; 13 Что печатает `validate`; 14 Метрики и рейтинги; 15 Ограничения; 16 Совместимость с SP2 — все рецепты блока 11 есть.
- `docs/ENV_GUIDE.md` §10 «`infos` для скриптовых ботов» есть, с ценой `rollout.vec_env: subprocess` (`StepResult` вместе с `infos` пиклится на каждом шаге).
- Правки повторного ревью волны (переданы в T6.7), коммит `f2f54ec`: (a) LEAGUE_GUIDE §16 — рост диска ≈ `шаги обучения / (checkpoint.interval · keep_every)` снимков; (b) CLAUDE.md — `--set` также у `record` / `bc` / `eval` (и формулировка слияния секций агента); (c) ENV_GUIDE §8 — свой `update_normalizers` должен принимать `obs=None` (прогрев критика вызывает `update_normalizers(None, global_state)`).
- README «Status and limitations» и CLAUDE.md «Implementation Status» / «Roadmap» / открытые пункты сверены с шагами 1–7: счёт тестов, slow-набор (10 тестов, ≈ 15 мин), team_tag одним прогоном (доля 0.2, played ≈ 0.36), пороги конвейера, дефолты APPO, распределённые отказы. Исправлено в коммите отчёта: статус SP3 («implemented on `sp3-league`, acceptance report written, waiting for the owner's acceptance» вместо «(done)» в Roadmap), даты отчёта и папки ревью вместо `<date>`, строка скорости (`f2f54ec`), «Parked during SP3» дополнен мелочами ревью вне волны и находками приёмки.

**PASS.**

## §3.9 Распределённый режим

- `.venv/bin/python -m pytest tests -v -k "distributed or grpc"` → **50 passed, 1671 deselected in 10.94s**: SP2 (`tests/integration/test_sp2_distributed_e2e.py::test_distributed_grpc_pipeline`, `tests/integration/test_sp2_grpc.py::*`, `tests/unit/test_sp2_distributed_roles.py::*`, `tests/unit/test_sp2_entry_points.py::test_distributed_roles_reject_the_config_before_serving_or_spawning`, …) и SP3 (`tests/unit/test_sp3_distributed_guards.py::test_sp3_settings_are_refused_with_a_pointer_to_sp5`, `::test_scripted_teachers_and_any_init_are_refused`, `::test_any_mix_but_latest_only_is_reduced_with_one_warning`, `::test_the_one_warning_also_says_that_mixed_teammates_are_ignored`, `::test_entry_points_refuse_before_a_run_dir_exists`, `::test_the_distributed_learner_keeps_snapshots_like_the_local_one`, `::test_values_decide_not_the_presence_of_a_key`; `tests/unit/test_sp3_learner_factory.py::test_the_distributed_learner_builds_through_the_factory`).
- Localhost smoke (порты 50051/50052, `run.dir` в рабочем каталоге SDD, `training.total_timesteps=20000`): `serve-weight-store` → `run-learner -c configs/examples/tic_tac_toe.yaml --agent agent_0` → через 10 с `run-workers … -l agent_0=localhost:50052` → **`workers exit=0`**. У лёрнера одно предупреждение «distributed mode plays every seat with the agents' latest weights … matchmaking.opponents {'agent_0': {'latest': 0.8, 'snapshots': 0.2, …}} reduced to latest only» (у воркеров — своё одно); после `kill` лёрнер логирует «stopped by SIGTERM» и пишет финальный `ckpt_v78` (`final: true` в `meta.json`; 78 шагов обучения < `checkpoint.interval` 100, поэтому промежуточных снимков нет — правила `keep_last` / `keep_every` распределённого лёрнера проверяет `test_the_distributed_learner_keeps_snapshots_like_the_local_one`). Процессы остановлены.
- При остановленном weight store (проверка до любого соединения):
  - `run-workers -c configs/examples/team_tag.yaml …` → `exit=1`, `Config error: distributed mode (run-learner / run-workers) keeps the SP2 scope until SP5 (a hub and a league across machines); train this config with 'colosseum train' on one machine, or remove: scripted/frozen agents ['random'] (anchors, fixed opponents, teachers)`;
  - `run-learner -c configs/examples/unit_harvest_league.yaml --agent main …` → `exit=1`, та же строка с `scripted/frozen agents ['greedy', 'random']`.
- Каталог smoke удалён.

**PASS.**

## Замеры

### team_tag: доля якоря (T6.3) и проверка приёмки (шаг 4)

T6.3 (сиды 0–5, `docs/benchmarks/team-tag-anchors.jsonl`, 18 прогонов): anchors 0.1 — min 0.915, mean 0.931; 0.2 — min 0.945, mean 0.965; 0.3 — min 0.930, mean 0.953. Все проходят 0.80; наибольший минимум у 0.2, минимум 0.3 в пределах 0.02 → по правилу ничьей выбрана меньшая доля **0.2** (ruling T6.3). Приёмка (сиды 10–15, `docs/benchmarks/team-tag-acceptance.jsonl`): min 0.905, mean 0.942 — таблица в §3.4.

### Конвейер на unit_harvest: калибровка порогов (T6.4) и конвейер против обучения с нуля

Пороги (ruling T6.4: `max(floor, floor_0.05(min − 0.05))` по сидам 0/1/2, точные десятичные): win против `random` 0.95 (калибровка 1.000 / 1.000 / 1.000); score против `greedy` 0.40 (0.483 / 0.500 / 0.475); score против `bc_net` 0.45 = нижняя граница (0.483 / 0.500 / 0.483). Бюджет: 160 000 шагов сред, 2 воркера, `record` 300 партий, `bc` 5 эпох, прогрев критика 30 шагов, kickstart от бота (DAgger, λ = 1, затухание 300 шагов). Приёмка: 1.000 / 0.492 / 0.483 (§3.3). Конвейер против обучения с нуля — таблица в §3.3.

### Дефолты APPO при `Units` (T6.5)

§3.7. 18 прогонов за 1 ч 50 мин (`scripts/units_experiment.py --cells per_unit:auto joint:auto per_unit:none --seeds 0 1 2 --steps 340000`).

### Скорость (T6.6 и перезамер приёмки)

§3.5. T6.6 (код `aea876e`): 4815 / 9361 / 14090 = 102.7 / 100.2 / 99.8 % от `main`. Приёмка (код `6fa1406`): 4911 / 9244 / 14107 = 106.8 / 99.0 / 98.2 %.

## Отклонения и замечания

1. **Порядок шагов.** Финальное ревью ветки и волна исправлений прошли до приёмки (ruling, журнал строка 181); раздел «Финальное ревью ветки» описывает их, повтор критерия 3.1 после волны не нужен — §3.1 и есть прогон после волны.
2. **Артефакты не в `/tmp`.** По указанию контроллера все прогоны писали в `.superpowers/sdd/2026-10-10-sp3-league/t67-runs/` (git-ignored): логи наборов, `--run-dir` team_tag, `run.dir` распределённого smoke, worktree `main` для замера скорости (удалён).
3. **Медленный тест конвейера** после волны идёт с одного конфига и с BC с сидом (пункты 1 и 15), а пороги T6.4 калибровались на BC без сида. Тест прошёл с запасом 0.05 / 0.092 / 0.033; пороги не трогались.
4. **Скорость перемерена** (шаг 5 брифа): волна меняла каталоги горячего пути. Новые числа заменили JSON и таблицу (`f2f54ec`); прежние числа T6.6 названы в `docs/benchmarks.md`.
5. **Конвейер README** — форма с одним конфигом (после волны). Шаг «скопируйте и впишите» выполнен скриптом по фрагменту README; остальные команды дословно. Время BC в README исправлено (`f2f54ec`).
6. **Распределённый smoke**: на 20 000 шагов лёрнер успевает 78 шагов обучения, меньше `checkpoint.interval` (100), поэтому в папке только финальный снимок; правила хранения распределённого лёрнера проверены тестом (§3.9).
7. **coin_grid** в slow-наборе — 4.01× (порог 2×), ниже предпроверки T6.4 (5.37×); два повтора на том же коде 5.31× и 3.76× — разброс прогонов, не регрессия (§3.3).
8. **Win-rate Quick start** tic-tac-toe в шаге 8 не оценивался (бриф требует exit 0 и время; время 57 с в диапазоне README 52–64 с); обучение tic-tac-toe проверено slow-тестом (0.927).
9. Поправки контракта A1–A33 и отклонения плана от спеки (нумерация частей, теги категорий на каждом месте, `infos` без отдельного хранения, удалённый `coordinator/matchmaker.py`, `validate` возвращает `ValidationReport`) — в `docs/superpowers/plans/2026-10-10-sp3-league/00-overview.md` («Deviations from the spec», «Contract amendments»); на приёмку не влияют.
10. **Два решения волны уточняют спеку** (вынесены владельцу): прогрев критика обновляет нормализаторы пути value (`global_state`) — спека §8 п. 7 говорила «статистика нормализаторов не обновляется» (цель спеки — побитово неизменная политика — сохранена); `record` / `bc` / `eval` проверяют конфиг в «play»-области (без проверок тёплого старта и матчмейкинга), чтобы конвейер шёл с одного конфига.

## Гигиена

- После шагов 1–11 и уборки `git status --porcelain` содержит только файлы этого отчёта (отчёт, папку ревью, `team-tag-acceptance.jsonl`, `README.md`, `CLAUDE.md`), всё уходит в коммит отчёта.
- `git status --porcelain --ignored` без `.venv`/кэшей/`.superpowers/`: только эти же файлы — тестовых или прогонных остатков нет (`runs/`, `data/`, `bc.pt`, `harvest.yaml` удалены).
- `ps -eo pid,args | grep -E "colosseum|multiprocessing|bench_throughput|team_tag_anchors|units_experiment"`: **no leftover processes**.
- `git worktree list`: только основной (`/home/viv/dev/repos/Colosseum`); временный worktree `main` для замера скорости удалён (`git worktree remove`).

---

## Решения контроллера во время SP3 (все записи «Ruling:» из журнала, по порядку)

Каждая запись — дословно из журнала контроллера `.superpowers/sdd/2026-10-10-sp3-league/progress.md` (каталог удаляется после слияния): что решено — почему — чем грозит, если решение неверно. Там, где журнал не называет цену ошибки, в квадратных скобках добавлена оценка. Всего 54 записи.

### Решения этапа плана (поправки контракта A1–A33)

Обязательные поправки контракта A1–A33 записаны в `docs/superpowers/plans/2026-10-10-sp3-league/00-overview.md`, раздел «Contract amendments» (там же «Deviations from the spec»). Они фиксируют имена и сигнатуры между частями плана (A1 `build_frozen_model`, A2 `check_bot_action`, … A28 `eval.load_player`, A31 `check_distributed_scope`, A33 зависимости T6.5) и согласованы до кода; цена ошибки у каждой — переделка интерфейса между задачами, пойманная набором тестов.

### Предполётные решения P1–P18 (скан плана, `constraints.md`)

Строки журнала P1–P18 — это тексты P1–P18 из `constraints.md` дословно, плюс причина и цена ошибки.

1. (журнал, строка 13) Ruling: P1 (T3.1, T4.1) T1.1's `TrainableAgent._not_supported_yet` guard is lifted step by step: T3.1 removes "matchmaking" from its field list and deletes the matchmaking case from `tests/unit/test_sp3_agent_kinds.py::test_bad_agent_entries_are_config_errors_with_a_hint`; T4.1 deletes `_not_supported_yet` (and the `ValidationInfo` import if unused) and deletes the init and kickstart cases from that test — otherwise T3.1/T4.1 per-agent tests fail against the guard — cost if wrong: a stray guard or test case; caught by the suite
2. (журнал, строка 14) Ruling: P2 (T6.6) When T6.6 rewrites `tic_tac_toe_multi.yaml` to the v3 form, it updates `tests/integration/test_sp2_league_runs.py`: line 51 `--set matchmaking.mode=self_play` becomes `"matchmaking.opponents": "{latest: 0.8, snapshots: 0.2, rivals: 0.0, anchors: 0.0}"`; lines 75–76 (`matchmaking.mode=league`, `self_play_ratio=0.0`) are dropped (the v3 config already has `rivals: 1.0`) — old knob + new key = ConfigError would break two SP2 integration tests — cost if wrong: a red suite at T6.6, caught by the suite
3. (журнал, строка 15) Ruling: P3 (T3.1, T6.6) `translate_sp2_matchmaking` rounds each of the four shares to 12 decimals (`round(x, 12)`); T3.1's tests keep `pytest.approx`; T6.6 keeps its exact `==` comparison. — float noise (0.19999999999999996) leaks into warnings, resolved configs and breaks T6.6's equality test — cost if wrong: cosmetic only
4. (журнал, строка 16) Ruling: P4 (T3.2) T3.2 Step 7c is applied to T1.5's `validate_config` (not SP2's): set `player_roles = resolve_player_roles(config, spec)` before `validate_matchmaking(spec, player_roles, config)`, and replace `layouts = list(enabled_layouts(...))` with `layouts = _all_enabled_layouts(config, spec)`, keeping both of its uses. — the plan step was written against the pre-T1.5 code (NameError) — cost if wrong: an implementer adapting differently; caught by review
5. (журнал, строка 17) Ruling: P5 (T1.3, T4.5) T1.3 sets `ActRecord.info=(env_state.result.infos or {}).get(seat)` for EVERY neural seat too (A15) and documents it; T4.5 does not touch `match_runner.py` for that. — A15 vs the plan code of T1.3 — cost if wrong: none
6. (журнал, строка 18) Ruling: P6 (T3.3, T6.2) Anchor-participation assertions use the `games` matrix, not ELO keys (after T2.3 `RatingBook.snapshot()` lists every player's ELO even unplayed): T3.3 asserts `games["agent_0"]["bot"] > 0`; T6.2's smoke asserts `games["main"]` against the bot, random and bc_net is > 0. — the planned assertions cannot fail — cost if wrong: weaker tests if mis-applied
7. (журнал, строка 19) Ruling: P7 (T5.1) `eval.load_player` raises `ConfigError` for a path that is neither a directory nor a `.pt` file (no FileNotFoundError traceback). — user-facing error quality — cost if wrong: none
8. (журнал, строка 20) Ruling: P8 (T5.1) `record` without `--against` and without `--layout` skips layouts the player cannot fill alone and logs ONE INFO line naming them; `ConfigError` ("add --against") only if no layout is left or an explicit `--layout` cannot be filled. — spec block 7 read for the default-layout case (all layouts by default would make a role-limited bot unusable) — cost if wrong: a user expecting an error sees an INFO line
9. (журнал, строка 21) Ruling: P9 (T4.1) `learner.factory.agent_ids` returns `config.agent_ids()` (no second definition with another order). — DRY / one order — cost if wrong: none
10. (журнал, строка 22) Ruling: P10 (T6.6) T6.6 updates README's `metrics.jsonl` table: `critic_warmup`, `kickstart_label_frac` in `train`; `opponents` in `episodes`; per-layout `pfsp` and scripted/frozen entities in `ratings`. — docs of the real state — cost if wrong: none
11. (журнал, строка 23) Ruling: P11 (T2.3) The console's arena win rate (`wr_arena_over_layouts`) averages only over trainable opponents (anchors excluded). — keep the SP2 meaning of 'arena' — cost if wrong: anchor wins missing from the console line (they are in ratings)
12. (журнал, строка 24) Ruling: P12 (T4.2, T4.3) Accepted: workers act with their own random init until the first weight sync (SP2 behaviour, V-trace corrects it); critic warm-up keeps the LEARNER's policy bit-identical — SP2 behaviour unchanged; fixing it is out of scope — cost if wrong: a few chunks of the very first seconds are collected by a random policy
13. (журнал, строка 25) Ruling: P13 (T6.2) The fast pipeline smoke stays in the fast suite (spec); its duration is measured and recorded as a ruling in the T6.2 report. — spec section 6 puts it in the fast suite — cost if wrong: a slower CI suite
14. (журнал, строка 26) Ruling: P14 (T6.7) T6.7 Step 6's grep for compatibility tests also matches `SP2_CHECKPOINT`, `SP2_TTT_TINY`, `SP2_CONFIGS` and `copy_sp2_run`. — otherwise most compat tests are missed — cost if wrong: none
15. (журнал, строка 27) Ruling: P15 (T4.5, T6.2) `tests/learning/sp3_bandits.py` is allowed (sp2_bandits precedent); T4.5 uses `GameFactory((SOLO, 1), infos=True)` / `TickGame(infos=True)` from game_helpers instead of a new `InfoGame`. — no duplicate toy games — cost if wrong: none
16. (журнал, строка 28) Ruling: P16 (T4.5) The launcher builds `scripted_teachers` once before the worker loop (keeps lines ≤120, ruff E501). — ruff — cost if wrong: none
17. (журнал, строка 29) Ruling: P17 (T1.5, T5.1) `cli._config_errors` also catches `PlayerError` and prints one line "Player error: ..." (exit 1); T5.1 drops its own try/except for it. — one-line user errors for bot failures — cost if wrong: none
18. (журнал, строка 30) Ruling: P18 (T4.4) T4.4 adds a short test in `test_sp3_neural_teacher.py`: the SP2 fixture checkpoint (`SP2_CHECKPOINT`) works as a kickstart teacher. — part 01's notes promised it; spec compat (SP2 checkpoints usable) — cost if wrong: none

### Решения по задачам T0.1–T6.6 (включая замеренные значения: доля якоря T6.3, пороги и бюджеты T6.4, дефолты T6.5, скорость T6.6)

19. (журнал, строка 5) Ruling: T0.1 starts while the pre-flight scan runs — T0.1 only adds SP2 fixtures and baseline tests on unchanged SP2 code, independent of later tasks; any pre-flight finding on it becomes a fix round — cost if wrong: one extra fix round on T0.1.
20. (журнал, строка 38) Ruling: T1.2 eval error order — load_eval_model now loads weights (build_frozen_model) before _check_model, so "do not match" is reported before a step failure and _check_model runs on the loaded model — one shared frozen loader (spec/plan intent); framework models do not change state in step — cost if wrong: a custom model_class that mutates state in step could alter loaded eval weights; a different first error message.
21. (журнал, строка 43) Ruling: T1.3 minors 1–2 are fixed in T1.4 (carried forward) instead of being deferred — MatchRunner passes COPIES of obs and mask to bot.act (a bot editing its mask in place could bypass the single legality gate; editing obs in place would corrupt record/BC data), and rejects a ScriptedPlayer seat with collect=True (ValueError; spec block 3: only a trainable agent's latest collects) — integrity of the legality gate and of recorded data — cost if wrong: one extra copy per scripted decision.
22. (журнал, строка 49) Ruling: T1.4 minor 1 (MatchRunner._resolve fallback could seat a bot served under "latest" as latest+collect for a snapshot seat in eval/record pools) is fixed in T1.5 (carried forward): the fallback raises ValueError when the (agent, "latest") pool entry is a ScriptedPlayer — closes the "scripted never collects" guarantee before T3.2/T5.1 build such lineups — cost if wrong: none (unreachable today).
23. (журнал, строка 55) Ruling: T1.5 Important (plan-mandated): _check_scripted_players duplicates _exercise_env's reset/step error wrapping (validation.py:369-375 vs 139-144, 409-413 vs 155-161) — deferred to the final whole-branch fix wave (extract _reset_layout/_step_env helpers) — maintainability only, no behaviour risk; a fix round now costs a full suite + re-review — cost if wrong: duplicated error wording until the fix wave.
24. (журнал, строка 60) Ruling: T2.1 minors 1–2 fixed in T2.2 (carried forward): CheckpointManager._retain/import update the index BEFORE calling on_evict callbacks (a raising callback must not leave the index listing deleted dirs; T2.2/T2.3 give on_evict real work); a resume (apply_resume_state / _resolve_resume) from a snapshot without trainer_state.pt while checkpoint.save_optimizer is on logs a WARNING (an imported-only run dir would otherwise silently drop optimizer/LR state) — robustness before on_evict does real work — cost if wrong: none.
25. (журнал, строка 80) Ruling: T3.2 minor 1 (zero-weight anchors still fill seats of roles no trainable agent plays) is fixed in T3.3 (carried forward): validate_matchmaking checks role coverage by trainable agents or anchors with POSITIVE weight at EVERY schedule point (ConfigError otherwise), and MixtureMatchmaker._role_player picks only anchors with weight > 0 at the current env step — spec: no silent fallbacks beyond the named runtime one — cost if wrong: a config with a role covered only by a decaying anchor is rejected (user adds a weight floor).
26. (журнал, строка 81) Ruling: T3.2 minor 2: T3.3 also ports the two dropped SP2 matchmaker guarantees as tests (FFA 15p: each opposing core drawn independently; teammates: mixed inside an OPPONENT team) — cheap regression net for _fill_team. [цена ошибки в журнале не записана; оценка контролёра приёмки: нет — только дополнительные тесты]
27. (журнал, строка 82) Ruling: SP2 configs with ONE trainable agent and mode: league + self_play_ratio: 0.0 translate to rivals: 1 only and now raise ConfigError (no fillable category) where SP2 silently ran self-play — the spec's config check requires it; no SP2 fixture config hits it — documented in LEAGUE_GUIDE's compatibility notes (T6.6) — cost if wrong: such a user config needs `opponents` written explicitly.
28. (журнал, строка 94) Ruling: kickstart teacher paths must be `.pt` files or checkpoint dirs (factory.py) — SP2 loaded any file path; matches the spec wording and resume/eval conventions — documented in LEAGUE_GUIDE compatibility notes (T6.6) — cost if wrong: an SP2 config with a `.pth` teacher must rename the file.
29. (журнал, строка 100) Ruling: T4.2 Important (plan-mandated, brief Step 4b): validate_config resolves init for agents that training.resume_from will restore, so a resume with the same config fails when the init source is gone — spec block 6 (resume wins, "the run continues with the same config") is binding over the plan — fix round 1: skip init for restored agents in validate (report line instead) — cost if wrong: none.
30. (журнал, строка 101) Ruling: T4.2 minor 2 fixed in the same round: a checkpoint used as init through a frozen-agent NAME is checked against the student's role signature too (name and path give the same answer). [цена ошибки в журнале не записана; оценка: нет — та же проверка сигнатуры, что и для пути]
31. (журнал, строка 114) Ruling: T4.4 minors 1 and 3 fixed in T4.5 (carried forward; T4.5 edits KickstartLoss): an end-to-end regression test of a STATELESS teacher with a RECURRENT (LSTM) student through validate_config + build_algorithm + train_step; check_teacher_state_layout and KickstartLoss use ONE "stateless" predicate (model.is_stateful) — the headline SP3 rule change deserves a config-level test; one predicate avoids an edge case — cost if wrong: none.
32. (журнал, строка 125) Ruling: T5.1 minors 1 and 5 fixed in T5.2 (carried forward): `record --against` rounds an odd --num-matches up to even for two-team layouts with several players, like eval (balanced sides; note in help/log), and a test drives RoleWriter with a small decisions_per_file (seat-episodes never split across files; part-00001.pt named and listed in record.json) and loads the multi-file output through bc — T5.2 depends on multi-file loading; spec "as in eval" — cost if wrong: none.
33. (журнал, строка 140) Ruling: play_lineups (in-process eval API) will force collect=False on every seat in the final fix wave (eval never collects; SeatAssignment's default collect=True makes a scripted seat raise) — usability trap found in T6.2 — until then T6.3/T6.4 pass collect=False explicitly for scripted seats — cost if wrong: none.
34. (журнал, строка 141) Ruling: P13 measured — the tic-tac-toe pipeline smoke takes ~7.1–7.6 s alone (not among the 15 slowest in the full suite); stays in the fast suite at total_timesteps=3000. [цена ошибки — как у P13: более медленный быстрый набор; смоук на этой приёмке шёл 6.1 с]
35. (журнал, строка 142) Ruling: T6.2 Important (plan-mandated verbatim duplication of newest_checkpoint/_CKPT_RE in tests/pipeline_kit.py and tests/learning/demo_learning.py) is fixed now in fix round 1 with minors 1–3 (smoke asserts critic_warmup==1 and kickstart_lambda>0 in train records; train_in_process uses algo.teacher_active like the learner; pipeline_kit.cli docstring notes restore_root_logging for in_process=True) — cheap, prevents drift before T6.4 reuses the kit — cost if wrong: none.
36. (журнал, строка 143) Ruling: the smoke runs record/bc/eval in-process via click's CliRunner and only train through tests/cli_runner.py — the plan mandates it and only process starts must use cli_runner — cost if wrong: none.
37. (журнал, строка 150) Ruling: team_tag anchors share = 0.2 (opponents latest 0.6 / snapshots 0.2 / rivals 0 / anchors 0.2 with a RandomBot agent `random`) — chosen by the pre-fixed T6.3 rule from 18 sequential runs (shares 0.1/0.2/0.3 × 6 seeds, docs/benchmarks/team-tag-anchors.jsonl): min/mean win vs random team 0.915/0.931, 0.945/0.965, 0.930/0.953; all pass ≥0.80, 0.3 within 0.02 of the best min → smaller share 0.2; no run passive (draws 2–8.5 %, episode length 6.4–10.1); fallback (draw penalty) not needed; mean training 173.6 s — cost if wrong: team_tag passivity could reappear on other seeds (acceptance re-checks 6 new seeds in T6.7).
38. (журнал, строка 151) Ruling (clarification of the team_tag ruling): 0.2 is the DRAW share of opponents.anchors; the PLAYED share is higher (≈0.36 in run a0.2-s0) because lineups persist per refresh and games vs the random team are short — measurement and slow test use the same semantics, so the choice stands; LEAGUE_GUIDE (T6.6) explains drawn vs played shares — cost if wrong: none.
39. (журнал, строка 158) Ruling: unit_harvest pipeline thresholds (T6.4 rule max(floor, floor_0.05(min−0.05)) over calibration seeds 0/1/2, computed in EXACT decimals — plain float gives 0.90 for 0.95 through 18.999…): win vs random 0.95 (runs 1.000/1.000/1.000), score vs greedy 0.40 (0.483/0.500/0.475), score vs bc_net 0.45 = floor (0.483/0.500/0.483; min over 7 pipeline runs 0.467) — the rule is fixed in the plan; exact decimals match its intent — cost if wrong: the bc_net check has a 0.017 margin, a flaky slow test is possible (acceptance reruns it; a failure becomes a measured ruling, thresholds never lowered silently).
40. (журнал, строка 159) Ruling: pipeline vs scratch (unit_harvest, 3 seeds, equal budget) is REPORTED, not asserted (spec §3.3): mean score vs the scripted bot 0.483 (pipeline) vs 0.492 (scratch) — both reach the game's ceiling (ties with the bot, as in SP2); the warm start gives no gain on this game and budget and costs ~129 s vs ~92 s — an honest finding for the acceptance report (warm start's value must be shown on harder games) — cost if wrong: none.
41. (журнал, строка 161) Ruling (addendum to the unit_harvest thresholds): flakiness estimate of the bc_net check — P(fail) ≈ 0.2 % at the pooled loss rate, ≈ 10 % at the worst run's rate (4/60); on this game the slow test cannot tell a working warm start from none (scratch reaches the same ceiling), so init/kickstart correctness rests on the fast and contract tests (T4.2–T4.5, T6.2 smoke) — documented in the acceptance report.
42. (журнал, строка 166) Ruling: APPO Units defaults stay (ratio_mode auto = per_unit, unit_trace auto = joint) — by the pre-fixed spec block-10 rule over 18 runs (unit_harvest, K ∈ {8,128}, 3 seeds): at K=128 joint ratio beats per_unit by +0.037 (0.223 vs 0.186) < 2·SE 0.092; unit_trace none is 0.005 worse than joint (2·SE 0.055); K=8 is at the 0.500 ceiling in every run — "keep" means no detectable difference with 3 seeds (seed SD up to 0.073), not that the defaults are proven better; closes SP2 open questions 2–3 — cost if wrong: a few points of share_vs_scripted on games with many units (users can set ratio_mode: joint).
43. (журнал, строка 167) Ruling: T6.5 grid launched at 1-min load 1.41 (brief precondition < 1.0) right after the smoke run — accepted: the first run's throughput (1820 steps/s) matches its siblings (1802–1832), no visible effect — cost if wrong: none.
44. (журнал, строка 172) Ruling: throughput criterion 3.5 passes — main and the branch alternated (2× each) on tic_tac_toe: branch 4815 / 9361 / 14090 env steps/s at 1/2/4 workers = 102.7 / 100.2 / 99.8 % of main, monotone (docs/benchmarks/sp3-throughput/). [цена ошибки в журнале не записана; перепроверено на этой приёмке перезамером после волны, §3.5]
45. (журнал, строка 174) Ruling: throughput measurement preconditions accepted — load < 1.0 only before the first run (8.6–10.1 lag of the previous run before the others) and the branch measured from the working tree identical in src to aea876e; alternating main/branch mitigates both; result 99.8–102.7 % — cost if wrong: none (criterion has 5 % slack; spread within a version up to 5.9 %).
46. (журнал, строка 175) Ruling: «Parked during SP3» in CLAUDE.md is a SELECTION of the most relevant deferred items by addressee; the complete list of deferred minors goes into the acceptance report appendix (T6.7), as in SP2 — keeps CLAUDE.md readable — cost if wrong: none.

### Решения по финальному ревью и волне исправлений

47. (журнал, строка 181) Ruling: the whole-branch final review and its single fix wave run BEFORE T6.7 acceptance (SP2 did acceptance first and re-ran criterion 3.1 after the wave) — the acceptance measurements (6 team_tag seeds, slow suite, fast suite) then run once on the final code — cost if wrong: none (same gates, less rework).
48. (журнал, строка 184) Ruling: one-config pipeline (A-I1/C-I1) — record/bc/eval validate in a "play" scope (skip warm-start checks of trainable agents, load only the fixed players the command uses) and get --set; train/validate/distributed keep the full check — spec blocks 1/6 promise one config for the whole pipeline; owner's user-friendliness priority — cost if wrong: a broken init/teacher path is reported at train instead of record/bc (still before any process starts).
49. (журнал, строка 185) Ruling: critic warm-up updates the VALUE-PATH (global-state/critic-encoder) normalizers; observation normalizers stay frozen (C-I2) — refines spec block 6 ("normalizer statistics are not updated"), whose stated purpose (learner's policy bit-identical) is kept; otherwise the critic's inputs jump at the end of warm-up — cost if wrong: none for the policy; owner may revert to freezing all normalizers.
50. (журнал, строка 186) Ruling: PFSP statistics of evicted snapshots never come back (B-I1): forgotten keys are remembered and skipped — spec "statistics of deleted snapshots are reset" — cost if wrong: none.
51. (журнал, строка 187) Ruling: LEAGUE_GUIDE lists every SP2 behaviour change from spec §8 (B-I2/C-I3) — docs of the real state — cost if wrong: none.
52. (журнал, строка 188) Ruling: final fix wave = fix-wave-brief.md items 1–18 (4 Important + 14 cheap minors chosen from the reviews' "fix now" lists); every other deferred minor stays parked and goes into the acceptance report appendix. [цена ошибки в журнале не записана; оценка: неисправленные мелочи остаются в приложении и в «Parked during SP3»]
53. (журнал, строка 191) Ruling: the play scope of validate (record/bc/eval) also skips the matchmaking checks (they need every anchor's roles = reading every frozen agent's file); validate/train keep them — cost if wrong: a bad anchor is reported at validate/train, not at record/bc/eval.
54. (журнал, строка 193) Ruling: play-scope validation reports a bad matchmaking layout with the env-contract wording (exit 1 unchanged) — accepted cost of the one-config ruling.


## Открытые пункты (отложены, не блокируют слияние)

Совпадают с «Parked during SP3» в CLAUDE.md (там — выборка по адресатам; полный список отложенных мелочей — в приложении ниже). Пункты, отложенные в SP1 и SP2, перечислены в CLAUDE.md («Open items parked during SP1, SP2 and SP3») и этой приёмкой не менялись; открытые вопросы SP2 о `ratio_mode` / `unit_trace` (закрыты правилом T6.5), о проверке team_tag (теперь один прогон) и об обратной совместимости (сделана) закрыты.

1. **SP4:** рейтинги снимков и хранение top-k (пока `keep_every` / frozen-агенты); онлайн-ELO остаётся индикатором прогресса; каждый `ratings` в `metrics.jsonl` повторяет полную таблицу PFSP (сжать); `system` `dropped_reward_episodes` по-прежнему суммирует только недавних воркеров; ценность тёплого старта нужно показать на игре, где обучение с нуля не доходит до бота (на `unit_harvest` выигрыша нет).
2. **SP5:** вся лига, скриптовые/frozen игроки, тёплый старт и свой матчмейкер в распределённом режиме (сейчас — `ConfigError` с указанием на SP5 и сведение смеси к latest); предупреждение «reduced to latest only» печатается до файлового логирования и срабатывает, когда долю несут только пустые категории; `validate_chunk_payload` проверяет у полей DAgger только наличие и форму (класс доверия к пиру); предупреждение перевода `pool_size` печатается до появления папки запуска.
3. **Поведение, мелкое:** подмена отсутствующего снимка на latest сохраняет `source="snapshots"` в сыгранных долях (описано в LEAGUE_GUIDE); `validate` играет ботами только во включённых раскладках и проверяет свой матчмейкер только на шаге 0 без снимков; `validate` не проверяет веса/сигнатуры продолжаемых агентов; `PfspStats` не проверяет prior; целые компоненты действия принимают `bool`; `FrozenSpec ==` падает на numpy; проверка сигнатуры прогрева отвергает подклассы APPO с `**kwargs`; общие параметры value/политики прогрев не обнаруживает (документированный контракт); `record` оставляет частичные файлы после сбоя; запас проверки `bc_net` в тесте конвейера 0.017–0.033 (оценка срыва 0.2–10 %, ruling T6.4); правило ничьей team_tag сравнивает без допуска на float; `kickstart.kl: reverse` молча игнорируется для учителя-скрипта; resume из снимка без `trainer_state.pt` повторяет прогрев критика и перезапускает λ kickstart (не документировано); воркеры спрашивают учителя-скрипта во время прогрева, хотя метки не используются; `bc` не сверяет игру / `env.kwargs` из `record.json` со своим конфигом; frozen-агенты, нужные только как учитель или `init`, строятся в каждом воркере; `eval`/`record` применяют `.pt`-настройки frozen-записи к пути из `-a name=path`; сыгранные доли не пишутся для команд со смешанными обучаемыми агентами, атрибуция якоря может делиться при `teammates: mixed`; финальный снимок каждого запуска хранится всегда и переносится каждым resume из папки запуска.
4. **Найдено при приёмке:** `colosseum bc` с потоками torch по умолчанию (8) идёт в ≈ 1.7 раза дольше, чем с `OMP_NUM_THREADS=1` (16 с против 10 с на конвейере README) — задать число потоков torch в процессе BC; отношение медленного теста coin_grid меняется между прогонами одного кода 3.8–5.4× (порог 2× держится; для сравнения «до/после» одного прогона мало).
5. **Качество кода, без риска для поведения:** дублирование поиска чекпоинтов в `learner/factory.py`, `record.parse_player` против `cli._parse_agent_spec`, предикат округления «соревновательной раскладки» в `eval` и `record`, обработка бота-учителя в `RolloutLoop` параллельна `MatchRunner`, логика W/D/L в трёх местах; повторные разборы ролей / `meta.json` / учителей / `init` (validate + launcher); недостающие тесты (см. приложение).
6. **GPU:** 22 GPU-теста (`docs/GPU_CHECKS.md`, включая новые `tests/unit/test_sp3_gpu_warmstart.py`) ещё не прогнаны на CUDA.

## Вопросы владельцу

1. **Принять SP3 и слить `sp3-league` в `main`?** (`git merge --no-ff`; шаги 14–15 брифа выполняются только после явного «да» владельца.)
2. **Подтвердить два решения волны исправлений, уточняющие спеку** (по умолчанию действуют):
   - прогрев критика обновляет нормализаторы пути value (`global_state` / критик-энкодер), нормализаторы наблюдений заморожены — иначе входы критика «прыгают» в конце прогрева (спека §8 п. 7 говорила «статистика нормализаторов не обновляется»; можно вернуть полное замораживание);
   - `record` / `bc` / `eval` проверяют конфиг в «play»-области: без проверок тёплого старта обучаемых агентов и матчмейкинга, загружая только используемых frozen-агентов — так весь конвейер идёт с одного конфига; ошибка в `init` / учителе / якоре теперь видна на `validate` / `train`, а не на `record`.
3. **GPU-проверки:** прогнать `docs/GPU_CHECKS.md` (22 теста) на машине с CUDA — по-прежнему ждёт владельца.

---

## Финальное ревью ветки

Три ревьюера по областям (Opus/high, только чтение, диапазон `c16fe5e..c31eda9`), отчёты: [`2026-10-10-sp3-review/final-review-A.md`](2026-10-10-sp3-review/final-review-A.md), [`final-review-B.md`](2026-10-10-sp3-review/final-review-B.md), [`final-review-C.md`](2026-10-10-sp3-review/final-review-C.md); бриф волны — [`fix-wave-brief.md`](2026-10-10-sp3-review/fix-wave-brief.md), отчёт волны — [`fix-wave-report.md`](2026-10-10-sp3-review/fix-wave-report.md).

| Область | Critical | Important | Minor | Вердикт |
|---|---|---|---|---|
| A — игроки, конфиг, исполнение | 0 | 1 (I-1: `record` / `bc` / `eval` проверяют источники тёплого старта, которые не используют, — конвейер не идёт с одного конфига) | 10 | с исправлениями |
| B — лига, рейтинги, метрики, распределённые ограничения | 0 | 2 (I1: удалённые снимки возвращаются в таблицу PFSP через поздние результаты; I2: LEAGUE_GUIDE преуменьшает изменения поведения против SP2) | 13 | с исправлениями |
| C — тёплый старт, путь данных, record/BC, тесты, документация, замеры | 0 | 3 (I-1 = A I-1; I-2: прогрев критика замораживал нормализатор `global_state` самого критика; I-3 = B I2) | 6 | с исправлениями |

Уникальных Important — 4 (A-I1 = C-I1, B-I2 = C-I3). Решения контроллера по ним — в списке выше (журнал, строки 184–188, 191, 193).

Единственная волна исправлений (`c31eda9..6fa1406`, 18 коммитов; TDD — сначала падающий тест), пункты брифа 1–18:
1. Конвейер с одного конфига (A-I1 = C-I1): `validate_config(config, *, scope="train" | "play", players=...)`; `record` / `bc` / `eval` проверяют в «play»-области (без тёплого старта и матчмейкинга, только используемые frozen-агенты) и принимают `--set`; тест `tests/integration/test_sp3_one_config.py`; README / LEAGUE_GUIDE §8 — форма с одним конфигом (`084a710`).
2. Прогрев критика обновляет нормализаторы пути value (C-I2): `APPO._update_normalizers(batch, value_path_only=warmup)`, `PolicyModel.update_normalizers(obs=None, ...)` (`0e878c4`).
3. Статистика PFSP удалённых снимков не возвращается (B-I1): `PfspStats._forgotten` (`c3b8d90`).
4. Изменения поведения против SP2 перечислены в LEAGUE_GUIDE §16 (B-I2 = C-I3): новая смесь по умолчанию, PFSP вместо равномерного выбора, снимки соперника в асимметричных играх, `self_play` с несколькими агентами, `league` по каждой команде, `keep_every` (`01ac5e2`).
5. `play_lineups` всегда ставит `collect=False` (`df3367d`).
6. Общие помощники `_reset_layout` / `_step_env` в `validate` (`860fd37`).
7. Формулировки T3.1, `allow_inf_nan=False` у `pfsp.exponent` / `halflife_games` / весов раскладок, подсказка `--set` для карты якорей и расписаний (`5728d34`).
8. Удаление снимков: `_notify_evicted` логирует все ошибки обратных вызовов; `Coordinator._evicted` не растёт без обновления составов (`653ad43`).
9. Журнал импорта пула считает только связанные снимки, нет строки «carried over: []» (`c675fe2`).
10. Удалены мёртвые псевдонимы `AgentOverride`, `ModelPool` (`a3993fa`).
11. `league/mixture.py`: удалён недостижимый запасной путь, `lineup_for` — `RuntimeError`, один помощник `league.base.build_matchmaker`, `CATEGORIES` выводится из `OPPONENT_CATEGORIES` (`286a6c5`).
12. Распределённое предупреждение называет игнорируемый `teammates: mixed` (`094d522`).
13. `_check_critic_warmup` оборачивает ошибки импорта в `ConfigError` (`860fd37`).
14. DAgger: `kickstart_label_frac` вне условия λ/прогрева, assert вместо запасного пути в `_teacher_label`, тест APPO с `Units` (K = 4) (`200088d`).
15. `colosseum bc --seed` (`084a710`).
16. Docstring `record.py` формулирует P8 (`6cddb66`).
17. Живые примеры: `tests/unit/test_example_configs.py::test_every_example_config_validates` уже гонял полный `validate` по всем примерам; усилен проверкой «Config is valid.» (`dc85637`).
18. CLAUDE.md: «Parked during SP3» без закрытых пунктов, счёт тестов 1689 / 10 / 22 (`51337cc`, `080128b`, `6fa1406`).

Итог волны: быстрый набор 1689 passed, 0 предупреждений; ruff чистый.

**Повторное ревью волны** (ограниченное, отдельного файла нет; вердикт — строка журнала 192, дословно): «Final fix wave: 18/18 items addressed, no new Critical/Important breakage (commits c31eda9..6fa1406); 3 docs Minors (LEAGUE_GUIDE §16 disk-growth formula uses total_timesteps instead of train steps; CLAUDE.md:299 --set command list lacks record/bc/eval; update_normalizers(obs=None) contract for custom models undocumented in guides) go to T6.7 as docs touch-ups.» Все три исправлены в `f2f54ec` (§3.8).

Остаются отложенными (Minor, не в волне): A M-2 (frozen-агенты строятся во всех воркерах), M-3 (`.pt`-настройки frozen-записи при `-a name=path`), M-10 (`bool` для целых компонентов); B M1, M2 (сыгранные доли при смешанных командах, атрибуция якоря), M12 (допуск в правиле ничьей team_tag), M13 (prior `PfspStats`), рекомендация о компактной таблице PFSP в `metrics.jsonl`; C minor 1 (`kl: reverse` при учителе-скрипте), 2 (resume без `trainer_state.pt` повторяет прогрев), 4 (запросы к учителю во время прогрева), 5 (сверка `record.json` с конфигом), 6 (пустые строки в `demo_learning.py`); прочие отложенные мелочи журнала — в приложении. Пункты «Открытые пункты» выше и «Parked during SP3» в CLAUDE.md совпадают.

---

## Приложение. Отложенные замечания по задачам (из журнала контроллера, дословно)

Каждая строка `minor (deferred)` журнала — дословно, по задачам, с отметкой, что закрыто волной исправлений или более поздней задачей, а что открыто. Ни одна не опущена (25 строк).

- **T0.1** (журнал, строка 10): Task T0.1: minor (deferred): resume-test comment "T2.1: the imported ckpt_v3 joins the pool" contradicts the assertion until T2.1 replaces it; eval "untouched" check compares only dir names (not file contents/mtimes); make_sp2_checkpoint.py rmtree replaces the committed reference when re-run (documented); copy_sp2_run fixed destination fails on a second call in one test
  - Статус после волны: Открыто всё, кроме: утверждение resume-теста заменено в T2.1 (перенос T0.1 → T2.1), комментарий вместе с ним.
- **T1.1** (журнал, строка 33): Task T1.1: minor (deferred): test_sp3_agent_kinds.py:137 `match="agents"` too weak to tell the two --set walk errors apart (plan-mandated); "Known agents" now config order (implicit agent_0 first) and "valid keys" list aliases — intentional wording changes, not regressions; CoreConfig.model_config redundant with StrictModel; StrictModel docstring names future aliases from/lambda
  - Статус после волны: Открыто.
- **T1.2** (журнал, строка 39): Task T1.2: minor (deferred): bool accepted for integer action components (cast to 1); load_fixed_players reads/checks a ckpt dir's meta twice; tests missing for an invalid .pt networks override, malformed meta.json via resolve_player_roles, assert_no_tensors(FixedPlayers); FrozenSpec.networks stored mode="json" (tuples→lists for code-built configs); FrozenSpec == raises (numpy) — consider eq=False
  - Статус после волны: Открыто (в «Parked during SP3»: `bool` для целых компонентов, `FrozenSpec ==`, недостающие негативные тесты).
- **T1.3** (журнал, строка 44): Task T1.3: minor (deferred): WrongShapeBot test regex unanchored/no context check (plan-mandated); no uint8-observation test for bots; ActRecord/bot info is the env's own infos object (document "by reference")
  - Статус после волны: Открыто («by reference» задокументировано в T1.3).
- **T1.4** (журнал, строка 51): Task T1.4: minor (deferred): setup_run resolves player roles twice (load_fixed_players calls resolve_player_roles again; meta.json read twice)
  - Статус после волны: Открыто.
- **T1.5** (журнал, строка 56): Task T1.5: minor (deferred): validate plays bots only in ENABLED layouts (spec says every layout with a seat; narrowing); untested: "no seat in the enabled layouts" line, bot reset raising in validate; evaluate restates _bad_path message and BotSpec construction; _resolve guard covers ScriptedPlayer only (a frozen PolicyModel served under latest but not a ckpt id would fall back to latest+collect — unreachable today; recheck in T3.2); _resolve ValueError fires at episode end (deferred fallback); eval loads requested frozen agents twice (validate + evaluate); ObsSpec rebuilt per bot decision in validate
  - Статус после волны: Открыто (в «Parked during SP3»).
- **T2.1** (журнал, строка 62): Task T2.1: minor (deferred): no test of the copy fallback / hard-link proof (samefile/st_nlink); "Imported N snapshots" counts skipped ids; coordinator logs "snapshot pool carried over: []" for agents absent from the old run; pool_size translation warning printed before the run dir exists (not in logs/main.log)
  - Статус после волны: Закрыто волной (п. 9): «Imported N snapshots» считает только связанные снимки, строки «carried over: []» для отсутствующих агентов нет. Открыто: тест копии / доказательства hardlink, предупреждение `pool_size` до создания папки запуска.
- **T2.2** (журнал, строка 66): Task T2.2: minor (deferred): _notify_evicted drops later callback errors without a log line; .pt resume-warning exclusion untested; worker_evictions=None path consumes evictions with no retry (SP2 4-arg test path only); Coordinator._evicted grows if match refresh is disabled; test 4 leaves spawn mp.Queues unclosed
  - Статус после волны: Закрыто волной (п. 8): `_notify_evicted` логирует каждую ошибку после первой; `Coordinator._evicted` не растёт без обновления составов. Открыто: исключение `.pt` в предупреждении resume без теста, путь `worker_evictions=None`, незакрытые очереди теста 4.
- **T2.3** (журнал, строка 70): Task T2.3: minor (deferred): hub._ratings_row placed between the *_over_layouts helpers (cosmetic); PfspStats does not validate prior in [0,1]; no test with a frozen/scripted TEAMMATE (opponent_type stays latest/arena)
  - Статус после волны: Открыто (в «Parked during SP3»: prior `PfspStats`, тест с frozen/scripted напарником).
- **T3.1** (журнал, строка 75): Task T3.1: minor (deferred): SP2 shim misreads non-normalized shares (1 - rivals@0; deleted in T3.2); --set cannot reach inside an anchor map/schedule ("not a section"; whole-value form works — add a hint); stale wording: get_agent_config docstring "deep-merged", its ConfigError lists only networks/algorithm/learner, agents field description lacks matchmaking; weak assertion for {"opponents":{"latest":-1}}; pfsp exponent accepts inf (allow_inf_nan=False)
  - Статус после волны: Закрыто волной (п. 7): подсказка `--set` для карты якорей / расписания, формулировки `get_agent_config` / поля `agents`, `allow_inf_nan=False` у `pfsp.exponent`. Прокладка SP2 удалена в T3.2. Открыто: слабое утверждение для `{"opponents":{"latest":-1}}`.
- **T3.2** (журнал, строка 83): Task T3.2: minor (deferred): mixture.CATEGORIES duplicates OPPONENT_CATEGORIES[:4]; lineup_for raises KeyError for "no playable layout" (RuntimeError/ConfigError fits better); coordinator test reads/sets private attributes (switch to the config path in T3.3); mixture.py:208-209 anchor fallback unreachable after validation
  - Статус после волны: Закрыто волной (п. 11): `CATEGORIES = OPPONENT_CATEGORIES[:4]`, `lineup_for` без играбельной раскладки — `RuntimeError` (для не-обучаемого владельца остался `KeyError`), недостижимый запасной путь удалён. Открыто: тест координатора читает приватные атрибуты.
- **T3.3** (журнал, строка 87): Task T3.3: minor (deferred): "construct user matchmaker class, wrap failure" duplicated in coordinator and validate with inconsistent ConfigError re-wrapping (one helper in league.base); missing-checkpoint fallback keeps source="snapshots" for a seat that played latest (document or tag fallback); validate exercises a custom matchmaker only at env step 0 without snapshots; mixture.py zero-weight anchor fallback branch still present though unreachable (remove or filter w>0)
  - Статус после волны: Закрыто волной (п. 11): один помощник `league.base.build_matchmaker`, недостижимая ветка удалена. Открыто: подмена отсутствующего снимка на latest сохраняет `source="snapshots"` (описано в LEAGUE_GUIDE); `validate` проверяет свой матчмейкер только на шаге 0 без снимков.
- **T4.1** (журнал, строка 96): Task T4.1: minor (deferred): test_sp2_validate.py:159 name no longer matches its content (add a real per-agent asymmetric .pt test later); teachers resolved twice (validate + launcher/distributed)
  - Статус после волны: Открыто.
- **T4.2** (журнал, строка 104): Task T4.2: minor (deferred): factory duplicates checkpoint discovery (_CKPT_DIR_RE vs checkpoint_manager._CKPT_RE, path classification vs classify_resume_source, signature message); init weights read twice in train (validate + launcher); validate does not check resumed weights/signatures per agent (pre-existing; a broken run-dir ckpt now also hides that agent's init check)
  - Статус после волны: Открыто (в «Parked during SP3»).
- **T4.3** (журнал, строка 109): Task T4.3: minor (deferred): _check_critic_warmup lets import_class errors of a mistyped algorithm_class escape as raw exceptions (wrap as ConfigError); signature check rejects **kwargs-forwarding APPO subclasses; tests missing: frozen params' .grad is None, recurrent core under warm-up, LR test does not train; shared value/policy params not detected (documented contract)
  - Статус после волны: Закрыто волной (п. 13): ошибки `import_class` — `ConfigError`. Открыто: подклассы APPO с `**kwargs`, недостающие тесты, общие параметры value/политики (документированный контракт).
- **T4.4** (журнал, строка 115): Task T4.4: minor (deferred): test "recurrent teacher ... gets the chunk state" asserts only loss > 0 (spy could check state0); _check_kickstart_teachers rebuilds the student model; loose "KL" match in a validate test
  - Статус после волны: Открыто.
- **T4.5** (журнал, строка 120): Task T4.5: minor (deferred → final fix wave candidates): APPO-level DAgger test with K>1 (Units) missing (compute_labels tested directly only); _teacher_label's dead fallback `_episode_layout[env] or lineup(env).layout` hides a broken hook; kickstart_label_frac reads 0 during critic warm-up (compute outside the gate or document); duplicated episode bookkeeping in RolloutLoop vs EpisodeTracker; validate_chunk_payload checks presence/shape only for teacher fields (SP5 peer-trust class); teacher bot handling parallels MatchRunner's (shared helper)
  - Статус после волны: Закрыто волной (п. 14): тест DAgger на уровне APPO с `Units` (K = 4), запасной путь `_teacher_label` заменён на assert, `kickstart_label_frac` считается и во время прогрева. Открыто: дублирование учёта эпизодов в `RolloutLoop`, `validate_chunk_payload` (класс доверия к пиру, SP5), параллель обработки бота-учителя с `MatchRunner`.
- **T5.1** (журнал, строка 126): Task T5.1: minor (deferred): record._parse_player duplicates cli._parse_agent_spec with a different exit code; load_player computes the path predicate twice; partial part files remain after a mid-run failure (rerun says "exists and is not empty", same message for an existing file); P8 test calls private _schedule; no neural Units recording test
  - Статус после волны: Открыто (`_parse_player` стал публичным `record.parse_player`, дублирование осталось).
- **T5.2** (журнал, строка 131): Task T5.2: minor (deferred): "competitive layout" rounding predicate duplicated in cli eval and record (stderr note vs INFO log); no test for --against with an explicit one-team --layout (no rounding); redundant bc_data_sources import inside the bc command
  - Статус после волны: Открыто.
- **T6.1** (журнал, строка 136): Task T6.1: minor (deferred): the "reduced to latest only" warning is emitted before file logging is set up (console only, not in the run's logs); spurious warning when only empty categories carry non-latest shares (configured, not effective shares); global teammates: mixed still silently ignored in distributed mode (SP5 parked); no test for several problems in one ConfigError
  - Статус после волны: Закрыто волной (п. 12): предупреждение называет игнорируемый `teammates: mixed`. Открыто: предупреждение до файлового логирования, ложное предупреждение при пустых категориях, тест на несколько проблем в одной `ConfigError`.
- **T6.2** (журнал, строка 146): Task T6.2: minor (deferred): bc has no seed (BC init varies between runs — T6.4 thresholds must allow for it); comments "T6.4 calibration" in unit_harvest_league.yaml and pipeline_kit must be updated by T6.4; PipelineSettings.init_from magic string; 3 blank lines left in demo_learning.py/pipeline_kit.py
  - Статус после волны: Закрыто: `bc --seed` (волна, п. 15); комментарии «T6.4 calibration» обновлены в T6.4; три пустые строки в `pipeline_kit.py` убраны. Открыто: магическая строка `PipelineSettings.init_from`, три пустые строки в `tests/learning/demo_learning.py` (строка ≈ 271).
- **T6.3** (журнал, строка 154): Task T6.3: minor (deferred): tie comparison without float tolerance (scripts/team_tag_anchors.py:74); W/D/L logic in three places (wdl, slow test, demo_learning.win_rate); the slow test's own printed win rate not captured (replay gave 0.920)
  - Статус после волны: Открыто (в «Parked during SP3»).
- **T6.4** (журнал, строка 162): Task T6.4: minor (deferred → final fix wave candidate): `colosseum bc` has no --seed (seeded pipeline runs are not reproducible); docs/benchmarks.md:247 attributes the extra training time to DAgger teacher calls without measuring it (say «вероятно»)
  - Статус после волны: Закрыто: `bc --seed` (волна, п. 15); в `docs/benchmarks.md` стоит «вероятно».
- **T6.5** (журнал, строка 168): Task T6.5: minor (deferred): compare decides with n=2 if a seed fails (driver exits 1); `choice` shows the default next to a "no decision" note; --cells parse errors are raw tracebacks after mkdir and --ratio-modes/--unit-traces silently ignored with --cells; float edge at K=8 −0.03 boundary; benchmarks timing line "≈ 6 min eval" inconsistent; joint:auto also switches entropy_reduction to sum (mention in benchmarks)
  - Статус после волны: Открыто (в «Parked during SP3»: n = 2, сырые трассировки `--cells`).
- **T6.6** (журнал, строка 178): Task T6.6: minor (deferred): record.py module docstring less precise than LEAGUE_GUIDE about skipped layouts; LEAGUE_GUIDE example's trivial __init__
  - Статус после волны: Закрыто волной (п. 16): docstring `record.py` формулирует P8. Открыто: тривиальный `__init__` в примере LEAGUE_GUIDE.

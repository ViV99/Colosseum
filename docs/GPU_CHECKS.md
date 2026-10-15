# Проверки на GPU

На машине разработки нет GPU, поэтому всё, что требует CUDA, помечено маркером `gpu`. Без CUDA такие тесты пропускаются (`tests/conftest.py`). Обычный прогон `pytest -m "not gpu and not slow"` их не запускает.

## Как запустить

На машине с NVIDIA GPU и драйвером CUDA:

```bash
scripts/setup-dev.sh --gpu          # ставит сборку torch с CUDA вместо CPU-сборки
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # должно напечатать True
.venv/bin/python -m pytest -m gpu -v
```

Все тесты из таблицы должны пройти (`passed`, не `skipped`). Если тест `skipped`, CUDA не видна: проверьте вывод второй команды.

Один тест (вместе со всеми его параметризациями) запускается командой из колонки «Команда». Без `-m gpu` эти команды тоже работают: маркер только исключает тесты из обычного прогона.

Машина с ROS. Если в окружении задан `PYTHONPATH` от ROS (например, `~/.bashrc` подключает ROS iron: `PYTHONPATH=/opt/ros/iron/lib/python3.10/site-packages`), pytest падает при старте: точка входа `pytest11` пакета ROS `launch_testing` → `ModuleNotFoundError: No module named 'lark'`. ROS не подменяет ни одной зависимости проекта, ломается только автозагрузка плагинов pytest. Обход — префикс `env -u PYTHONPATH` у каждой команды `.venv/bin/python` / `.venv/bin/colosseum` / `.venv/bin/ruff`, например `env -u PYTHONPATH .venv/bin/python -m pytest -m gpu -v`.

## Результаты 2026-10-15 (RTX 4070 Ti)

Первый прогон на машине с CUDA: WSL2, 20 ядер, RTX 4070 Ti (12 ГБ, sm_89), torch 2.14.1+cu130, cuDNN 9.24.0. Подробности, команды и все числа — в отчёте [`superpowers/reports/2026-10-15-gpu-checks.md`](superpowers/reports/2026-10-15-gpu-checks.md).

| Проверка | Результат |
|---|---|
| `gpu`-тесты | 22 passed после исправления теста `test_state.py` (до него 2 падения `[lstm]`, `[gru]` из-за TF32 в cuDNN RNN, см. «TF32» ниже); с тестами головы в fp32 — 25 passed |
| Быстрый набор | 1689 passed за 311.65 с (с тестом головы в fp32 — 1690 passed за 297.74 с), без предупреждений; интеграционные тесты на `configs/examples/` учат лёрнер на CUDA |
| Медленный набор, лёрнеры на CUDA | 10 passed за 423.96 с; все пороги пройдены, значения близки к CPU при приёмке SP3 |
| Пайплайн README под AMP float16 и bfloat16 | все команды exit 0, нет NaN/inf, прогрев только 30 шагов, λ убывает; trained против random 1.000, против greedy 0.500 (то же после перевода головы политики в fp32) |
| M-6 (log-ratio при нулевом лаге под AMP) | исправлено в коде: голова политики под autocast считается в fp32 (G-R6); было bf16 до 0.19 p95 и 4.4 % клипа при σ = e^−4, стало до 0.10 и 0.2 %; fp16 до 1.29e-2, клипа нет (см. «AMP и гауссовы действия» ниже) |
| Resume AMP-прогона, CUDA-чекпоинты без CUDA, распределённый лёрнер на CUDA, WandB | всё работает (exit 0; распределённый лёрнер — 143 по SIGTERM) |

## Что проверяется

- AMP: шаг обучения APPO в `float16` (с GradScaler) и `bfloat16`, без NaN/inf в метриках и весах, веса остаются float32;
- состояние GradScaler и всего алгоритма (`state_dict`) сохраняется и восстанавливается, когда модель на CUDA;
- `learner.pin_memory`: батч закрепляется в памяти и переносится на устройство с `non_blocking=True`, результат совпадает с обычным переносом;
- деревья наблюдений, действий и масок на CUDA: батч лёрнера из чанков v2 (`uint8`-листья доходят до модели без приведения), `act`/`boot`/`pad`-слоты;
- распределения `Units` (`UnitsDist` внутри `TreeDist`) на CUDA и под AMP: шаг APPO с отсутствующими юнитами (`unit=False`) даёт конечные лосс и метрики, per-unit log-prob лёрнера считаются на CUDA. Отдельной проверки на CUDA, что `log_prob` = сумма валидных `unit_log_prob` и что пустые строки масок не дают NaN, нет: это проверяют только CPU-тесты;
- режимы `ratio_mode` / `unit_trace` / `entropy_reduction` на CUDA — только значения по умолчанию для `Units` (`auto`: `per_unit`, `joint`, `mean_valid`); остальные режимы проверяются на CPU;
- централизованный критик (`critic_encoder` и `global_state`) на CUDA, в том числе под AMP;
- перенос модели и состояния `State` (LSTM, GRU, окно внимания) между CPU и CUDA (`state_to`), одинаковые выходы на обоих устройствах;
- kickstart на CUDA: учитель и студент на одном устройстве, маски на устройстве;
- BC на CUDA с деревьями наблюдений (`uint8`), включая обучение по последовательностям для модели с состоянием.
- голова политики `ComposedModel` под CUDA autocast считается в float32 (G-R6), значение и энкодер — под autocast; регрессия M-6 при нулевом лаге (bf16, LSTM, σ = e^−4).
- тёплый старт SP3 под AMP: прогрев критика (`critic_warmup_steps`) — у параметров пути политики нет градиентов, `GradScaler.unscale_` / `step` должны их пропускать, политика остаётся побитово прежней, обновляется только value-путь (с `critic_encoder`); DAgger-лосс скриптового учителя (`λ · mean(−log π(a_учителя))`) под autocast конечен и обучает политику. CPU-варианты этих тестов (без AMP) идут в обычном прогоне.

### TF32

По умолчанию torch на GPU Ampere и новее включает TF32 для cuDNN conv и cuDNN RNN (`torch.backends.cudnn.allow_tf32 == True`), а матричные умножения cuBLAS оставляет в IEEE fp32 (`torch.backends.cuda.matmul.allow_tf32 == False`). Ядра LSTM/GRU идут через cuDNN RNN, поэтому их выходы на CUDA отличаются от CPU fp32 примерно на 1e-4 (у TF32 мантисса 10 бит); ядра `none` и `attention` (обычные matmul) проходят строгое сравнение и с TF32 по умолчанию. С выключенным TF32 разница состояния LSTM / GRU — 2.53e-06 / 1.88e-06. Тест `test_model_and_state_move_between_devices` поэтому сравнивает строго (rtol 1e-4, atol 1e-5) с выключенным TF32 (`torch.backends.cudnn.flags(enabled=True, allow_tf32=False)`) и отдельно закрепляет поведение по умолчанию с допуском TF32 (rtol 1e-2, atol 1e-3). Решение G-R1: фреймворк оставляет умолчания torch (своей настройки точности у лёрнера нет) — TF32 ускоряет cuDNN RNN/conv, а его влияние на log-prob лёрнера намного меньше диапазона клипа PPO (при нулевом лаге per-decider |log-ratio| p95 1.4e-5–3e-5 с ядром LSTM, до 1.7e-3 для `Box` с `log_std` −4; `eps_clip` 0.2).

### AMP и гауссовы действия (M-6)

Финальное ревью SP2 (область A, M-6): под AMP голова политики выдавала параметры распределения (например, `mean` гауссова действия) в fp16/bf16 (до G-R6), и `gaussian_log_prob` / `gaussian_entropy` (в том числе для `Box` внутри `Units`) считались в той же точности, потому что записанные float32-действия приводятся к `mean.dtype`. При нулевом лаге это давало численный шум в log-ratio лёрнера, который растёт при уменьшении σ.

Исправление встроено (решение G-R6, ответ владельца на Q1): `ComposedModel` вызывает голову политики через `_policy_head`, и пока на CUDA включён autocast, голова работает с выключенным autocast на float32-признаках (вещественные листья `aux` тоже приводятся к float32, целые не трогаются). Энкодер, ядро и голова value по-прежнему работают под autocast; без autocast (воркеры на CPU, BC, лёрнер без AMP) голова получает те же самые тензоры, ничего не меняется. Свой `PolicyModel` (не `ComposedModel`) стоит устроить так же: голову — под `torch.autocast(device_type="cuda", enabled=False)` на float32-входах (см. `ComposedModel._policy_head`).

Замер — скрипт (по умолчанию меряет фреймворк как есть; `--amp-head` возвращает прежнее поведение, когда голова тоже работала под autocast):

```bash
env -u PYTHONPATH .venv/bin/python scripts/measure_amp_logratio.py [--amp-head] [--fp32-gauss] [--json out.json]
```

Он сравнивает per-decider log-ratio лёрнера на CUDA (fp32 / AMP fp16 / AMP bf16) с log-prob поведения от CPU fp32 модели при нулевом лаге. Результат 2026-10-15 (RTX 4070 Ti), per-decider |log-ratio| p95, `Units(6, Box(4))` (полные таблицы со всеми ролями — в отчёте):

| ядро | log_std | fp32 | fp16 | bf16 | bf16 clip_fraction | до исправления (`--amp-head`): fp16 / bf16 |
|---|---|---|---|---|---|---|
| lstm | 0 | 3.05e-05 | 6.39e-05 | 3.04e-04 | 0 | 2.12e-03 / 1.76e-02 |
| lstm | −4 | 1.68e-03 | 3.51e-03 | 1.66e-02 | 0 | 1.96e-02 / 1.37e-01 |
| none | 0 | 4.77e-07 | 2.36e-04 | 1.84e-03 | 0 | 2.17e-03 / 1.78e-02 |
| none | −4 | 9.54e-06 | 1.29e-02 | 1.00e-01 | 0.0022 | 2.51e-02 / 1.94e-01 |

Для `Units(6, Discrete(5))` теперь fp16 p95 не больше 1.05e-4, bf16 не больше 8.31e-4; `clip_fraction` у fp16 везде 0, у bf16 до исправления доходил до 0.0441. Колонка fp32 у LSTM — эффект TF32 (выше). Повышение точности внутри самого `gaussian_log_prob` (`--fp32-gauss`) почти ничего не меняет (2.51e-02 → 2.48e-02 без исправления головы).

Откуда шум (разложение контроллера, `Units(6, Box(4))`, `log_std` −4): если в fp32 считать и голову, и энкодер, шум падает до пола fp32 (9.54e-06 без ядра), то есть источников ровно два — собственный выход головы в низкой точности и признаки, которые энкодер выдаёт под autocast. С ядром LSTM главный источник — голова (bf16: 0.137 → 0.0166, в 8 раз), без ядра — примерно половина (0.194 → 0.100); остаток с LSTM (2.99e-03 при голове и энкодере в fp32) — cuDNN (TF32 и LSTM в float16 под autocast).

Остаток после исправления — округление признаков энкодера и ядра: это и есть AMP (их matmul в низкой точности), поэтому он остаётся; bf16 без ядра при малом σ всё ещё даёт p95 ≈ 0.10. bf16 в 5–8× шумнее fp16 (до исправления — в 7–8×); при малом σ шум растёт почти как 1/σ (множитель (x − μ)/σ²).

Скорость: голова в fp32 стоит около +1–3 % шага AMP-лёрнера (+7.8 % в одном шумном случае). Но на RTX 4070 Ti AMP вообще не быстрее обычного fp32 ни для одной из проверенных моделей: на моделях примеров на 1–13 % медленнее с головой под autocast и на 3–16 % — с головой в fp32; на LSTM-512 примерно на 50 % (fp32 уже использует TF32 в cuDNN, AMP добавляет приведения типов и GradScaler, развёртка LSTM упирается в запуск ядер; таблица — в `docs/benchmarks.md`).

Практически: оставляйте `algorithm.use_amp: false` (по умолчанию), если AMP заметно не ускоряет вашу модель. С AMP, `Box`-действиями и малым σ предпочитайте fp16, а не bf16; следите за `log_rho_abs_p95` в начале обучения — при нулевом лаге он показывает численный пол. `rho_clip_frac` на GPU при нулевом лаге читается 0.13–0.54 и в fp32 тоже: метрика считает любой ρ > ρ̄ = 1, то есть и численный шум.

Замечание о памяти: `train_step` переносит на устройство весь батч шага сразу, а не по минибатчу. Пиковая память GPU под данные батча равна полному батчу шага (`batch_chunks × chunk_length` слотов, включая `global_state`, если он есть); обычно это мало по сравнению с активациями, но при больших наблюдениях стоит смотреть на неё.

## Список тестов

Каждая строка — один node id из `.venv/bin/python -m pytest -m gpu --collect-only -q`. Для каждой строки ожидается `passed`.

Известная нестабильность: в тесте сохранения GradScaler (`test_grad_scaler_state_round_trips_on_cuda`, AMP float16) проверка `_growth_tracker > 0` может
не пройти, если последний шаг fp16 был пропущен GradScaler'ом (переполнение, бывает редко). Тогда тест надо просто
перезапустить; это не ошибка кода. Повторное падение — уже повод разбираться.

| Тест | Что проверяет | Команда |
|---|---|---|
| `tests/unit/test_appo_v2.py::test_amp_train_step_on_cuda_with_dict_uint8_obs_and_global_state[bfloat16]` | AMP `bfloat16`, ядро GRU, Dict-наблюдение с `uint8`-листом, `Units(3)` и `uint8`-`global_state` (критик): листья батча на CUDA остаются `uint8`, две эпохи по минибатчам, метрики конечны, веса float32 без NaN/inf | `.venv/bin/python -m pytest -v tests/unit/test_appo_v2.py -k test_amp_train_step_on_cuda_with_dict_uint8_obs_and_global_state` |
| `tests/unit/test_appo_v2.py::test_amp_train_step_on_cuda_with_dict_uint8_obs_and_global_state[float16]` | то же с AMP `float16` (GradScaler): листья батча на CUDA остаются `uint8`, метрики конечны (или шаг пропущен GradScaler'ом), веса float32 без NaN/inf | `.venv/bin/python -m pytest -v tests/unit/test_appo_v2.py -k test_amp_train_step_on_cuda_with_dict_uint8_obs_and_global_state` |
| `tests/unit/test_appo_v2.py::test_amp_train_step_on_cuda_with_units[lstm-bfloat16]` | AMP `bfloat16`, ядро LSTM, действие `Units(6)` с отсутствующими юнитами, `pin_memory`: шаг APPO на CUDA даёт конечный лосс, per-unit log-prob лёрнера на CUDA с 6 решающими | `.venv/bin/python -m pytest -v tests/unit/test_appo_v2.py -k test_amp_train_step_on_cuda_with_units` |
| `tests/unit/test_appo_v2.py::test_amp_train_step_on_cuda_with_units[lstm-float16]` | то же с AMP `float16` и ядром LSTM | `.venv/bin/python -m pytest -v tests/unit/test_appo_v2.py -k test_amp_train_step_on_cuda_with_units` |
| `tests/unit/test_appo_v2.py::test_amp_train_step_on_cuda_with_units[none-bfloat16]` | то же с AMP `bfloat16` без ядра | `.venv/bin/python -m pytest -v tests/unit/test_appo_v2.py -k test_amp_train_step_on_cuda_with_units` |
| `tests/unit/test_appo_v2.py::test_amp_train_step_on_cuda_with_units[none-float16]` | то же с AMP `float16` без ядра | `.venv/bin/python -m pytest -v tests/unit/test_appo_v2.py -k test_amp_train_step_on_cuda_with_units` |
| `tests/unit/test_appo_v2.py::test_grad_scaler_state_round_trips_on_cuda` | AMP `float16` на CUDA (`Units`, две эпохи по минибатчам): состояние GradScaler после трёх шагов сохраняется в `state_dict` и восстанавливается в новом APPO | `.venv/bin/python -m pytest -v tests/unit/test_appo_v2.py -k test_grad_scaler_state_round_trips_on_cuda` |
| `tests/unit/test_appo_v2.py::test_pin_memory_batch_preparation_on_cuda[lstm]` | `pin_memory=True` (ядро LSTM, `Units`): батч чанков v2 на CUDA (все слоты `act`/`boot`/`pad`) совпадает с батчем без закрепления памяти по устройству, dtype и значениям; шаг обучения даёт конечные метрики | `.venv/bin/python -m pytest -v tests/unit/test_appo_v2.py -k test_pin_memory_batch_preparation_on_cuda` |
| `tests/unit/test_appo_v2.py::test_pin_memory_batch_preparation_on_cuda[none]` | то же без ядра | `.venv/bin/python -m pytest -v tests/unit/test_appo_v2.py -k test_pin_memory_batch_preparation_on_cuda` |
| `tests/unit/test_appo_v2.py::test_state_dict_round_trip_with_model_on_cuda` | `APPO.state_dict()` с LSTM-моделью и `Units` на CUDA: сохранённые тензоры на CPU, после загрузки состояние оптимизатора снова на CUDA, LR/версия/`consumed_samples` совпадают, следующий шаг даёт версию 2 и те же веса | `.venv/bin/python -m pytest -v tests/unit/test_appo_v2.py -k test_state_dict_round_trip_with_model_on_cuda` |
| `tests/unit/test_fp32_policy_head.py::test_bf16_zero_lag_log_ratio_with_a_small_sigma_stays_small` | регрессия M-6: `Units(6, Box(4))`, ядро LSTM 256, все `log_std` = −4, AMP `bfloat16`, нулевой лаг: p95 per-decider \|log-ratio\| лёрнера на CUDA < 0.05 (с головой под autocast было 0.137, с головой в fp32 — 0.0166) | `.venv/bin/python -m pytest -v tests/unit/test_fp32_policy_head.py -k test_bf16_zero_lag_log_ratio_with_a_small_sigma_stays_small` |
| `tests/unit/test_fp32_policy_head.py::test_under_cuda_autocast_the_policy_head_runs_in_float32[bfloat16]` | под `torch.autocast("cuda", bfloat16)` в `step` и `unroll` голова политики получает float32-признаки и float32-листья `aux` (целый лист сохраняет dtype), внутри головы autocast выключен, после неё снова включён (выход головы value в bfloat16), параметры распределения float32 | `.venv/bin/python -m pytest -v tests/unit/test_fp32_policy_head.py -k test_under_cuda_autocast_the_policy_head_runs_in_float32` |
| `tests/unit/test_fp32_policy_head.py::test_under_cuda_autocast_the_policy_head_runs_in_float32[float16]` | то же с `float16` | `.venv/bin/python -m pytest -v tests/unit/test_fp32_policy_head.py -k test_under_cuda_autocast_the_policy_head_runs_in_float32` |
| `tests/unit/test_kickstart_v2.py::test_kickstart_on_cuda_with_masks_and_an_lstm_core[False]` | kickstart (forward KL) на CUDA с LSTM, `Units` и масками, AMP выключен: учитель и студент на CUDA, `kickstart_loss` > 0, метрики конечны, λ убывает | `.venv/bin/python -m pytest -v tests/unit/test_kickstart_v2.py -k test_kickstart_on_cuda_with_masks_and_an_lstm_core` |
| `tests/unit/test_kickstart_v2.py::test_kickstart_on_cuda_with_masks_and_an_lstm_core[True]` | то же с AMP | `.venv/bin/python -m pytest -v tests/unit/test_kickstart_v2.py -k test_kickstart_on_cuda_with_masks_and_an_lstm_core` |
| `tests/unit/test_sp2_bc_trainer.py::test_bc_with_tree_data_on_cuda[lstm]` | BC на CUDA (ядро LSTM, окна по 8 шагов со сбросом на `dones`): Dict-наблюдение с `uint8`-листом, маски; параметры на CUDA, энкодер получает `uint8` и `float32`, `bc_loss` конечен, accuracy в [0, 1] | `.venv/bin/python -m pytest -v tests/unit/test_sp2_bc_trainer.py -k test_bc_with_tree_data_on_cuda` |
| `tests/unit/test_sp2_bc_trainer.py::test_bc_with_tree_data_on_cuda[none]` | то же без ядра | `.venv/bin/python -m pytest -v tests/unit/test_sp2_bc_trainer.py -k test_bc_with_tree_data_on_cuda` |
| `tests/unit/test_sp3_gpu_warmstart.py::test_critic_warmup_under_amp_updates_only_the_value_path[cuda-float16]` | прогрев критика 3 шага под AMP `float16` (GradScaler) с `critic_encoder`: веса пути политики побитово прежние, у них нет `.grad`, состояние Adam только у value-пути, value-путь изменился (если GradScaler не пропустил все шаги), лоссы конечны | `.venv/bin/python -m pytest -v tests/unit/test_sp3_gpu_warmstart.py -k test_critic_warmup_under_amp_updates_only_the_value_path` |
| `tests/unit/test_sp3_gpu_warmstart.py::test_critic_warmup_under_amp_updates_only_the_value_path[cuda-bfloat16]` | то же с AMP `bfloat16` | `.venv/bin/python -m pytest -v tests/unit/test_sp3_gpu_warmstart.py -k test_critic_warmup_under_amp_updates_only_the_value_path` |
| `tests/unit/test_sp3_gpu_warmstart.py::test_dagger_label_loss_under_amp_is_finite_and_trains[cuda-float16]` | DAgger-лосс (метки учителя на всех ACT-слотах) под AMP `float16`: `kickstart_loss` > 0 и конечен, `kickstart_label_frac` = 1, остальные метрики конечны, веса политики изменились (если шаг не пропущен), λ убывает | `.venv/bin/python -m pytest -v tests/unit/test_sp3_gpu_warmstart.py -k test_dagger_label_loss_under_amp_is_finite_and_trains` |
| `tests/unit/test_sp3_gpu_warmstart.py::test_dagger_label_loss_under_amp_is_finite_and_trains[cuda-bfloat16]` | то же с AMP `bfloat16` | `.venv/bin/python -m pytest -v tests/unit/test_sp3_gpu_warmstart.py -k test_dagger_label_loss_under_amp_is_finite_and_trains` |
| `tests/unit/test_state.py::test_model_and_state_move_between_devices[attention]` | модель и `State` (ядро окна внимания): CPU → CUDA → CPU, все тензоры состояния на нужном устройстве, log-prob и состояние на CUDA с выключенным TF32 cuDNN совпадают с CPU (rtol 1e-4, atol 1e-5), с TF32 по умолчанию — в допуске TF32 (rtol 1e-2, atol 1e-3) | `.venv/bin/python -m pytest -v tests/unit/test_state.py -k test_model_and_state_move_between_devices` |
| `tests/unit/test_state.py::test_model_and_state_move_between_devices[gru]` | то же для ядра GRU | `.venv/bin/python -m pytest -v tests/unit/test_state.py -k test_model_and_state_move_between_devices` |
| `tests/unit/test_state.py::test_model_and_state_move_between_devices[lstm]` | то же для ядра LSTM | `.venv/bin/python -m pytest -v tests/unit/test_state.py -k test_model_and_state_move_between_devices` |
| `tests/unit/test_state.py::test_model_and_state_move_between_devices[none]` | то же без ядра | `.venv/bin/python -m pytest -v tests/unit/test_state.py -k test_model_and_state_move_between_devices` |

## Сквозная проверка на CUDA

Тесты из таблицы проверяют части по отдельности; целиком обучение на CUDA впервые прогнано 2026-10-15 по этой процедуре (все пункты пройдены, [отчёт](superpowers/reports/2026-10-15-gpu-checks.md)). Во всех конфигах `configs/examples/` стоит `learner.device: auto`, и `colosseum bc` берёт устройство из того же поля, поэтому на машине с GPU лёрнеры и BC сами работают на CUDA (воркеры по-прежнему считают политику на CPU). Устройство лёрнера видно в его логе `runs/<run>/logs/learner-<agent>.log`: строка `starting on device=cuda`. AMP работает только на CUDA: на CPU `algorithm.use_amp` молча не действует.

1. Быстрый набор `.venv/bin/python -m pytest -m "not gpu and not slow" -q -rw` должен пройти так же, как на машине разработки, без предупреждений. Интеграционные тесты, которые запускают `colosseum train` на конфигах из `configs/examples/`, учат лёрнер на CUDA.
2. Медленный набор `.venv/bin/python -m pytest -m slow -v` (10 тестов, около 15 минут на 8 ядрах CPU): обучение семи демо-игр и пайплайн `unit_harvest` (record → bc → train → eval) с лёрнером и BC на CUDA. Пороги откалиброваны на CPU, а на CUDA вычисления не побитово те же: единичное падение по порогу — повод перезапустить тест, повторное — разбираться.
3. Пайплайн из README с AMP (в остальных сквозных прогонах AMP выключен):

```bash
CFG=configs/examples/unit_harvest_league.yaml
.venv/bin/colosseum record -c $CFG --player greedy --num-matches 300 --output gpu-check/data --seed 0
.venv/bin/colosseum bc -c $CFG --agent main --data gpu-check/data --output gpu-check/bc.pt --epochs 5 --seed 0
.venv/bin/colosseum train -c $CFG --set run.name=gpu-check-amp --set algorithm.use_amp=true \
  --set agents.main.init.from=gpu-check/bc.pt --set agents.main.init.critic_warmup_steps=30 \
  --set agents.main.kickstart.teacher=greedy
NEW=$(ls -d runs/gpu-check-amp/checkpoints/main/ckpt_v* | sort -V | tail -1)
.venv/bin/colosseum eval -c $CFG -a trained=$NEW -a greedy -a random --num-matches 100 --deterministic
```

Для `bfloat16` повторите `train` и `eval` с `--set algorithm.amp_dtype=bfloat16` и другим `run.name`.

Ожидается: все команды завершаются с кодом 0; в логе лёрнера `device=cuda`; в записях `train` файла `runs/gpu-check-amp/metrics.jsonl` нет `NaN`/`inf`, `skipped_updates` почти всегда 0 (GradScaler изредка пропускает шаг при переполнении — это нормально), `critic_warmup` равен 1 только на первых 30 шагах лёрнера, `kickstart_lambda` после них убывает; в `eval` агент `trained` выигрывает у `random` не меньше 95 % игр и набирает против `greedy` не меньше 0,40 очка (пороги медленного теста пайплайна; на CPU при приёмке SP3 было 1,00 и 0,50).

После проверки: `rm -rf gpu-check runs/gpu-check-amp`.

## Ручная проверка WandB (без GPU)

Автотесты WandB используют подмену модуля `wandb`, поэтому настоящий WandB проверяется вручную, один раз, на любой машине.

```bash
uv pip install --python .venv/bin/python -e ".[wandb]"
WANDB_MODE=offline .venv/bin/colosseum train -c configs/examples/tic_tac_toe_multi.yaml \
  --set run.name=wandb-check --set metrics.use_wandb=true --set training.total_timesteps=60000
```

`WANDB_MODE=offline` пишет run в `./wandb/` без входа в аккаунт. Для отправки в облако: `wandb sync wandb/offline-run-*` (нужен `WANDB_API_KEY` или `wandb login`). Можно сразу запускать без `WANDB_MODE=offline`, если задан `WANDB_API_KEY`.

На что смотреть (в веб-интерфейсе после `wandb sync`, либо в файле `wandb/offline-run-*/run-*.wandb` и в определениях метрик run; wandb 0.30 в offline-режиме не пишет `files/wandb-summary.json`):

- один run на всю тренировку, его имя — имя папки запуска (`wandb-check`);
- метрики обучения каждого агента названы `<agent>/<metric>` (`agent_alpha/total_loss`, `agent_beta/lr`, ...) и строятся по своей оси `<agent>/train_step`; графики обоих агентов есть и не обрезаны, хотя агенты учатся с разной скоростью;
- `ratings/*` (вложенные ключи по вариантам партии, вида `ratings/<layout>/elo/<agent>`, `ratings/<layout>/wr_vs_past/<agent>`), `system/*` (`system/env_steps_per_sec`, `system/queue_depths/<agent>`, ...) и `episodes/*` (`episodes/<agent>/return_mean`, ...) строятся по оси `env_steps`;
- в консоли нет предупреждений `WandB ... disabled`, код выхода 0;
- те же значения есть в `runs/wandb-check/metrics.jsonl` (это источник истины; WandB только показывает их).

В `tic_tac_toe_multi.yaml` (соперники `rivals: 1.0`) агент никогда не играет со своими снимками, поэтому `ratings/<layout>/wr_vs_past/<agent>` там не появляется (значения None не пишутся). Чтобы увидеть этот ключ, запустите `WANDB_MODE=offline .venv/bin/colosseum train -c configs/examples/tic_tac_toe.yaml --set run.name=wandb-check-past --set metrics.use_wandb=true --set rollout.match_refresh_interval_sec=5 --set training.total_timesteps=400000` (проверено 2026-10-15: `ratings/2p/wr_vs_past/agent_0` = 0.718 по оси `env_steps`, как в `metrics.jsonl`).

После проверки: `rm -rf runs/wandb-check runs/wandb-check-past wandb`.

Распределённые роли (`run-learner`, `run-workers`) до SP5 не пишут ни в WandB, ни в `metrics.jsonl`: у них есть только логи процессов.

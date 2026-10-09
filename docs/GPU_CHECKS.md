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
| `tests/unit/test_kickstart_v2.py::test_kickstart_on_cuda_with_masks_and_an_lstm_core[False]` | kickstart (forward KL) на CUDA с LSTM, `Units` и масками, AMP выключен: учитель и студент на CUDA, `kickstart_loss` > 0, метрики конечны, λ убывает | `.venv/bin/python -m pytest -v tests/unit/test_kickstart_v2.py -k test_kickstart_on_cuda_with_masks_and_an_lstm_core` |
| `tests/unit/test_kickstart_v2.py::test_kickstart_on_cuda_with_masks_and_an_lstm_core[True]` | то же с AMP | `.venv/bin/python -m pytest -v tests/unit/test_kickstart_v2.py -k test_kickstart_on_cuda_with_masks_and_an_lstm_core` |
| `tests/unit/test_sp2_bc_trainer.py::test_bc_with_tree_data_on_cuda[lstm]` | BC на CUDA (ядро LSTM, окна по 8 шагов со сбросом на `dones`): Dict-наблюдение с `uint8`-листом, маски; параметры на CUDA, энкодер получает `uint8` и `float32`, `bc_loss` конечен, accuracy в [0, 1] | `.venv/bin/python -m pytest -v tests/unit/test_sp2_bc_trainer.py -k test_bc_with_tree_data_on_cuda` |
| `tests/unit/test_sp2_bc_trainer.py::test_bc_with_tree_data_on_cuda[none]` | то же без ядра | `.venv/bin/python -m pytest -v tests/unit/test_sp2_bc_trainer.py -k test_bc_with_tree_data_on_cuda` |
| `tests/unit/test_state.py::test_model_and_state_move_between_devices[attention]` | модель и `State` (ядро окна внимания): CPU → CUDA → CPU, все тензоры состояния на нужном устройстве, log-prob на CUDA совпадают с CPU (rtol 1e-4) | `.venv/bin/python -m pytest -v tests/unit/test_state.py -k test_model_and_state_move_between_devices` |
| `tests/unit/test_state.py::test_model_and_state_move_between_devices[gru]` | то же для ядра GRU | `.venv/bin/python -m pytest -v tests/unit/test_state.py -k test_model_and_state_move_between_devices` |
| `tests/unit/test_state.py::test_model_and_state_move_between_devices[lstm]` | то же для ядра LSTM | `.venv/bin/python -m pytest -v tests/unit/test_state.py -k test_model_and_state_move_between_devices` |
| `tests/unit/test_state.py::test_model_and_state_move_between_devices[none]` | то же без ядра | `.venv/bin/python -m pytest -v tests/unit/test_state.py -k test_model_and_state_move_between_devices` |

## Ручная проверка WandB (без GPU)

Автотесты WandB используют подмену модуля `wandb`, поэтому настоящий WandB проверяется вручную, один раз, на любой машине.

```bash
uv pip install --python .venv/bin/python -e ".[wandb]"
WANDB_MODE=offline .venv/bin/colosseum train -c configs/examples/tic_tac_toe_multi.yaml \
  --set run.name=wandb-check --set metrics.use_wandb=true --set training.total_timesteps=60000
```

`WANDB_MODE=offline` пишет run в `./wandb/` без входа в аккаунт. Для отправки в облако: `wandb sync wandb/offline-run-*` (нужен `WANDB_API_KEY` или `wandb login`). Можно сразу запускать без `WANDB_MODE=offline`, если задан `WANDB_API_KEY`.

На что смотреть (в веб-интерфейсе после `wandb sync`, либо в `wandb/offline-run-*/files/wandb-summary.json` и в определениях метрик run):

- один run на всю тренировку, его имя — имя папки запуска (`wandb-check`);
- метрики обучения каждого агента названы `<agent>/<metric>` (`agent_alpha/total_loss`, `agent_beta/lr`, ...) и строятся по своей оси `<agent>/train_step`; графики обоих агентов есть и не обрезаны, хотя агенты учатся с разной скоростью;
- `ratings/*` (вложенные ключи по вариантам партии, вида `ratings/<layout>/elo/<agent>`, `ratings/<layout>/wr_vs_past/<agent>`), `system/*` (`system/env_steps_per_sec`, `system/queue_depths/<agent>`, ...) и `episodes/*` (`episodes/<agent>/return_mean`, ...) строятся по оси `env_steps`;
- в консоли нет предупреждений `WandB ... disabled`, код выхода 0;
- те же значения есть в `runs/wandb-check/metrics.jsonl` (это источник истины; WandB только показывает их).

После проверки: `rm -rf runs/wandb-check wandb`.

Распределённые роли (`run-learner`, `run-workers`) до SP5 не пишут ни в WandB, ни в `metrics.jsonl`: у них есть только логи процессов.

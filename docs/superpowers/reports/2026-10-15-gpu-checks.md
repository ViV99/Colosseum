# Проверки на GPU (RTX 4070 Ti), 2026-10-15

- Что это: прогон `docs/GPU_CHECKS.md` на первой машине с CUDA по просьбе владельца (вопрос 3 отчёта о приёмке SP3, [`2026-10-10-sp3-acceptance.md`](2026-10-10-sp3-acceptance.md)). Это не подпроект: ветка `gpu-checks` от `main` `98b2ab6`; правка теста, скрипт замера, документация, после ответов владельца — голова политики в fp32 под AMP. Владелец принял ветку 2026-10-15; она слита в `main` через `git merge --no-ff` (слияние — следующий коммит после последнего коммита ветки, поэтому один `git revert -m 1 <merge>` её откатывает).
- Дата: 2026-10-15.
- Машина: WSL2 (Linux 6.18.54.1-microsoft-standard-WSL2), 20 логических ядер (`nproc`), 25 ГБ ОЗУ, NVIDIA GeForce RTX 4070 Ti (12 ГБ, Ada, sm_89), драйвер 610.88 (nvidia-smi: CUDA UMD 13.3).
- Версии: Python 3.12.15, torch 2.14.1+cu130 (CUDA 13.0, cuDNN 9.24.0 = `torch.backends.cudnn.version()` 92400), установлено `scripts/setup-dev.sh --gpu`.
- Окружение: на машине в `~/.bashrc` подключён ROS iron, поэтому каждая команда `.venv/bin/python` / `.venv/bin/colosseum` / `.venv/bin/ruff` ниже запускалась с префиксом `env -u PYTHONPATH` (см. §9).
- Коммиты ветки (по порядку): `2035c57` (`test:` сравнение переноса между устройствами с выключенным TF32 cuDNN), `d825db6` (`feat:` `scripts/measure_amp_logratio.py`), `14d64dd` и `314be73` (`docs:` этот отчёт, `docs/GPU_CHECKS.md`, `docs/benchmarks.md`, README, CLAUDE.md; исправления по ревью), `5470cb6` (`refactor:` скрипт), после ответов владельца — `a2d00a6` (`feat:` голова политики в fp32 под autocast, G-R6), `136c190` (`refactor:` скрипт с `--amp-head`) и `docs:` с этим разделом.

## Итог

**Все проверки `docs/GPU_CHECKS.md` пройдены.** Единственное падение (2 из 22 `gpu`-тестов) — следствие TF32 в cuDNN RNN, а не ошибка кода; тест исправлен (решение G-R1), после исправления 22 из 22. Обучение на CUDA впервые прошло целиком: медленный набор с лёрнерами на CUDA, быстрый набор на машине с CUDA (лёрнер на CUDA — в его интеграционных тестах на `configs/examples/`), пайплайн README под AMP float16 и bfloat16, resume AMP-прогона, распределённый лёрнер на CUDA. Найдены два вопроса численной точности (AMP-шум в log-ratio при `Box`-действиях с малым σ; диагностика `rho_clip_frac` считает численный шум) и две особенности окружения (скачки часов WSL2, `PYTHONPATH` от ROS). В первом проходе код фреймворка не менялся; после ответов владельца `ComposedModel` считает голову политики под AMP в float32 (решение G-R6, раздел «Ответы владельца и что сделано после» в конце). Все четыре вопроса владельцу закрыты: Q1 — сделано, Q2 — вариант 3 в SP4, Q3 — далёкое будущее, Q4 — ветка принята и слита.

| № | Проверка | Результат |
|---|---|---|
| 1 | 22 `gpu`-теста | до исправления 20 passed, 2 failed (`test_state.py` `[lstm]`, `[gru]`, TF32); после `2035c57` — **22 passed**; с тестами головы в fp32 (`a2d00a6`) — **25 passed** |
| 2 | Быстрый набор на машине с CUDA | **1689 passed, 32 deselected, 311.65 с**, exit 0, без предупреждений; ruff чистый; интеграционные тесты на `configs/examples/` учат лёрнер на CUDA; после `a2d00a6` — **1690 passed, 35 deselected, 297.74 с**; после `fa8bf27` (последний раунд исправлений) — **1690 passed, 35 deselected, 311.07 с** |
| 3 | Медленный набор, лёрнеры на CUDA | **10 passed за 423.96 с** (на CPU при приёмке SP3 — 910.0 с); все пороги пройдены, значения близки к CPU при приёмке SP3 (§3) |
| 4 | Пайплайн README под AMP (fp16 и bf16) | все команды exit 0; нет NaN/inf; `critic_warmup` только на первых 30 шагах; λ убывает; trained против random **1.000**, против greedy **0.500** (оба dtype) |
| 5 | M-6: log-ratio при нулевом лаге под AMP | измерено (скрипт `scripts/measure_amp_logratio.py`): было fp16 p95 до 2.51e-2, clip_fraction 0; bf16 до 0.19 p95 и 4.4 % клипа при σ = e^−4; **исправлено в коде** (`a2d00a6`, G-R6: голова политики в fp32 под autocast): bf16 до 0.10 p95 и 0.2 % клипа, fp16 до 1.29e-2 |
| 6 | Доп.: resume AMP на CUDA, CUDA-чекпоинты без CUDA, распределённые роли с лёрнером на CUDA, WandB | всё exit 0 (лёрнер распределённого режима — 143 по SIGTERM, как положено); состояние GradScaler восстанавливается, прогрев не повторяется |
| 7 | Лёрнер CUDA против CPU (нагрузка `bench_throughput.py`) | CUDA-лёрнер быстрее в 1.8–2.8× (нагрузка упирается в лёрнер) |
| 8 | Часы WSL2 | `time.time()` скачет на +9.3 с каждые ~5.4 с; `bench_throughput.py` на этой машине не работает (G-R4); код фреймворка не затронут |
| 9 | ROS `PYTHONPATH` | pytest падает при старте; обход — `env -u PYTHONPATH` |

---

## §1 Тесты `gpu`

Команда: `env -u PYTHONPATH .venv/bin/python -m pytest -m gpu -v -rA` (7.25 с).

- До исправления: **20 passed, 2 failed** — `tests/unit/test_state.py::test_model_and_state_move_between_devices[lstm]` и `[gru]`, на финальном сравнении состояния (`torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-5)`): LSTM — наибольшая абсолютная разница 5.79e-05 (10 из 48 элементов), GRU — 8.67e-05 (24 из 48). Проверка log-prob прошла.
- Причина: по умолчанию torch на GPU Ampere и новее включает TF32 для cuDNN conv и cuDNN RNN (`torch.backends.cudnn.allow_tf32 == True`; новый API: `torch.backends.cudnn.rnn.fp32_precision == "tf32"`), а матричные умножения cuBLAS остаются в IEEE fp32 (`torch.backends.cuda.matmul.allow_tf32 == False`). Ядра LSTM/GRU идут через cuDNN RNN, поэтому их выходы на CUDA отличаются от CPU fp32 на ~1e-4 (у TF32 мантисса 10 бит). Ядра `none` и `attention` используют обычные matmul и проходят. Та же проверка с `torch.backends.cudnn.allow_tf32 = False`: максимум |разницы состояния| 2.53e-06 (LSTM), 1.88e-06 (GRU).
- Исправление (`2035c57`): строгое сравнение (rtol 1e-4, atol 1e-5, log-prob и состояние) идёт внутри `torch.backends.cudnn.flags(enabled=True, allow_tf32=False)` — только старый API (смешение со `fp32_precision` в torch 2.14 даёт `RuntimeError` о разных флагах TF32; старый API в torch 2.14 предупреждений не выдаёт, проверено с `-W error`; `pyproject` допускает torch ≥ 2.6, где есть только старый API). Второй шаг на CUDA с флагами по умолчанию проверяет log-prob и состояние с допуском TF32 (rtol 1e-2, atol 1e-3: мантисса 10 бит, наблюдаемая разница ~1e-4).
- После исправления: `env -u PYTHONPATH .venv/bin/python -m pytest -v tests/unit/test_state.py -k test_model_and_state_move_between_devices` → 4 passed; `env -u PYTHONPATH .venv/bin/python -m pytest -m gpu -v` → **22 passed, 1699 deselected in 2.45 s**.

## §2 Быстрый набор

Команда: `env -u PYTHONPATH timeout 3600 .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw -p no:cacheprovider --durations=25`.

- **1689 passed, 32 deselected in 311.65 s**, exit 0, предупреждений нет; `env -u PYTHONPATH .venv/bin/ruff check .` — чисто.
- Повторно на коде ветки на коде ветки (`2035c57`, `d825db6`; прогон перед коммитом документации, тесты документацию не читают): `env -u PYTHONPATH timeout 1200 .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw -p no:cacheprovider` → **1689 passed, 32 deselected in 288.62 s**, exit 0, предупреждений нет; ruff чистый.
- Интеграционные тесты, которые учат на конфигах `configs/examples/` (`learner.device: auto`), использовали CUDA: повторный прогон `test_demo_train_smoke.py`, `test_sp3_pipeline_smoke.py`, `test_sp2_ttt_example.py`, `test_reference_examples_smoke.py` с `--basetemp` → 5 passed, во всех 5 логах лёрнера `starting on device=cuda`. Тестовые конфиги из `game_helpers.write_test_config` закрепляют `learner.device: cpu`, поэтому остальные интеграционные тесты учат лёрнер на CPU.

## §3 Медленный набор

Команда: `env -u PYTHONPATH timeout 3600 .venv/bin/python -m pytest -m slow -v -rA -p no:cacheprovider --basetemp=... --durations=0`.

**10 passed in 423.96 s** (приёмка SP3 на 8-ядерной машине без GPU: 910.0 с); во всех 10 логах лёрнера `device=cuda`. Машина и на CPU быстрее (20 ядер против 8), поэтому отношение времён — не чистый эффект GPU.

Напечатанные значения (в скобках — CPU при приёмке SP3; порог):

| Тест | Обучение | Шагов сред/с | Результат |
|---|---|---|---|
| coin_grid | 20 с | 20351 | score 11.075, random 2.345, **4.72×** [4.01×; разброс сидов 3.8–5.4×] (≥ 2×) |
| tic_tac_toe | 29 с | 25707 | **win 0.940** [0.927] (≥ 0.80) |
| unit_harvest | 68 с | 4310 | **win 1.000** [1.000] (≥ 0.80) |
| team_tag (один прогон) | 78 с | 5077 | **win 0.955**, ничьи 0.045, средняя длина 6.725 [0.940] (≥ 0.80) |
| tron | 60 с | 6089 | **2p 0.990, первое место 4p 0.953** [0.965 / 0.932] |
| predator_prey | 40 с | 11661 | **hunter 1.000, prey 0.985** (random против random 0.550) [1.000 / 0.990] (≥ 0.70) |
| coop_buttons | 31 с | 16122 | **coop_a 7.390, coop_b 7.390** (порог 4.404, оракул 7.340) [7.390 / 7.390] |
| пайплайн unit_harvest (seed 0) | record 8 / bc 5 / train 49 / eval 10 с | 3960 | 30000 решений; **против random W/D/L 60/0/0, win 1.000; против greedy 0/58/2, score 0.483; против bc_net 0/58/2, score 0.483** [1.000 / 0.492 / 0.483] (пороги 0.95 / 0.40 / 0.45) |

`test_vtrace_slots_torch_compile`, `test_appo_torch_compile`: PASSED.

## §4 Пайплайн README под AMP

Шаг 3 `docs/GPU_CHECKS.md`, команды дословно (с префиксом `env -u PYTHONPATH`; прогон float16 явно задавал `--set algorithm.amp_dtype=float16` — значение по умолчанию); `run.name` `gpu-check-amp` для float16 и `gpu-check-amp-bf16` с `--set algorithm.amp_dtype=bfloat16`; `eval` дополнительно с `--output`:

```bash
CFG=configs/examples/unit_harvest_league.yaml
env -u PYTHONPATH .venv/bin/colosseum record -c $CFG --player greedy --num-matches 300 --output gpu-check/data --seed 0
env -u PYTHONPATH .venv/bin/colosseum bc -c $CFG --agent main --data gpu-check/data --output gpu-check/bc.pt --epochs 5 --seed 0
env -u PYTHONPATH .venv/bin/colosseum train -c $CFG --set run.name=gpu-check-amp --set algorithm.use_amp=true \
  --set agents.main.init.from=gpu-check/bc.pt --set agents.main.init.critic_warmup_steps=30 \
  --set agents.main.kickstart.teacher=greedy
NEW=$(ls -d runs/gpu-check-amp/checkpoints/main/ckpt_v* | sort -V | tail -1)
env -u PYTHONPATH .venv/bin/colosseum eval -c $CFG -a trained=$NEW -a greedy -a random --num-matches 100 --deterministic
```

- Все команды завершились с кодом 0.
- `record` 9 с (300 матчей, 30000 решений, средний возврат greedy 7.535); `bc` 5 с (NLL по эпохам 0.4334, 0.0033, 0.0008, …). BC шёл на CUDA: процесс `colosseum bc` держал контекст CUDA (`nvidia-smi --query-compute-apps`); своё устройство `bc` в лог не пишет (открытый пункт).
- `train` 55 с (fp16) / 54 с (bf16), `eval` по 9 с. В логах лёрнера `starting on device=cuda`.
- `metrics.jsonl` (`kind: train`, запись каждые 10 шагов обучения): fp16 — 97 записей, bf16 — 101; ни в одном поле нет NaN/inf; `skipped_updates` ненулевой в 1 из 97 записей (fp16; значение 1), в bf16 — 0; `critic_warmup == 1` только на записанных шагах 1, 11, 21 (прогрев 30 шагов); `kickstart_lambda` не возрастает: 1.0 → 0.9814 на шаге 961 (fp16), → 0.9806 на шаге 1001 (bf16) — ровно линейное убывание за `kickstart.decay_steps` = 50000 шагов обучения после прогрева (1 − (961 − 30)/50000 = 0.98138), то есть λ убывает, при этом бюджете медленно — так задумано.
- `eval` (`--deterministic`, 100 матчей на пару): fp16 — trained против random W/D/L 100/0/0, **win 1.000**; trained против greedy 0/100/0, **score 0.500**; bf16 — то же (1.000 / 0.500). Пороги 0.95 / 0.40 (CPU при SP3: 1.00 / 0.50).
- Финальные чекпоинты: `ckpt_v969` (fp16), `ckpt_v1003` (bf16).

## §5 M-6: log-ratio при нулевом лаге под AMP

Пункт ручной проверки `docs/GPU_CHECKS.md` (финальное ревью SP2, область A, M-6): под AMP `gaussian_log_prob` / `gaussian_entropy` (и для `Box` внутри `Units`) считаются в fp16, потому что записанные float32-действия приводятся к `mean.dtype`. Теперь это скрипт: `env -u PYTHONPATH .venv/bin/python scripts/measure_amp_logratio.py [--fp32-head] [--fp32-gauss] [--json PATH]` (`d825db6`; повторные прогоны дают те же числа). Ниже — замер до перевода головы политики в fp32; с `136c190` флаг `--fp32-head` заменён на `--amp-head`, а прежние колонки «fp32 head» стали поведением по умолчанию (раздел «Ответы владельца и что сделано после»).

Постановка: роли с наблюдением `Box(-1, 1, (16,))` и действиями `Units(6, Box(4))` (`units_box4`), `Units(6, Dict(move=Discrete(5), aim=Box(2)))` (`units_dict`), `Units(6, Discrete(5))` (`units_discrete`, база: категориальные логиты под autocast тоже в fp16/bf16); модели `make_test_model` с ядрами `none` (hidden 64) и `lstm` (hidden 256); все `log_std` = 0, −2, −4; 8 чанков `synthetic_chunk(..., logp_noise=0.0, random_units=True)` — log-prob поведения от CPU fp32 модели, как на воркере, то есть лаг нулевой. Лёрнер на CUDA (`APPO(..., device="cuda")`, путь оценки обучения) в fp32 / AMP fp16 / AMP bf16. Ниже — per-decider |log-ratio| (p95 / max) по валидным решающим на `act`-слотах; `eps_clip` по умолчанию 0.2, поэтому `clip_fraction` считает |ratio − 1| > 0.2. Колонки «fp32 head» — с флагом `--fp32-head` (голова политики в float32 под autocast).

| роль | ядро | log_std | fp32 p95 / max | fp16 p95 / max | bf16 p95 / max | bf16 clip_fraction | fp16 p95 (fp32 head) | bf16 p95 (fp32 head) |
|---|---|---|---|---|---|---|---|---|
| units_box4 | lstm | 0.0 | 3.05e-05 / 8.39e-05 | 2.12e-03 / 7.37e-03 | 1.76e-02 / 4.16e-02 | 0.0 | 6.39e-05 | 3.04e-04 |
| units_box4 | lstm | -2.0 | 2.28e-04 / 6.20e-04 | 3.46e-03 / 1.07e-02 | 2.49e-02 / 8.73e-02 | 0.0 | 4.75e-04 | 2.24e-03 |
| units_box4 | lstm | -4.0 | 1.68e-03 / 4.58e-03 | 1.96e-02 / 5.28e-02 | 1.37e-01 / 3.35e-01 | 0.0107 | 3.51e-03 | 1.66e-02 |
| units_box4 | none | 0.0 | 4.77e-07 / 1.91e-06 | 2.17e-03 / 5.77e-03 | 1.78e-02 / 5.67e-02 | 0.0 | 2.36e-04 | 1.84e-03 |
| units_box4 | none | -2.0 | 1.19e-06 / 2.98e-06 | 4.10e-03 / 1.07e-02 | 3.18e-02 / 8.26e-02 | 0.0 | 1.75e-03 | 1.36e-02 |
| units_box4 | none | -4.0 | 9.54e-06 / 2.10e-05 | 2.51e-02 / 7.05e-02 | 1.94e-01 / 4.89e-01 | 0.0441 | 1.29e-02 | 1.00e-01 |
| units_dict | lstm | 0.0 | 2.62e-05 / 7.15e-05 | 1.65e-03 / 4.96e-03 | 1.20e-02 / 4.04e-02 | 0.0 | 5.25e-05 | 2.55e-04 |
| units_dict | lstm | -2.0 | 1.63e-04 / 4.39e-04 | 2.63e-03 / 9.73e-03 | 1.91e-02 / 6.11e-02 | 0.0 | 3.48e-04 | 1.74e-03 |
| units_dict | lstm | -4.0 | 1.18e-03 / 3.16e-03 | 1.66e-02 / 4.31e-02 | 1.01e-01 / 3.33e-01 | 0.0067 | 2.52e-03 | 1.29e-02 |
| units_dict | none | 0.0 | 4.77e-07 / 9.54e-07 | 1.58e-03 / 5.93e-03 | 1.25e-02 / 4.53e-02 | 0.0 | 2.00e-04 | 1.56e-03 |
| units_dict | none | -2.0 | 7.15e-07 / 1.91e-06 | 3.04e-03 / 1.00e-02 | 2.48e-02 / 5.93e-02 | 0.0 | 1.30e-03 | 1.04e-02 |
| units_dict | none | -4.0 | 5.25e-06 / 1.38e-05 | 1.94e-02 / 4.48e-02 | 1.60e-01 / 4.59e-01 | 0.0245 | 9.49e-03 | 7.74e-02 |
| units_discrete | lstm | 0.0 | 1.37e-05 / 2.73e-05 | 1.33e-04 / 2.50e-04 | 9.07e-04 / 1.97e-03 | 0.0 | 2.91e-05 | 1.32e-04 |
| units_discrete | none | 0.0 | 1.19e-07 / 2.38e-07 | 2.01e-04 / 4.08e-04 | 1.44e-03 / 3.10e-03 | 0.0 | 1.05e-04 | 8.31e-04 |

- `clip_fraction` для fp16 равен 0.0 везде; ненулевой только у bf16 при `log_std` −4 (колонка выше).
- `--fp32-gauss` (повышение точности только внутри `gaussian_log_prob`): `units_box4` fp16 p95 2.17e-03 → 1.56e-03 (none, 0.0), 4.10e-03 → 3.70e-03 (none, −2), 2.51e-02 → 2.48e-02 (none, −4); bf16 1.94e-01 → 1.93e-01 (none, −4) — почти без изменений.
- Чтение: ошибка возникает в `mean`, который голова политики (её `Linear`) под autocast выдаёт в низкой точности (относительное округление 2^-11 у fp16, 2^-8 у bf16), и усиливается множителем (x − μ)/σ², поэтому при малом σ растёт почти как 1/σ (от σ = e^−2 до e^−4 p95 растёт в 5.3–6.5 раза при σ, меньшем в 7.4 раза; от σ = 1 до e^−2 — лишь в 1.4–2.0 раза). Голова в fp32 уменьшает p95 шума в 1.7–58× (сильнее всего с ядром LSTM при log_std 0: 31–58×; слабее всего без ядра при log_std −2 / −4: 1.9–2.4×; у units_discrete 1.7–6.9×), но не до нуля: входные признаки головы тоже округлены. bf16 в 6–8× шумнее fp16. Колонка fp32 у ядра LSTM — эффект TF32 (cuDNN RNN, §1), у ядра `none` — чистый шум fp32.
- `rho_clip_frac` при нулевом лаге показывает 0.13–0.54 во всех режимах, и в fp32 тоже: метрика считает ρ > ρ̄ = 1, то есть любой численный шум выше 1 (открытый пункт, вопрос Q2).

## §6 Дополнительные сквозные проверки (в `docs/GPU_CHECKS.md` их нет)

### Resume AMP-прогона fp16 на CUDA

`env -u PYTHONPATH .venv/bin/colosseum train -c configs/examples/unit_harvest_league.yaml --set run.name=gpu-check-amp-resume --set algorithm.use_amp=true --set training.resume_from=runs/gpu-check-amp --set training.total_timesteps=240000 --set agents.main.init.critic_warmup_steps=30 --set agents.main.kickstart.teacher=greedy`

- exit 0, `Resume [main]: .../ckpt_v969 (policy_version 969)`, счётчик шагов сред продолжается с 161136, пул снимков перенесён (10 снимков), лёрнер `device=cuda`.
- `trainer_state.pt`: GradScaler v969 — scale 4096, `_growth_tracker` 945 → в первом чекпоинте после resume (v1000) scale 4096, `_growth_tracker` 976 (+31 = шаги 969 → 1000: состояние восстановлено, не сброшено); `critic_warmup_done` 30 в обоих (второго прогрева нет); шаг kickstart 939 → 970.
- metrics: шаги обучения 970 → 1390, ни одной записи с `critic_warmup == 1`, λ 0.98122 → 0.97282, `skipped_updates` 0, нет неконечных значений.

### CUDA-чекпоинты без CUDA

`CUDA_VISIBLE_DEVICES=""` (`torch.cuda.is_available()` = False):

- `colosseum eval -a trained=<ckpt_v1399 продолженного прогона> -a random --num-matches 50 --deterministic` — exit 0, win 1.000;
- `colosseum train --set agents.main.init.from=<тот же чекпоинт> --set training.total_timesteps=20000` — exit 0, лёрнер `device=cpu`, `init ... (strict): 20 tensors`;
- resume CUDA/AMP-прогона на CPU с `use_amp=true` и с `false`: оба exit 0, лёрнер `device=cpu`, в логе `Resume: the saved GradScaler state is ignored because this run has no GradScaler.`

### Распределённые роли на localhost, лёрнер на CUDA под AMP fp16

`configs/examples/tic_tac_toe.yaml`, `total_timesteps` 200000, `checkpoint.interval` 20; `serve-weight-store --port 50771`, `run-learner --agent agent_0 --traj-port 50772 --weight-store localhost:50771`, `run-workers --weight-store localhost:50771 -l agent_0=localhost:50772`.

- `run-workers` exit 0 через 34 с; лёрнер `starting on device=cuda`, 801 шаг обучения, финальный чекпоинт v801 сохранён, exit 143 по SIGTERM; ожидаемое предупреждение «matchmaking.opponents ... reduced to latest only»; ошибок ни в одном логе нет.
- `serve-weight-store` завершается с кодом 130 по SIGINT (проверено при обычной диспозиции SIGINT; процесс, запущенный через `&` из неинтерактивного bash-скрипта, наследует SIGINT=ignored — это поведение shell, не ошибка).

### WandB

Ручная проверка из `docs/GPU_CHECKS.md`, wandb 0.30.0, `WANDB_MODE=offline`, команда из документа дословно:

- exit 0; один run с именем `wandb-check`; 118 записей `define_metric`: `agent_alpha/*` и `agent_beta/*` по оси `<agent>/train_step`, `ratings/*`, `system/*`, `episodes/*` по оси `env_steps`; ключи как в документе (`ratings/2p/elo/<agent>`, `system/env_steps_per_sec`, `system/queue_depths/<agent>`, `episodes/<agent>/return_mean`, …); предупреждения «WandB ... disabled» нет; значения совпадают с `metrics.jsonl` (agent_beta train_step 111, total_loss 0.3159320652484894, ELO 1154.92 / 1245.08).
- `wr_vs_past`: в `tic_tac_toe_multi.yaml` (соперники `rivals: 1.0`) агент никогда не играет со своими снимками, поэтому ключ `ratings/<layout>/wr_vs_past/<agent>` там не появляется (значения None не пишутся). На `configs/examples/tic_tac_toe.yaml` с `--set rollout.match_refresh_interval_sec=5 --set training.total_timesteps=400000` он есть: `ratings/2p/wr_vs_past/agent_0` = 0.718 по оси `env_steps`, как в `metrics.jsonl` (`past_games` 500).
- wandb 0.30 в offline-режиме не пишет `files/wandb-summary.json`; значения лежат в `run-*.wandb` (или смотреть после `wandb sync`).
- Пакет wandb после проверки удалён (окружение осталось таким, каким его делает `setup-dev.sh --gpu`).

## §7 Лёрнер на CUDA и на CPU (эта машина)

Нагрузка — зафиксированная `scripts/bench_throughput.py::_make_config` (крестики-нолики, 8 сред на воркер, chunk 32, батч 8 чанков, `torch_threads` лёрнера auto); 60 с на прогон по таймеру остановки; скорости = итоговые счётчики (сумма `env_steps` воркеров, `Total train_steps` лёрнера) / 60 с, запуск включён. Сам `bench_throughput.py` на этой машине не работает (§8).

| Воркеры | CPU-лёрнер, шаги сред/с | CPU, апдейты/с | CUDA-лёрнер, шаги сред/с | CUDA, апдейты/с |
|---|---|---|---|---|
| 1 | 5103 | 20.5 | 9387 | 37.8 |
| 2 | 6465 | 26.0 | 18358 | 74.0 |
| 4 | 12229 | 49.2 | 22941 | 92.4 |

С CPU-лёрнером нагрузка упирается в лёрнер (очередь 27–31 из 32 в пробном запуске `bench_throughput.py`), поэтому CUDA-лёрнер даёт 1.8–2.8×. С числами 8-ядерной машины из `docs/benchmarks.md` не сравнимо (другая машина, другой способ подсчёта).

## §8 Скачки часов WSL2

- `time.time()` против `time.monotonic()` за 30 с: скачки на +9.33…+9.35 с, каждый отменяется через 0.25 с (−9.74…−9.77 с), примерно каждые 5.4 с (10 скачков за 30 с).
- Следствие: `ts` в `metrics.jsonl` (`metrics/jsonl.py`, `time.time()`) и метки времени логов не монотонны (11 инвертированных пар среди 2224 записей `train` 60-секундного прогона); `scripts/bench_throughput.py` (окна по `ts`) завершался ошибкой при каждом запуске (с CUDA-лёрнером отбракованы все числа воркеров, с CPU-лёрнером — хотя бы одно; JSON не записан) («first sample … after the window start» / «last sample … before the window end»).
- Код не затронут: все интервалы, таймауты и дедлайны используют `time.monotonic()` (31 место); `time.time()` встречается только для меток времени (`metrics/jsonl.py`, метка в `meta` и проверка возраста временных папок в `checkpoint_manager.py`).
- Вероятно, проблема синхронизации времени хоста/WSL (на стороне владельца стоит попробовать `wsl --shutdown`).

## §9 Окружение ROS

В `~/.bashrc` машины подключён ROS iron: `PYTHONPATH=/opt/ros/iron/lib/python3.10/site-packages`. С ним pytest падает при старте: точка входа `pytest11` пакета ROS `launch_testing` → `ModuleNotFoundError: No module named 'lark'`. ROS не подменяет ни одной зависимости проекта, ломается только автозагрузка плагинов pytest. Обход: префикс `env -u PYTHONPATH` у каждой команды `.venv/bin/python` / `.venv/bin/colosseum` / `.venv/bin/ruff`. Конфигурация машины не менялась.

## Решения контроллера («что — почему — цена ошибки»)

- **G-R1 TF32.** Фреймворк оставляет умолчания torch для TF32 (никакой глобальной настройки точности в лёрнере) — почему: TF32 ускоряет cuDNN RNN/conv на Ampere и новее, а его влияние на log-prob лёрнера намного меньше диапазона клипа PPO (измерено: при нулевом лаге per-decider |log-ratio| p95 1.4e-5–3e-5 с ядром LSTM для `Discrete` / `Box` с `log_std` 0, до 1.7e-3 для `Box` с `log_std` −4; `eps_clip` 0.2) — цена ошибки: расхождение log-prob лёрнера и воркера такого размера; ручку точности `learner` можно добавить позже. Тест проверяет корректность переноса между устройствами, поэтому сравнивает с выключенным TF32 и отдельно закрепляет поведение TF32 по умолчанию с допуском TF32.
- **G-R2 M-6** (заменено G-R6). Код сейчас не меняется — почему: AMP включается явно (`algorithm.use_amp` по умолчанию false), дискретные действия почти не затронуты (fp16 p95 ≤ 2.01e-4), fp16 с `Box` остаётся ≤ 2.51e-2 p95 даже при σ = e^−4 (clip_fraction 0); настоящее исправление — проектный выбор (голова политики в fp32) → вопрос Q1 — цена ошибки: прогоны bf16 с гауссовыми политиками малого σ получают шум ratio вблизи диапазона клипа PPO (p95 0.10–0.19, clip_fraction до 4.4 % при нулевом лаге).
- **G-R3 `rho_clip_frac` / `c_clip_frac`.** Оставлены как есть (семантика диагностики; наблюдаемость SP4) → вопрос Q2 — цена ошибки: вводящая в заблуждение диагностика (0.13–0.54 при нулевом лаге на CUDA, 0.01–0.08 на CPU).
- **G-R4 `bench_throughput.py`.** Не менялся из-за скачков часов (проблема окружения; контроллер мерил по счётчикам) — цена ошибки: бенчмарк не работает на машинах со скачущими настенными часами.
- **G-R5 Ветка.** `gpu-checks` (не подпроект): правка теста + скрипт + документация; слияние в `main` только после явного «да» владельца (получено 2026-10-15, Q4).

## Открытые пункты

- ~~SP6 / вопрос Q1: голова политики в fp32 под AMP (числа M-6 — §5).~~ Сделано (`a2d00a6`, G-R6).
- SP4 / вопрос Q2 (владелец выбрал вариант 3: заменить на обрезанную массу весов, см. ниже): `rho_clip_frac` / `c_clip_frac` считают численный шум выше ρ̄ = 1 (0.13–0.54 при нулевом лаге на CUDA, и в fp32 тоже; на CPU при нулевом лаге тоже не 0: 0.01–0.08 на чанках M-6).
- Далёкое будущее / вопрос Q3: проверить, нужна ли на самом деле ручка TF32 у лёрнера (владелец, 2026-10-15); пока действуют умолчания torch (TF32 в cuDNN RNN/conv).
- SP6: на RTX 4070 Ti AMP медленнее fp32 для моделей размера примеров (на моделях примеров на 1–13 % медленнее с головой под autocast и на 3–16 % — с головой в fp32; на LSTM-512 примерно на 50 %); вернуться к AMP (или рекомендовать против него) вместе с быстрыми путями.
- Окружение: скачки настенных часов WSL2 ломают `scripts/bench_throughput.py` и делают `ts` в `metrics.jsonl` немонотонным на CUDA-машине владельца; `PYTHONPATH` от ROS требует `env -u PYTHONPATH`.
- `colosseum bc` не пишет в лог своё устройство (мелочь, без срока).

## Вопросы владельцу

Ответы владельца — после каждого вопроса; подробности и что сделано — в разделе ниже.

1. **Q1 (M-6).** Считать голову политики в float32 под AMP (например, `ComposedModel` вызывает `self.policy(...)` внутри `torch.autocast(device_type="cuda", enabled=False)` с float32-входами)? Измеренный выигрыш — 1.9–58× по p95 шума при нулевом лаге для Box / Dict (bf16, LSTM, σ = e^−4: p95 0.137 → 0.017, 8×; меньше всего без ядра при σ = e^−4: bf16 0.194 → 0.100, 1.9×), цена — matmul головы в fp32 (для типичных голов мало). Рекомендация контроллера: да, как небольшой пункт SP6 или сейчас отдельным исправлением; до тех пор — рекомендации в `docs/GPU_CHECKS.md`. **Ответ:** «Окей давай если дешево то fp32 если мы уверены что проблема именно в этом и нам дешево так делать» → сделано (`a2d00a6`, G-R6).
2. **Q2.** Считать в `rho_clip_frac` / `c_clip_frac` только ρ заметно выше ρ̄ (например, `ρ > ρ̄·(1 + 1e-3)`) или показывать вместо доли обрезанную массу? SP4. **Ответ:** «Для rho давай твое предложение 3й вариант» → в SP4 заменить на обрезанную массу весов (ниже).
3. **Q3.** Добавить ручку точности лёрнера (`learner.tf32` или подобную)? Контроллер: сейчас не нужна (G-R1). **Ответ:** «Пока нет, можно записать на далекое будущее проверку реальной надобности» → пункт на далёкое будущее.
4. **Q4.** Принять ветку `gpu-checks` и слить её в `main`? **Ответ:** «И да можешь залить в main» → ветка принята 2026-10-15 и слита в `main` (`--no-ff`).

## Ответы владельца и что сделано после

Ответы владельца (2026-10-15, в чате):

1. Q1: «Окей давай если дешево то fp32 если мы уверены что проблема именно в этом и нам дешево так делать».
2. Q2: сначала «Тут подробнее - что это и какие варианты» (контроллер объяснил в чате и предложил четыре варианта), затем «Для rho давай твое предложение 3й вариант».
3. Q3: «Пока нет, можно записать на далекое будущее проверку реальной надобности».
4. Q4: «И да можешь залить в main».

### Проверка условий владельца по Q1

(A) «Проблема именно в этом?» — разложение p95 per-decider |log-ratio| при нулевом лаге, `units_box4` (`Units(6, Box(4))`), `log_std` −4, та же постановка, что в `scripts/measure_amp_logratio.py`; «head32» — голова политики под `autocast(enabled=False)` на float32-входах, «head32+enc32» — дополнительно энкодер в float32 (замер контроллера):

| ядро | AMP dtype | как было | head32 | head32+enc32 |
|---|---|---|---|---|
| none (hidden 64) | float16 | 2.51e-02 | 1.29e-02 | 9.54e-06 |
| none (hidden 64) | bfloat16 | 1.94e-01 | 1.00e-01 | 9.54e-06 |
| lstm (hidden 256) | float16 | 1.96e-02 | 3.51e-03 | 2.99e-03 |
| lstm (hidden 256) | bfloat16 | 1.37e-01 | 1.66e-02 | 2.99e-03 |

Чтение: когда в fp32 и голова, и энкодер, шум на полу fp32 (9.54e-06 = колонка fp32), значит у шума ровно два источника: собственный выход головы в низкой точности (`mean`) и признаки в низкой точности, которые энкодер выдаёт под autocast. С ядром LSTM главный источник — голова (8× для bf16); без ядра — примерно половина (остальное — выход энкодера в bf16/fp16). Остаток 2.99e-03 с ядром LSTM — cuDNN (TF32 и LSTM в float16 под autocast: выход ядра был float16 даже под bfloat16-autocast, латент энкодера — bfloat16). Часть энкодера — это и есть AMP (его matmul в низкой точности); держать его в fp32 значило бы отказаться от AMP для ствола, поэтому он остаётся.

(B) «Дёшево ли?» — мс на один `APPO.train_step` на CUDA (с синхронизацией; лучший из 3–4 повторов по 200 шагов для батча 8×32 и по 40 шагов для 16×128; настоящие модели примеров собраны `build_model` из конфигов):

| модель | батч | AMP fp16: как было → голова в fp32 | AMP bf16: как было → голова в fp32 | без AMP (fp32) |
|---|---|---|---|---|
| `unit_harvest_league.yaml` (`main`) | 8×32 | 11.08 → 11.94 мс (+7.8 %) | 12.02 → 12.34 мс (+2.6 %) | 10.63 мс |
| `tic_tac_toe.yaml` | 8×32 | 8.31 → 8.41 мс (+1.2 %) | 8.15 → 8.37 мс (+2.8 %) | 7.57 мс |
| `team_tag.yaml` | 8×32 | 9.50 → 9.62 мс (+1.3 %) | 9.82 → 9.98 мс (+1.6 %) | 9.37 мс |
| синтетическая `make_test_model` lstm-512, наблюдение 64, `Units(32, Dict(move=Discrete(9), aim=Box(2)))` | 16×128 | 122.61 → 125.87 мс (+2.7 %) | 128.65 → 127.31 мс (−1.0 %) | 83.20 мс |

Чтение: голова в fp32 стоит около +1–3 % шага AMP-лёрнера (+7.8 % в одном шумном случае; первый, более короткий замер дал −8 %…+20 % на маленьких моделях, то есть шум на таком размере). И ещё: на этом GPU AMP не быстрее обычного fp32 ни для одной из этих моделей — на моделях примеров на 1–13 % медленнее с головой под autocast и на 3–16 % — с головой в fp32; на LSTM-512 примерно на 50 % (fp32 уже использует TF32 в cuDNN; AMP добавляет приведения типов и GradScaler; развёртка LSTM упирается в запуск ядер). Отсюда рекомендация: оставлять `algorithm.use_amp: false` (по умолчанию), если AMP заметно не ускоряет модель.

Вывод контроллера: оба условия выполнены (голова — доказанный независимый источник; цена ~1–3 %), поэтому исправление вошло в код.

### Решение G-R6 (заменяет G-R2)

- **G-R6 Голова политики в fp32.** `ComposedModel` считает голову политики в float32 под CUDA autocast, всегда (без ручки в конфиге) — почему: ответ владельца на Q1, измеренная цена +1–3 % шага AMP, шум в 2–8× ниже (до 58× в случае LSTM с `log_std` 0) — цена ошибки: модель с тяжёлой головой политики (внимание по юнитам в голове) теряет скорость AMP в голове; тогда можно добавить ручку.

G-R2 выше оставлен как был и помечен «заменено G-R6».

### Что сделано

- `a2d00a6` `feat: ComposedModel runs the policy head in float32 under CUDA autocast (M-6)`: `step` и `unroll` вызывают голову через `ComposedModel._policy_head`; пока включён CUDA autocast (`torch.is_autocast_enabled("cuda")`), голова работает внутри `torch.autocast(device_type="cuda", enabled=False)` на `features.float()` и `aux`, в котором вещественные листья приведены к float32 (целые и bool не трогаются); иначе голова получает те же самые тензоры. Голова value, критик-энкодер, распределения и APPO не менялись. Новые тесты `tests/unit/test_fp32_policy_head.py`: CPU — без autocast голова получает тот же объект `features` и тот же `aux` в `step` и `unroll`; `gpu` (float16 / bfloat16) — под autocast голова видит float32-признаки и float32-листья `aux` (целый лист — `long`), autocast внутри головы выключен, после неё снова включён (выход головы value в низкой точности), параметры распределения float32; `gpu` — регрессия M-6: bf16, LSTM 256, `log_std` −4, p95 < 0.05 (до изменения тест падал с 0.1368, после — проходит).
- `136c190` `refactor:` `scripts/measure_amp_logratio.py`: флаг `--fp32-head` заменён на `--amp-head` (возвращает прежнее поведение: `ComposedModel` вызывает голову напрямую). Прогон по умолчанию теперь побайтно совпадает с прежней таблицей `--fp32-head`, прогон `--amp-head` — с прежней таблицей по умолчанию (§5).
- Новая таблица по умолчанию (per-decider |log-ratio|, p95 / max):

| роль | ядро | log_std | fp32 | fp16 | bf16 | bf16 clip_fraction |
|---|---|---|---|---|---|---|
| units_box4 | none | 0.0 | 4.77e-07 / 1.91e-06 | 2.36e-04 / 8.34e-04 | 1.84e-03 / 5.31e-03 | 0 |
| units_box4 | lstm | 0.0 | 3.05e-05 / 8.39e-05 | 6.39e-05 / 1.53e-04 | 3.04e-04 / 7.10e-04 | 0 |
| units_box4 | none | -2.0 | 1.19e-06 / 2.98e-06 | 1.75e-03 / 6.16e-03 | 1.36e-02 / 3.91e-02 | 0 |
| units_box4 | lstm | -2.0 | 2.28e-04 / 6.20e-04 | 4.75e-04 / 1.13e-03 | 2.24e-03 / 5.25e-03 | 0 |
| units_box4 | none | -4.0 | 9.54e-06 / 2.10e-05 | 1.29e-02 / 4.56e-02 | 1.00e-01 / 2.84e-01 | 0.002226 |
| units_box4 | lstm | -4.0 | 1.68e-03 / 4.58e-03 | 3.51e-03 / 8.35e-03 | 1.66e-02 / 3.89e-02 | 0 |
| units_dict | none | 0.0 | 4.77e-07 / 9.54e-07 | 2.00e-04 / 6.81e-04 | 1.56e-03 / 3.09e-03 | 0 |
| units_dict | lstm | 0.0 | 2.62e-05 / 7.15e-05 | 5.25e-05 / 1.44e-04 | 2.55e-04 / 6.65e-04 | 0 |
| units_dict | none | -2.0 | 7.15e-07 / 1.91e-06 | 1.30e-03 / 4.77e-03 | 1.04e-02 / 2.48e-02 | 0 |
| units_dict | lstm | -2.0 | 1.63e-04 / 4.39e-04 | 3.48e-04 / 9.82e-04 | 1.74e-03 / 4.52e-03 | 0 |
| units_dict | none | -4.0 | 5.25e-06 / 1.38e-05 | 9.49e-03 / 3.49e-02 | 7.74e-02 / 1.86e-01 | 0.0004452 |
| units_dict | lstm | -4.0 | 1.18e-03 / 3.16e-03 | 2.52e-03 / 7.19e-03 | 1.29e-02 / 3.29e-02 | 0 |
| units_discrete | none | 0.0 | 1.19e-07 / 2.38e-07 | 1.05e-04 / 2.06e-04 | 8.31e-04 / 1.79e-03 | 0 |
| units_discrete | lstm | 0.0 | 1.37e-05 / 2.73e-05 | 2.91e-05 / 5.70e-05 | 1.32e-04 / 2.32e-04 | 0 |

### Повторные проверки после изменения

- `env -u PYTHONPATH .venv/bin/python -m pytest -m gpu -v` → **25 passed** (22 прежних + 3 новых).
- Быстрый набор `env -u PYTHONPATH timeout 1200 .venv/bin/python -m pytest -m "not gpu and not slow" -q -rw -p no:cacheprovider` → **1690 passed, 35 deselected in 297.74 s**, exit 0, без предупреждений; `ruff check .` чистый.
- Медленный набор (`env -u PYTHONPATH timeout 3600 .venv/bin/python -m pytest -m slow -v -rA -p no:cacheprovider --basetemp=<scratch> --durations=0`; идёт без AMP, поэтому новое поведение в нём не включается) → **10 passed in 406.63 s**, во всех 10 логах лёрнера `device=cuda`. Значения (в скобках — первый прогон §3): coin_grid score 13.265, random 2.345, **5.657×** [4.72×]; tic_tac_toe **win 0.958** [0.940]; unit_harvest **1.000** [1.000]; team_tag **0.980**, ничьи 0.020, средняя длина 6.740 [0.955]; tron **2p 0.975, первое место 4p 0.953** [0.990 / 0.953]; predator_prey **hunter 1.000, prey 0.970** (random против random 0.550) [1.000 / 0.985]; coop_buttons **7.390 / 7.390** (порог 4.404, оракул 7.340) [7.390 / 7.390]; пайплайн unit_harvest: record 8 / bc 5 / train 47 / eval 9 с, 30000 решений, 3783 шагов сред/с, **против random 60/0/0 win 1.000, против greedy 0/58/2 score 0.483, против bc_net 0/58/2 score 0.483** [те же]; обе проверки torch.compile — PASSED. Разница с первым прогоном — разброс между прогонами (обучение на CUDA недетерминировано), все пороги пройдены.
- Пайплайн README под AMP (команды шага 3 `docs/GPU_CHECKS.md` с `env -u PYTHONPATH`, все выходные файлы вне репозитория: `--output` в рабочий каталог и `--set run.dir=<scratch>/runs`; `eval` с `--output`): все команды exit 0; record — 30000 решений, средний возврат greedy 7.535; bc — NLL по эпохам 0.4334, 0.0033, 0.0008, 0.0003, 0.0002 (время по настенным часам ненадёжно из-за скачков §8); лёрнер `starting on device=cuda` в обоих train.
  - float16: 101 запись `train`, неконечных значений нет; `skipped_updates` ненулевой в 1 записи (значение 1); `critic_warmup == 1` только на шагах 1, 11, 21; `kickstart_lambda` не возрастает, 1.0 → 0.9806 на шаге 1001; финальный чекпоинт `ckpt_v1002`; eval: trained против random W/D/L 100/0/0, **win 1.000**; против greedy 0/100/0, **score 0.500**.
  - bfloat16: 106 записей `train`, неконечных значений нет; `skipped_updates` 0; `critic_warmup == 1` только на шагах 1, 11, 21; λ 1.0 → 0.9796 на шаге 1051, не возрастает; `ckpt_v1054`; eval: против random **win 1.000**, против greedy **score 0.500**.
  - Пороги 0.95 / 0.40 пройдены; результат тот же, что до изменения (§4). В корне репозитория ничего не осталось.

### Остальные вопросы

- Q3 записан на далёкое будущее: проверить, нужна ли на самом деле ручка TF32 у лёрнера.
- Q2 — вариант 3, реализация в SP4 (сейчас ничего не меняется). Предложенные варианты: 1 — оставить и задокументировать; 2 — допуск `ρ > ρ̄·(1 + ε)`; 3 — обрезанная масса весов; 4 — убрать обе метрики. Выбран 3: в SP4 заменить `rho_clip_frac` / `c_clip_frac` на обрезанную массу весов важности по ACT-слотам, `rho_trunc_mass = Σ max(ρ − ρ̄, 0) / Σ ρ` и `c_trunc_mass = Σ max(ρ − c̄, 0) / Σ ρ`: ровно 0 on-policy и растёт с тем, сколько V-trace действительно обрезает. Почему: доли читаются 0.41–0.52 (медиана) во всех настоящих прогонах здесь — unit_harvest с лёрнером на CUDA 0.41 при `log_rho_abs_mean` 0.001, unit_harvest с лёрнером на CPU 0.42 при 0.0001, крестики-нолики на CUDA 0.52 при 0.13, — поэтому почти on-policy от off-policy по ним не отличить.
- Q4 — ветка принята владельцем 2026-10-15 и слита в `main` через `git merge --no-ff` (слияние — следующий коммит после последнего коммита ветки).

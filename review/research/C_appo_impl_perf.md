# R3. APPO / IMPALA: как правильно реализовывать и ускорять (справочник для ревью Colosseum)

Дата исследования: 2026-10-07. Все ссылки на код — на реальные клоны репозиториев, прочитанные локально
(Sample Factory @ 2026-10-01, Ray/RLlib master, scalable_agent, rlax, CleanRL, PufferLib 3.0 и master, PettingZoo, OpenSpiel, RLCard, TorchRL, SB3).
Цифры бенчмарков, помеченные «(измерено)», получены мной на AMD Ryzen 7 4700U (8 потоков) при loadavg 18–22 (машина была занята
другими процессами), поэтому абсолютные значения завышены, а относительные соотношения надёжнее. Утверждения, которых я не проверял
по первоисточнику, помечены «(не проверено)».

---

## 0. Резюме (главное)

1. **Нет единого «эталонного APPO».** Три эталона считают три разные вещи:
   - **Sample Factory (SF)**: ratio `π_θ/μ` считается относительно *behavior-политики из воркера* (хранимый `log_prob_actions`), V-trace **опционален** (`--with_vtrace=False` по умолчанию, по умолчанию GAE λ=0.95), при включении V-trace пересчитывается **на каждом минибатче** по текущим θ.
   - **RLlib APPO (IMPACT)**: ratio и V-trace считаются относительно **target-сети лёрнера** (медленно обновляемой копии), а behavior-политика входит только через клипнутый IS-вес `μ/π_target ∈ [0,2]`. На каждый батч — `circular_buffer_iterations_per_batch=2` прогона.
   - **IMPALA (DeepMind)**: один градиентный шаг на батч, target = текущая θ, bootstrap `V(x_T)` пересчитывается лёрнером, без PPO-клипа.
   Конструкция «PPO-clip по ratio к behavior + V-trace-advantage по θ + пересчёт на каждом минибатче» (как у Colosseum) — это ровно SF с `with_vtrace=True`; она допустима, но это не «каноническая» формула, а вариант SF.
2. **Advantage в APPO = V-trace `pg_advantage` = `ρ_t (r_t + γ v_{t+1} − V(x_t))`**, где `v` — V-trace-цели, а не GAE поверх V-trace. «GAE-подобный» параметр λ входит как `c_t = λ·min(c̄, ρ_t)` (IMPALA Remark 2; rlax `lambda_`; PufferLib `puff_advantage`). Andrychowicz: λ=0.9 хорошо работает и для GAE, и для V-trace; между GAE и V-trace значимой разницы нет.
3. **Конец эпизода в V-trace**: нужно *два* флага — `terminated` (обнуляет bootstrap γ·V(s')) и `done = terminated|truncated` (обрывает распространение следа `c`); при truncation bootstrap берётся от `V(final_obs)`, а не от `V(s_{t+1})` следующего (auto-reset) эпизода. TorchRL и SB3 так и делают; SF приближает `V(s_t)`; IMPALA/CleanRL/RLlib-legacy truncation не различают.
4. **Рекуррентность**: «stored state» (h_init на начале чанка) + сброс скрытого состояния *внутри* последовательности по `done` — обязательны (SF, CleanRL, IMPALA делают это). R2D2 показывает, что stored-state страдает от representational drift, лечится burn-in (первые 40 из 80 шагов без лосса). Для APPO со staleness 0–несколько версий достаточно stored state; burn-in — nice-to-have для длинных чанков/большого lag.
5. **Must-have детали** (по Andrychowicz/Engstrom/Huang): нормализация наблюдений (Andrychowicz: «crucial»), ортогональная инициализация с масштабом 0.01 на policy-голове, global grad-norm clip, нормализация advantage (по минибатчу; влияние невелико), корректный γ. **Value clipping — не включать** (Andrychowicz: вредит независимо от порога; Engstrom: нет доказательств пользы). Value/return normalization — сильно зависит от задачи (MAPPO: «never hurts»; Andrychowicz: критично на HalfCheetah/Humanoid, вредит на Walker2d).
6. **Нормализация в распределённой системе**: безопасный паттерн SF — статистики живут как `buffers` внутри модели, **единственный писатель — лёрнер**, воркеры получают их вместе с весами; в чанках хранятся *сырые* наблюдения. Альтернатива RLlib — локальные фильтры + периодический `merge_states` (параллельная формула Welford/Chan) + broadcast. В self-play пуле у каждого замороженного оппонента должны быть **свои** сохранённые статистики.
7. **Пошаговые игры**: правильная модель — траектория *на игрока* из его собственных точек принятия решений; награда между двумя его ходами накапливается (PettingZoo AEC `_cumulative_rewards`); финальная награда **обязательно** попадает в последний переход каждого игрока, включая того, кто ходил не последним (RLCard `reorganize`, OpenSpiel «step all agents with final info state»). Не ходящий игрок не генерирует переход (но его последний переход закрывается на terminal).
8. **Маски действий**: логиты `masked_fill` большим отрицательным (−1e8…−1e9, а не `−inf`, чтобы `0·(−inf)` не давал NaN в энтропии), маска хранится в чанке и применяется при пересчёте log-prob/энтропии/KL; behavior log-prob должен быть *маскированным*; при AMP маскирование и log_softmax — в fp32. Для составных действий — маска зависимых голов (OpenAI Five: «mask out the ignored ones since their gradients would be pure noise»).
9. **Производительность**: измеренные факторы — (a) `torch.save`/`torch.load` на чанке ≈271 KiB в 13–36 раз медленнее pickle-5/raw (380/690 мкс против 30/19 мкс); (b) на float32-случайных данных lz4/zstd бесполезны (ratio 1.04–1.13), на разреженных игровых планах lz4 даёт 13×, zstd-1 — 26×; (c) на CPU для маленькой MLP numpy-forward в ~2.5× быстрее torch при B=1 и в ~1.3× при B=8–128, `torch.compile` на CPU для крошечной MLP **медленнее** eager; батчинг по средам даёт ~10× на сэмпл (B=1 → B=32); (d) Python-цикл V-trace T=256,B=64 ≈2.9 мс, numpy ≈0.73 мс, `torch.compile` ≈0.39 мс (компиляция 35 с); (e) `tensor.pin_memory()` на каждый батч **медленнее**, чем обычная передача (PyTorch tutorial) — нужны заранее выделенные pinned-буферы.
10. Утверждение из `CLAUDE.md` «Cleanba paper confirms: IMPALA doesn't lose data efficiency from async, PPO does» **слишком сильное**: авторы Cleanba сами пишут, что причина «may be» в 16 градиентных шагах на роллаут у PPO против 4 у IMPALA, и что вывод специфичен для их гиперпараметров (раздел 5.3, 6).

---

## 1. APPO / IMPALA: точные формулировки в эталонах

### 1.1 Сводная таблица эталонов

| Пункт | IMPALA (DeepMind `scalable_agent`) | Sample Factory (APPO) | RLlib APPO (IMPACT, новый стек) | PufferLib (3.0 / master) | Colosseum (быстрое чтение) |
|---|---|---|---|---|---|
| Ratio для PPO-clip | нет PPO-clip | `exp(logπ_θ − logμ)`, μ = log-prob из воркера; `clamp(0.05, 20)` | `π_θ/π_target` (target = старая копия сети лёрнера) | `π_θ/μ` (PPO) | `π_θ/μ`, clamp log-ratio ±20 |
| Диапазон clip | — | асимметричный `[1/(1+ε), 1+ε]`, ε=0.1 | `1±clip_param`, `clip_param=0.4` | `clip_coef=0.2` | `1±eps_clip` |
| Как используется V-trace | цели `vs` для baseline и `pg_advantage` | **опционально** (`with_vtrace=False` по умолч.); если включён — `adv`, `vs` пересчитываются на каждом минибатче по текущим θ | всегда (`vtrace=True`); ρ считается от **target-сети** к μ; цели постоянны на все 2 прогона батча | опционально (`vtrace=0` по умолч.), «GAE with ρ=1»; при `vtrace=1` ρ̄/c̄ клипы внутри `puff_advantage` | всегда, по текущим θ, пересчёт на каждом минибатче (как SF) |
| Advantage | `ρ̄_pg·(r + γ v_{s+1} − V)` | то же (V-trace), затем нормализация по минибатчу | то же (`pg_advantages`), **без** нормализации | GAE-подобный λ=0.90 + клипы ρ/c | то же + нормализация |
| Value target | `vs` (V-trace) | `vs` (V-trace) или GAE-returns | `vs` | `adv + V` | `vs` |
| Value loss | `0.5·Σ(vs−V)²`, без clipping | `max((V−t)², (V_clip−t)²)·coef`, **value clip включён** (`ppo_clip_value=1.0`) | `0.5·mean((V−vs)²)·0.5`, без clipping | PPO-style value clip `vf_clip_coef=0.2` | `mse_loss` (без clipping) — хорошо |
| Несколько проходов по данным | 1 градиентный шаг | `num_epochs`×`num_batches_per_epoch` (по умолч. 1×1) | `circular_buffer_iterations_per_batch=2` | `replay_ratio`, minibatch | `num_epochs` |
| Bootstrap `V(x_T)` | learner-ом по текущим θ (unroll T+1) | learner-ом по текущей сети (`_prepare_batch`) | learner-ом (+1ts трюк, `loss_mask`) | из буфера/последний шаг | **из воркера** (behavior-сеть) — расхождение |
| Max policy lag | нет | `max_policy_lag=1000` версий, старое → `valids=False` | нет явного; target-сеть + IS-клип `[0,2]` | `async`: ставка на 1 эпоху | не найден (см. раздел 6) |
| Grad clip | 40 (global norm, RLlib) / в оригинале через PBT | `max_grad_norm=4.0` | `grad_clip=40.0`, global_norm | `max_grad_norm=1.5` | `max_grad_norm` |
| Оптимизатор | RMSProp lr=4.8e-4, decay .99, eps .1 | Adam lr=1e-4, eps 1e-6 | Adam lr=5e-4 | — | Adam |
| Entropy coef | 0.00025 (**сумма** по времени и батчу) | 0.003 | 0.01 | 0.001 | `entropy_coeff` |
| Reward | clip `[-1,1]` (abs_one) | `reward_scale`, `reward_clip`, + return normalization | — | clamp `[-1,1]` | — |

Источники кода:
- SF: `sample_factory/algo/learning/learner.py` — `_calculate_losses` (L537), `_policy_loss` (L431), `_value_loss` (L441), `_prepare_batch` (L943), `_train` (L671); `sample_factory/cfg/cfg.py`.
- RLlib: `python/ray/rllib/algorithms/appo/torch/appo_torch_learner.py` (`compute_loss_for_module`, L46–237), `.../impala/torch/vtrace_torch_v2.py` (`vtrace_torch`, L73–169), `.../appo/appo.py` (дефолты L107–147), легаси `.../appo/appo_torch_policy.py` (`loss`).
- IMPALA: `scalable_agent/vtrace.py` (`from_importance_weights`), `experiment.py` (`build_learner` L334–415, `unroll` L184–208, `build_actor` L228–).
- PufferLib: `src/algo.cu` (`puff_advantage`, ~L1698), `config/default.ini`; tag 3.0: `pufferlib/pufferl.py`, `pufferlib/vector.py`.

### 1.2 Точные фрагменты

**IMPALA V-trace (`scalable_agent/vtrace.py::from_importance_weights`)**
```python
rhos = tf.exp(log_rhos)                      # π/μ
clipped_rhos = tf.minimum(clip_rho_threshold, rhos)   # ρ̄  (определяет fixed point)
cs = tf.minimum(1.0, rhos)                   # c̄ = 1     (скорость сходимости, variance)
values_t_plus_1 = tf.concat([values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)
# обратный scan:  acc = delta_t + discount_t * c_t * acc
vs = vs_minus_v_xs + values
vs_t_plus_1 = tf.concat([vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
pg_advantages = clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values)
```
Математика (IMPALA, §4.1, Remark 1–3): `v_s = V(x_s) + δ_s V + γ c_s (v_{s+1} − V(x_{s+1}))`; ρ̄ задаёт fixed point (политику π_ρ̄ между μ и π), c̄ — только скорость сходимости/variance; `λ` (Remark 2): `c_i = λ·min(c̄, π/μ)`; при on-policy и n=∞ V-trace → TD(λ).

**Что значит `discounts` в IMPALA.** `discounts = (~done) * γ` (`experiment.py` L372), где `done` — флаг, пришедший в `env_outputs` на шаге t+1 (`t[1:]`). Окружение (`environments.py::FlowEnvironment`) на `done=True` возвращает *первое наблюдение следующего эпизода*, а `reward` — награда терминального перехода. То есть `discounts[t]` относится к переходу `t → t+1` и равен 0 именно на терминальном переходе — это «discount=0 на terminal».

**IMPALA: выравнивание T+1.** Актор отдаёт `T+1` элементов (`first_values` + `T` шагов), `learner_outputs`/`agent_outputs` сдвигаются: `learner_outputs[:-1]` и `agent_outputs[1:]`; `bootstrap_value = learner_outputs.baseline[-1]` (пересчитан **learner-ом**). Последний элемент чанка = первый элемент следующего (overlap 1).

**IMPALA: LSTM** (`unroll`):
```python
for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
    # If the episode ended, the core state should be reset before the next.
    core_state = nest.map_structure(functools.partial(tf.where, d), initial_core_state, core_state)
    core_output, core_state = self._core(input_, core_state)
```
Состояние сбрасывается **перед** обработкой наблюдения, у которого `done=True` (это первое наблюдение нового эпизода). `agent_state` на начале unroll сохраняется актором (`first_agent_state`) — это stored state.

**RLlib APPO (новый стек), существенное** (`appo_torch_learner.py`):
```python
old_target_policy_actions_logp = old_target_policy_dist.logp(batch[ACTIONS])   # target-сеть, forward_target
behaviour_actions_logp = batch[ACTION_LOGP]
...
discounts_time_major = (1.0 - terminateds) * config.gamma * loss_mask_time_major   # terminated, НЕ truncated
vtrace_adjusted_target_values, pg_advantages = vtrace_torch(
    target_action_log_probs=old_actions_logp_time_major,     # <-- от target-сети, не от текущей θ
    behaviour_action_log_probs=behaviour_actions_logp_time_major, ...,
    clip_pg_rho_threshold=config.vtrace_clip_pg_rho_threshold, clip_rho_threshold=config.vtrace_clip_rho_threshold)
is_ratio = torch.clip(torch.exp(behaviour_actions_logp - old_actions_logp), 0.0, 2.0)   # μ/π_target
logp_ratio = is_ratio * torch.exp(target_actions_logp - behaviour_actions_logp)        # = π_θ/π_target
surrogate_loss = torch.minimum(pg_advantages * logp_ratio,
                               pg_advantages * torch.clip(logp_ratio, 1 - clip_param, 1 + clip_param))
delta = values_time_major - vtrace_adjusted_target_values
vf_loss = 0.5 * torch.sum(torch.pow(delta, 2.0) * loss_mask_time_major)
```
Выводы: (i) ratio и V-trace ρ — относительно **target-сети** (обновляется раз в `target_network_update_freq(2) × circular_buffer_num_batches(8) × iterations_per_batch(2) × train_batch_size` шагов, `tau=1.0` — жёсткая копия); (ii) behavior-политика входит через IS-вес `[0,2]` (`target_worker_clipping=2.0`); (iii) truncation **не** обнуляет bootstrap: `discounts` используют `terminateds`, а на границе чанка применяется трюк «+1 timestep» (`add_one_ts_to_episodes_and_truncate`: к каждому чанку добавляется фиктивный шаг для forward `V(s_{T})`, шаг исключён из лосса `loss_mask=False`); (iv) в IMPALA-части RLlib комментарий признаёт, что цикл V-trace «раньше считали на CPU», а сейчас «быстрее оставить на GPU» — единого мнения нет (см. 5.3).

**Sample Factory** (`learner.py::_calculate_losses`, V-trace-ветка, сокращённо):
```python
ratio = torch.exp(log_prob_actions - mb.log_prob_actions)   # π_θ / μ
ratio = torch.clamp(ratio, 0.05, 20.0)
vtrace_rho = torch.min(rho_hat, ratio); vtrace_c = torch.min(c_hat, ratio)
next_values = values[recurrence-1::recurrence] - rewards[recurrence-1::recurrence]; next_values /= gamma
next_vs = next_values
for i in reversed(range(recurrence)):
    not_done_gamma = (1.0 - dones[i::recurrence]) * gamma
    delta_s = vtrace_rho[i] * (rewards[i] + not_done_gamma * next_values - values[i])
    adv[i]  = vtrace_rho[i] * (rewards[i] + not_done_gamma * next_vs     - values[i])
    next_vs = values[i] + delta_s + not_done_gamma * vtrace_c[i] * (next_vs - next_values)
    vs[i] = next_vs;  next_values = values[i]
...
adv = (adv - adv_mean) / clamp_min(adv_std, 1e-7)      # по минибатчу, только по valids
policy_loss = -mean(min(ratio*adv, clamp(ratio, 1/(1+ε), 1+ε)*adv))
```
Замечания (по моему чтению кода): (a) инициализация `next_values = (V_last − r_last)/γ` делает `δ=0` на последнем шаге роллаута при `done=0` — т.е. **bootstrap-значение `V(s_T)`, вычисленное в `_prepare_batch`, в V-trace-ветке не используется**, а последний шаг каждого роллаута получает нулевой advantage; (b) `with_vtrace` требует `recurrence == rollout` (assert в `_train`); (c) выбор `ρ̄ = c̄ = 1`, `max_grad_norm=4`.

**SF: value bootstrapping по таймауту** (`_prepare_batch`, по умолчанию выключено `value_bootstrap=False`):
```python
buff["rewards"].add_(self.cfg.gamma * denormalized_values[:, :-1] * buff["time_outs"] * buff["dones"])
```
Здесь `V(s_{t+1})` приближено `V(s_t)` (в коде прямо сказано: «we don't have obs(t+1)»).

### 1.3 Конец эпизода внутри чанка, bootstrap и truncation (как должно быть)

Обозначим для шага t: `term_t` — терминальное состояние достигнуто после `a_t`; `trunc_t` — обрыв по лимиту времени; `done_t = term_t ∨ trunc_t`.

```
V_next_t = V(s_{t+1})            если t < T-1 и эпизод не закончился
         = V(s_T)                 на конце чанка (считать лёрнером по ТЕКУЩЕЙ сети, как IMPALA/SF)
         = 0                      если term_t
         = V(final_obs_t)         если trunc_t   (а не V следующего эпизода!)
δ_t       = ρ_t · ( r_t + γ·(1 − term_t)·V_next_t − V(s_t) )
(v_t − V(s_t)) = δ_t + γ·(1 − done_t)·c_t·(v_{t+1} − V(s_{t+1}))      # след обрывается на ЛЮБОМ done
A_t^pg   = ρ̄_pg,t · ( r_t + γ·(1 − term_t)·v_next_t − V(s_t) ),   v_next_t = v_{t+1} внутри эпизода, иначе V_next_t
ρ_t = min(ρ̄, π/μ),  c_t = λ·min(c̄, π/μ)
```
Эталоны:
- **TorchRL** (`torchrl/objectives/value/functional.py::vtrace_advantage_estimate`): отдельные `done` и `terminated`; `terminated_discounts = gamma * not_terminated; deltas = clipped_rho * (reward + terminated_discounts * next_state_value - state_value)`; `next_state_value` передаётся *отдельным* тензором (в нём можно положить `V(final_obs)`).
- **rlax** (`rlax/_src/vtrace.py::vtrace`): `v_tm1`, `v_t` — отдельные массивы, `discount_t` — произвольный, `lambda_`, `sample_mask`. Реализация scan: `acc = learnable*(td_error + discount*c*acc) + (1-learnable)*acc` — шаги с `sample_mask=0` пропускаются без дисконтирования (полезно для пошаговых игр, см. §3).
- **SB3** (`on_policy_algorithm.py::collect_rollouts`): «Handle timeout by bootstrapping with value function»: `rewards[idx] += self.gamma * terminal_value`, где `terminal_value = V(infos[idx]["terminal_observation"])` при `TimeLimit.truncated`.
- Andrychowicz §3.6: при очень большом лимите шагов (1000) специальная обработка таймаутов **не влияет**; для коротких лимитов (игры с ограничением ходов) — влияет, это не опровергается их результатом. Pardo et al., «Time Limits in RL» (arXiv 1712.00378) — первоисточник проблемы.

Практическое правило для Colosseum: хранить в чанке `terminated[t]`, `truncated[t]` и `bootstrap_at_trunc[t] = V_behavior(final_obs)` (или `final_obs` для пересчёта), `bootstrap_value` на конце чанка пересчитывать лёрнером (хранить `obs_T`, а не `V(s_T)`); тогда одна формула выше покрывает все случаи.

### 1.4 max_policy_lag, много эпох на одних данных, value clipping, нормализация

- **max_policy_lag.** SF: `valids = (policy_id == self.policy_id) & (train_step − policy_version < max_policy_lag)`; невалидные шаги маскируются в лоссе и advantage, `actions[invalid]=0`, `log_prob_actions[invalid]=-1`, **LR умножается на долю валидных данных** (иначе шумные градиенты), предупреждение при >50% невалидных. Дефолт 1000 версий — по сути «выключено»; документация SF (`docs/07-advanced-topics/policy-lag.md`): «policy lag < 20–30 SGD steps is usually fine»; LSTM/GRU и сложные (Tuple/MultiDiscrete) пространства действий чувствительнее. OpenAI Five (§ «Staleness»): целились в staleness 0–1; рост на ~8 версий даёт значимое замедление. RLlib: явного фильтра нет, защита — target-сеть и клип IS-весов.
- **Мера lag**: SF формула `Lag ~ num_epochs·num_workers·num_envs_per_worker·agents·rollout / batch_size`; метрики `version_diff_{min,avg,max}` — стоит логировать и у нас (`policy_version` лёрнера минус версия чанка).
- **Несколько эпох с V-trace.** Две допустимые схемы: (A, SF) пересчитывать V-trace каждый минибатч по текущим θ (ratio к μ клипится PPO-клипом, `ρ̄ = 1`); (B, RLlib) зафиксировать target-сеть, цели `vs` постоянны, меняется только surrogate. Андрыхович (§3.5): устаревшие advantage вредят, лучше **пересчитывать advantage в начале каждого прохода**; перемешивать отдельные переходы, а не целые чанки — но для рекуррентных политик и V-trace перемешиваются целые последовательности (SF: `_get_minibatches` берёт индексы стартов сегментов длиной `recurrence`).
- **Value clipping.** Не включать (Andrychowicz C13: «hurts the performance regardless of the clipping threshold»; Engstrom: нет подтверждений пользы; Huang «37 details», #9). Если используется (как SF по умолчанию), порог `ppo_clip_value=1.0` осмыслен только при нормализованных целях.
- **Нормализация advantage**: SF — по минибатчу и только по `valids`; RLlib/IMPALA — нет; Andrychowicz C67 — «does not affect the performance too much».
- **Entropy/LR/clip**: Five — entropy 0.01→0.001, LR 5e-5→5e-6 линейно, PPO clip 0.2, GAE λ 0.95, value loss weight 1.0 после нормализации ревордов по бегущей std; SF — entropy 0.003, LR 1e-4 константа, clip 0.1.

### 1.5 Рекуррентные политики

| Аспект | SF | CleanRL `ppo_atari_lstm.py` | IMPALA | R2D2 | RLlib | PufferLib |
|---|---|---|---|---|---|---|
| Что хранится | `rnn_states` на **каждом** шаге роллаута (+ `T+1`) | `initial_lstm_state` на начало роллаута | `agent_state` на начало unroll | stored state на начало последовательности | `state_in` на начало чанка | 3.0: **ноль** на каждом горизонте (`lstm_h.zero_()` в `evaluate`); master: `reset_every_horizon=0` = carry, `async` через `initial_states` |
| BPTT | сегменты длиной `recurrence` (≤ rollout); `rollout % recurrence == 0`; V-trace требует `recurrence==rollout` | весь роллаут (128 шагов), минибатч по средам | весь unroll | `m=80`, перекрытие 40, burn-in 40 | `max_seq_len` + zero-padding и `loss_mask` | `bptt_horizon` |
| Сброс внутри последовательности | `PackedSequence`: последовательность режется на эпизоды, `rnn_states * is_same_episode`; в `done_or_invalid` попадают и данные чужой политики | `(1.0 - d) * lstm_state` **перед** шагом, где `d = done[t]` помечает «наблюдение t — первое нового эпизода» | `tf.where(done, initial, state)` перед шагом | эпизоды не пересекаются | эпизод = отдельная строка батча | зануление при terminal |
| Gradient через границу эпизода | нет | нет (state обнулён) | нет | — | нет | нет |

Ключевые факты:
- R2D2: «stored state» страдает от representational drift/recurrent state staleness; «zero start state» хуже; лучше всего stored state + burn-in (первые 40 шагов 80-шаговой последовательности дают только start state). (Первоисточник: Kapturowski et al., ICLR 2019, openreview.net/forum?id=r1lyTjAqYX; прочитано через вторичные пересказы и Acme, оригинальный PDF недоступен — цифры «80/40/40» — не проверено по оригиналу.)
- OpenAI Five (§Table 2): роллауты по 256 шагов, обучение «sample = unrolled LSTM of 16 frames», т.е. скрытое состояние хранится на границах 16-шаговых сегментов — truncated BPTT с stored state. Это вариант между «весь чанк» и «burn-in».
- SF-документация: для RNN политик «not only the action distributions, but also the hidden states change between the behavior and target policies» — lag чувствительнее.
- **Сброс скрытого состояния — до обработки первого наблюдения нового эпизода**; эквивалентно «после шага t при `done_t`», если `done_t` относится к *переходу*. Главное — чтобы воркер и лёрнер использовали **одну и ту же** конвенцию и один и тот же сдвиг (см. §6).
- Для LSTM сбрасывать и `h`, и `c`, по всем слоям; для GRU — `h`.
- Скорость: цикл по T шагов с `nn.LSTM` по одному шагу — это T запусков cuDNN-ядра; SF решает через `PackedSequence`, разрезая роллаут на эпизодные сегменты (один вызов `forward_core`). Альтернатива: вызывать cuDNN-LSTM на непрерывных отрезках между `done` и зануления стартов.

---

## 2. «Детали реализации», которые реально важны

Источники: Huang et al. «The 37 Implementation Details of PPO» (ICLR Blog Track 2022); Andrychowicz et al. 2020 «What Matters In On-Policy RL» (arXiv 2006.05990, 250 000 экспериментов, 5 задач непрерывного управления); Engstrom et al. 2020 «Implementation Matters in Deep Policy Gradients» (arXiv 2005.12729); OpenAI Five (arXiv 1912.06680); MAPPO (arXiv 2103.01955); PopArt (arXiv 1602.07714, 1809.04474).

| Деталь | Статус | Что показывают источники | Как делать у нас |
|---|---|---|---|
| Нормализация наблюдений (running mean/std, clip ±5…10) | **must-have** | Andrychowicz C64: «crucial … on all environments apart from Hopper»; Five: running mean/std «of all data ever observed», clip (−5,5); SF: `RunningMeanStdInPlace`, eps 1e-5, clip 5.0, float64 аккумуляторы | Per-agent; единственный писатель — лёрнер; хранить в чанках сырые obs (или версию статистик); сохранять в чекпойнт и в замороженных оппонентов (см. 2.1) |
| Ортогональная инициализация | **must-have** | Engstrom #3, Huang #2: hidden √2, policy 0.01, value 1; Andrychowicz C57: последний слой policy в 100× меньше «boosts Humanoid by 66%» | `std=0.01` для policy-головы (особенно CompositeDist), `1.0` для value |
| Global grad-norm clip | **must-have** | Engstrom #9 (0.5), SF 4.0, IMPALA/RLlib 40. Andrychowicz C68: «small boost, threshold barely matters» | Порог зависит от редукции лосса (sum vs mean); начать с 0.5–4 при `mean`-редукции, логировать норму до клипа |
| Нормализация advantage | must-have (дёшево) | Huang #7: по минибатчу; Andrychowicz C67: слабое влияние; RLlib/IMPALA не делают | По минибатчу, по `valid`-маске |
| Adam eps | nice-to-have | Huang #3: 1e-5; SF: 1e-6; Andrychowicz: lr 3e-4 как «safe default», β1=0.9 | `eps=1e-5…1e-6` |
| LR annealing | nice-to-have | Engstrom #4 и Huang #4 — помогает; Andrychowicz: «of secondary importance» | Линейный к 0 по числу **SGD-шагов или env-шагов** (одна единица, см. §6) |
| γ | **tune** | Andrychowicz C20: «one of the most important hyperparameters», старт 0.99 | Для игр с терминальной наградой — 0.997–1.0 (+ GAE/V-trace λ 0.9–0.95) |
| λ (GAE / V-trace) | must-have параметр | Andrychowicz: λ=0.9 хорош и для GAE, и для V-trace (C8/C9); SF-help: низкий `c̄` ≈ λ<1; Puffer: 0.90 | Добавить `c_t = λ·min(c̄,ρ)` в V-trace (сейчас в Colosseum λ нет) |
| Value clipping | **не включать** | Andrychowicz C13 «hurts regardless»; Engstrom: нет доказательств | выключено (уже так) |
| Huber vs MSE для value | MSE | Andrychowicz C11: Huber хуже | MSE |
| Value/return normalization | **nice-to-have → must-have для разных шкал** | MAPPO §5.1: «never hurts … often improves»; Andrychowicz C66: критично на 2 задачах, вредит на Walker2d; SF `normalize_returns=True`; Five: reward по бегущей std, value-loss weight после нормализации | Простой running-std нормализатор return-целей (SF) per-agent; PopArt — если несколько шкал/голов (Hessel 2018) |
| Reward scaling/clipping | nice-to-have | Engstrom #2/#5: масштабирование на std rolling discounted sum — важно; IMPALA: clip `[-1,1]`; Five: running std | Для zero-sum ±1 не нужно; при shaped-наградах — return normalization |
| Раздельные policy/value сети | рекомендация | Andrychowicz C47: раздельные лучше на 4 из 5 задач; value-сеть можно шире | Опционально |
| Батч/число сред | настройка | Andrychowicz C1/C4: больше сред → короче чанки → хуже sample complexity; больший батч не вредит | T=128–512 — ок |
| Число проходов по данным | настройка | Andrychowicz: «crucial»; Cleanba: PPO с 16 SGD-шагами страдает от async сильнее IMPALA с 4 | `num_epochs`·`minibatches` ≈ 4–8, смотреть KL/clipfrac |
| Регуляризаторы (entropy/KL) | nice-to-have | Andrychowicz §3.8: ни один не помогает значимо (кроме HalfCheetah) | entropy малый; kickstart-KL — отдельная ручка |
| Tanh активации | nice-to-have | Andrychowicz C-act: tanh лучше, relu хуже; Engstrom #8 | для маленьких MLP |
| Начальный std действий (непрерывные) | must-have для Box | Andrychowicz: 0.5 лучше всего; softplus + offset | см. §4 |

### 2.1 Синхронизация running stats в распределённой системе

Три эталонных подхода:
1. **SF**: статистика — буферы `nn.Module` (`RunningMeanStdInPlace`: `running_mean`, `running_var`, `count`, float64), обновляются **лёрнером** в `_prepare_and_normalize_obs` под `policy_lock` (при `training=True`), воркеры получают их вместе с `state_dict`. Нормализация на воркере/policy-worker — по последним доступным статистикам; в роллаут-буфер кладутся **сырые** наблюдения (`buff["obs"]`), лёрнер нормализует их заново текущими статистиками (небольшое расхождение train/behavior принимается). Return-статистики (`returns_normalizer`) только на лёрнере; value-головa предсказывает нормализованные значения; для GAE/bootstrap значения денормализуются.
2. **RLlib `MeanStdFilter`** (`connectors/env_to_module/mean_std_filter.py`): каждый env-runner обновляет локальный `RunningStat`; алгоритм периодически собирает `get_state()`, объединяет `merge_states()` (параллельное слияние моментов) и рассылает `set_state()`.
3. **OpenAI Five**: бегущие среднее и std «по всем данным за всё время», clip (−5,5).

Формула слияния (SF `_update_mean_var_count_from_moments`, Chan et al.):
```
δ = μ_b − μ;  n = n_a + n_b
μ_new = μ + δ·n_b/n
M2 = var_a·n_a + var_b·n_b + δ²·n_a·n_b/n;   var_new = M2/n
```
Правила для Colosseum: (a) один писатель на агента (лёрнер) либо merge по формуле — не усреднять средние без весов; (b) статистики версионировать вместе с весами (`WeightPayload`) и класть в чекпойнт; (c) чекпойнты в пуле оппонентов и `eval` используют **свои** сохранённые статистики; (d) не обновлять статистики на eval/замороженных оппонентах; (e) для нескольких агентов с разными энкодерами — отдельные нормализаторы; (f) clip после нормализации, `eps` под корнем, float64-аккумуляторы.

**PopArt** (van Hasselt et al. 2016): `μ_t=(1−β)μ_{t−1}+βY`, `ν_t=(1−β)ν_{t−1}+βY²`, `σ_t=√(ν_t−μ_t²)`; при смене статистик последний линейный слой корректируется, чтобы **сохранить выход**: `W_new=(σ/σ_new)W`, `b_new=(σb+μ−μ_new)/σ_new`. Hessel et al. 2018 применили PopArt в IMPALA для мульти-задачи с разными шкалами. В нашем случае (каждый обучаемый агент — свой лёрнер) обычной нормализации достаточно; PopArt нужен, если одна value-голова обслуживает несколько шкал.

---

## 3. Многоагентные траектории в играх

### 3.1 Пошаговые игры (один игрок за шаг)

Что делают эталоны:
- **PettingZoo AEC** (`pettingzoo/utils/env.py`): `last()` возвращает `_cumulative_rewards[agent]` — **награду, накопленную с момента последнего хода этого агента**; `step()` вызывает `_accumulate_rewards()`; умерший агент получает ещё один «dead step» (`_was_dead_step`), чтобы потребитель увидел его финальное наблюдение/награду/termination. Из статьи (Appendix C.1): «At every step, every agent j receives the partial reward r′».
- **OpenSpiel** (`rl_environment.py::get_time_step`): `TimeStep.rewards` — список по **всем** игрокам на каждом шаге; `discounts=0` на терминале; в `examples/tic_tac_toe_qlearner.py`: после конца эпизода «`# Episode is over, step all agents with final info state.` — `for agent in agents: agent.step(time_step)`»; в `pytorch/dqn.py::step` переход `(prev_timestep, prev_action, time_step)` формируется на **следующем** вызове step() этого агента (в т.ч. на терминальном), а действие выбирается только если `self.player_id == time_step.current_player()`.
- **RLCard** (`rlcard/utils/utils.py::reorganize`): траектории по игрокам `[s, a, s, a, …, s_final]`; награда `payoffs[player]` и `done=True` ставятся на **последний переход каждого игрока**, независимо от того, кто сделал последний ход; `Env.run` добавляет терминальное состояние **всем** игрокам.

Вывод/правило:
1. Не ходящий в данный момент игрок **не генерирует переход** (нет inference, нет записи) — как в OpenSpiel/RLCard. Время для V-trace/GAE — индекс **собственных решений** игрока (semi-MDP; γ применяется «за собственный ход»).
2. Награда перехода игрока p = сумма всех наград p между его действием и его следующим решением (AEC-семантика), плюс на терминале — финальная награда. **Проигравший, который не делал последний ход, получает `reward=-1, done=True` на своём последнем (предыдущем) переходе**, а `V_next=0`. Этот «закрывающий» переход нужно сформировать для *каждого* слота, собирающего траектории, в момент конца игры — это самая частая ошибка (см. §6).
3. Наблюдение следующего перехода игрока p — это наблюдение на его **следующем** ходу (после ответа соперника), а не сразу после его хода.
4. LSTM: состояние обновляется только на собственных ходах (то, что соперник сделал, видно из следующего наблюдения), воркер и лёрнер обязаны использовать одну и ту же схему. Альтернатива (общая для одновременных и пошаговых режимов): подавать каждый env-шаг, но считать лосс только на «ходящих» шагах — как `sample_mask` в rlax (пропуск без дисконтирования), `valids`/`inactive agents` в SF (`docs/07-advanced-topics/inactive-agents.md`: невалидные шаги маскируются, LR масштабируется долей валидных, «не более 50%»), `loss_mask` в RLlib. Дороже по вычислениям, но проще для смешанных режимов.
5. Self-play с историческими оппонентами: переходы собираются **только** у слотов с `collect_mask`; ходы оппонента — часть «среды» с точки зрения обучаемого. Если игра заканчивается ходом оппонента, терминальный переход обучаемого всё равно должен быть записан (см. п.2). Для pooled-оппонентов считать их inference в `eval`-режиме, не накапливая в статистики.
6. Value-голова должна получать наблюдение **с точки зрения ходящего игрока** (эгоцентричное, перспектива «я/соперник»), награды — в тех же координатах; случайная рассадка по местам при N>2 (round-robin) — иначе value смешивает места.

### 3.2 FFA / N-player

- Zero-sum rank-based: ввести `r_p = f(rank_p)` с нулевой суммой, напр. `(N−1−2·rank_p)/(N−1) ∈ [−1,1]` или `score_p − mean(score)`; ничьи — средний ранг. Награды выдавать в момент выбывания игрока (его траектория закрывается `done=True`, дальше он «inactive»), а победителю — на конце игры. OpenAI Five: «symmetrize rewards by subtracting the reward earned by the opposing team» и затухание нефинальных наград по времени игры `ρ ← ρ·0.6^{T/10 min}` (аппендикс Reward, чтобы ранние шаги не терялись).
- Для совместных (team) игр — MAPPO (arXiv 2103.01955): критик получает «агент-специфичное» глобальное состояние; value normalization «never hurts». OpenAI Five: Team Spirit (α 0.3→0.8→1.0) смешивает индивидуальную и командную награду.
- Сложность одной общей value-головы на N мест: добавлять вход «seat/relative position» либо отдельные головы на место.

### 3.3 Маскирование действий

Эталоны и правила:
- Huang & Ontañón 2020 (arXiv 2006.14171): маска = замена логитов недопустимых действий на `M=−1e8`; это **корректный policy gradient** для политики `π'=softmax(mask(l))` (маска — дифференцируемая функция логитов; градиент по замаскированным логитам = 0); сохранение маски при обучении принципиально; снятие маски после обучения вредно.
- SF (`algo/utils/action_distributions.py`): `logits + (mask==0)*-1e9`, `probs *= mask`, энтропия `-(log_probs*probs).sum(-1)` — считается по **маскированному** распределению; `action_mask` приходит в `obs`, т.е. лёрнер получает её из того же буфера и пересчитывает логпробы той же маской. Док `action-masking.md`.
- OpenAI Five: «depending on the primary action, some [parameter heads] are read and others ignored (when optimizing, we mask out the ignored ones since their gradients would be pure noise)».
- Правила:
  1. Храните маску **в чанке** (bool/bit-packed), пересчитывайте `log_prob`, `entropy`, `KL` (в т.ч. kickstart и `old_policy`) **той же** маской. Behavior `log_prob`, записанный воркером, обязан быть логпробом из **маскированного** распределения, иначе в 1-й эпохе ratio≠1.
  2. `−inf` ломает энтропию (`0·(−inf)=NaN`); использовать `−1e8…−1e9` или `finfo.min`, либо считать `where(mask, p*logp, 0)`. В AMP fp16 `masked_fill(−1e9)` переполняется (max 65504): логиты/log_softmax приводить к fp32.
  3. Если в строке нет ни одного допустимого действия (паддинг, терминал) — исключить строку из лосса (`valid`-маска), иначе энтропия/KL будут считаться по равномерному распределению.
  4. Для мульти-голов: маска зависимых голов (см. §4.3), а энтропия/KL неиспользуемых голов не учитывается.
  5. Энтропийный коэффициент: максимум энтропии маскированной политики `log n_legal` меняется от состояния к состоянию — это нормально; следить, чтобы бонус не «растягивал» вероятность на допустимые, но заведомо плохие ходы (малый коэффициент).
  6. KL к учителю (kickstarting) считать на маскированных распределениях обеих сторон; учитель, не знающий маски, должен быть ренормализован по допустимым действиям.

---

## 4. Непрерывные и составные действия

### 4.1 tanh-squash против clip

- **Clip в среде (CleanRL `ppo_continuous_action.py`, `gym.wrappers.ClipAction`)**: политика сэмплирует `u ~ N(μ,σ)` (неограниченное), log-prob считается для **`u`**, буфер хранит `u`; клиппинг выполняется только на входе в среду. ratio согласован. Минус: на границах градиент «не видит» клиппинг (смещение), энтропия не ограничена.
- **Squash `a=tanh(u)`** (SAC, arXiv 1812.05905, Приложение C): `log π(a|s) = log N(u; μ,σ) − Σ_i log(1 − tanh²(u_i))` (+ε для численной стабильности); хранить нужно **`u`** (или `a` и `atanh(clip(a))` — менее устойчиво). Энтропии в замкнутой форме нет — оценивать `−E[log π]`. PPO-ratio около границ нестабилен (множитель `1−tanh²`).
- Andrychowicz C63: tanh «slightly better overall (HalfCheetah +30%)», но авторы отмечают, что разница, вероятно, объясняется меньшей начальной амплитудой действий; рекомендации: последний слой policy ×0.01, softplus для std с отрицательным offset (начальный std ≈0.5), tanh как активация и как преобразование семпла.
- SF: неограниченное Гауссово, tanh применяется к **средним** (`continuous_tanh_scale`), `stddev ∈ [1e-4, 1e4]`; клиппинг действий — на стороне среды.
- Практика для Colosseum: хранить в чанке сырой семпл (`u`), в среду отдавать `clip`/`tanh`-преобразованное, ratio и KL считать по `u`; при `tanh` добавить якобиан; масштаб `low/high` применять **после** squash вне политики.

### 4.2 CompositeDist (независимые головы)

- Huang #37: для MultiDiscrete PPO рассматривает компоненты как независимые: `log π(a)=Σ_h log π_h(a_h)`; энтропия = сумма энтропий голов; KL = сумма KL.
- Рекомендация: `log_prob`, `entropy`, `kl` суммировать по головам **с маской использования** `m_h(a_type)`; ratio считать по сумме. Когда голов много, малое изменение политики даёт большое изменение `log π(a)` (SF policy-lag doc: «with complex action spaces small changes … can cause large changes to probabilities of individual actions») → PPO-клип срабатывает чаще; следить за `clipfrac`, уменьшать LR/epochs или применять клип **по головам** (per-head ratio) — nice-to-have.

### 4.3 Авторегрессивные головы и зависимые маски

- AlphaStar (Vinyals et al., Nature 2019; не пере-проверено по тексту): авторегрессивное декодирование `action_type → delay → queue → selected_units → target`, каждый следующий компонент обусловлен эмбеддингом выбранных предыдущих; `log π(a)=Σ log π(a_k | a_{<k})`; при обучении — teacher forcing по записанным действиям.
- OpenAI Five: «primary action» + параметры (Delay, Unit selection, Offset); неиспользуемые параметры маскируются при оптимизации; невалидные комбинации трактуются как no-op.
- Чек-пункты: (1) `m_h` вычисляется из *записанного* primary action; (2) маски допустимых значений зависимой головы зависят от ранее выбранных частей (например, цели допустимы только для атаки) — их нельзя пересчитать на лёрнере без правил среды, поэтому **хранить в чанке маску на все возможные условия** (например `[num_types, param_dim]`) или функцию, зависящую только от obs; (3) энтропия: `H = H(type) + Σ_type π(type)·H(head|type)`; одно-сэмпловая оценка `H(type)+m_h·H_h` несмещена; (4) KL аналогично; (5) bonus энтропии для «условных» голов считать только на шагах, где голова использована.

---

## 5. Инженерия производительности в Python RL

### 5.1 Инференс на CPU-воркерах

Эталоны:
- SF: `torch.set_num_threads(1)` в воркерах (`algo/sampling/rollout_worker.py:70`, `init_torch_runtime(max_num_threads=1)`), `torch.multiprocessing.set_sharing_strategy("file_system")`, `inference_context`: `torch.inference_mode()` в async-режиме и `torch.no_grad()` в serial (там те же тензоры используются лёрнером); флаг в `cfg.py` (около L389) с `threadpoolctl`, чтобы OpenMP/MKL внутри env не плодили потоки; nice воркерам; CPU affinity.
- **Измерение (мои)**: MLP `256→256→256 → {pi(9), v(1)}`, 1 поток, мкс на вызов:

| B | torch no_grad | torch inference_mode | jit trace+freeze | numpy matmul (BLAS 1 поток) | torch.compile |
|---|---|---|---|---|---|
| 1 | 68 | 63 | 44 | **25** | 319 (eager в том же прогоне 95) |
| 8 | 101 | 104 | 97 | **74** | — |
| 32 | 221 | 189 | 200 | **159** | 565 (eager 449) |
| 128 | 692 | 690 | 628 | **538** | — |

Выводы: `inference_mode` против `no_grad` — разницы в пределах шума; основной вклад — накладные расходы диспетчеризации (~60 мкс на вызов); numpy-вариант выигрывает ~2.5× при B=1 и 1.3–1.4× при B=8–128, **но** это годится только для простых MLP (рекуррентные/сложные энкодеры numpy-вариантом не заменить); `torch.compile` на CPU для крошечной MLP дал замедление (guard/overhead) — не использовать для воркеров без собственного замера; ONNX Runtime я не мерил (в среде нет пакета `onnx`) — **не проверено**; SF имеет документ `exporting-to-onnx.md` (для деплоя, не для ускорения тренировки).
- **Батчинг по средам** — самый большой рычаг: стоимость на сэмпл B=1 ≈ 64 мкс, B=32 ≈ 6 мкс, B=128 ≈ 5.4 мкс (≈10×). VectorEnv с K средами должен делать **один** forward на группу `(agent_id, network_id)`.
- **Double-buffered sampling** (SF paper §3.2, `docs/07-advanced-topics/double-buffered.md`): воркер держит 2·N сред, пока inference считает действия для группы A, шагаются среды группы B; «практически устраняет простой CPU воркеров». В нашей схеме (inference **локально** в процессе воркера) выгода другая: можно только перекрывать `env.step` в subprocess-средах с inference на главном потоке воркера (Puffer: `num_envs/batch_size` — «async recv/send»); при чисто последовательной VectorEnv двойная буферизация ничего не даёт.
- **Векторизация сред**: EnvPool (arXiv 2206.10558, Table 1, FPS Atari / MuJoCo на ноутбуке 12 ядер): for-loop 4 893 / 12 861; Subprocess 15 863 / 36 586; Sample-Factory 28 216 / 62 510; EnvPool sync 37 396 / 66 622; EnvPool async 49 439 / 105 126; на DGX-A100 «Subprocess has extremely poor scalability with an almost flat curve». PufferLib (tag 3.0, `pufferlib/vector.py`): `Multiprocessing`-векторизатор с общими `RawArray` для obs/actions/rewards/terminals/truncateds/masks, `zero_copy=True`, `num_workers`, `batch_size` (async recv/send), `overwork=False` (защита от oversubscription).
- Для составных/маленьких env основной выигрыш — **shared memory буферы + semaphores** (Puffer) или C++ thread-pool (EnvPool), а не `mp.Queue`/`Pipe` с pickle на каждый шаг.

### 5.2 Передача данных: mp.Queue, shared memory, сериализация, сжатие, сеть

- SF paper §3.2: «we do not perform any kind of data serialization … At full throttle, Sample Factory generates and consumes more than 1 GB of data per second, and even the fastest serialization/deserialization mechanism would severely hinder throughput»; предаллоцированные тензоры в shared memory, по FIFO-очередям ходят **только индексы буферов**. Для весов — копирование из shared memory (<1 мс).
- `torch.multiprocessing`: стратегия `file_descriptor` (по умолчанию) передаёт дескрипторы; при низком лимите открытых файлов нужно `file_system` (прямо в документации; SF так и делает) — **«Too many open files»** при десятках чанков в полёте; CUDA-тензоры: отправитель обязан держать оригинал, пока получатель его использует.
- **Измерение (мои)**: чанк 271 KiB (obs `[256,256]` float32 + 6 вспомогательных тензоров), мкс:

| Метод | сериализация | десериализация | размер |
|---|---|---|---|
| `torch.save` в BytesIO / `torch.load(weights_only=True)` | 380–410 | 670–710 | 274 KiB |
| `pickle` protocol 5 (in-band, numpy-массивы) | 30 | 19–24 | 271 KiB |
| `pickle` protocol 5 + out-of-band buffers (`buffer_callback`) | 19–23 | 12–14 | метаданные 309 B + буферы |
| `numpy.tobytes()` + join | 15 | (`frombuffer`, ~0) | 271 KiB |
| `safetensors.torch.save/load` | 175–190 | 46–75 | 271 KiB |

`torch.save` — в 13× медленнее на сериализации и в ~36× на загрузке, чем pickle-5, при том же размере. PEP 574 (pickle protocol 5): out-of-band буферы устраняют лишние копии; выигрыш реализуется, только если транспорт умеет отправлять буферы без склейки (ZeroMQ multipart, `socket.sendmsg`, shared memory); у `mp.Queue` pickle-5 in-band всё равно копирует.
- **Сжатие (мои)**, тот же чанк, `lz4.frame`/`lz4.block`/`zstd` (мкс; ratio):

| Данные obs | lz4.frame | lz4.block | zstd-1 | zstd-3 |
|---|---|---|---|---|
| float32 `standard_normal` (несжимаемо) | 50 µs; **1.04** | 26 µs; 1.04 | 292 µs; 1.13 | 335 µs; 1.13 |
| float32, бинарные плоскости (5% единиц) | 72 µs; **13.1** | 72 µs; 13.1 | 239 µs; **25.9** | 218 µs; 24.8 |
| float32, квантованные (шаг 0.25) | 462 µs; 2.30 | 389 µs; 2.30 | 860 µs; 4.25 | 922 µs; 4.32 |

Вывод: на несжимаемых данных lz4 почти бесплатен (ловит «несжимаемые блоки»), zstd съедает 300 мкс на чанк впустую; для игровых obs с большой избыточностью zstd-1 вдвое плотнее lz4 при 3–4× больших затратах CPU. Практика: (1) **сначала сузить dtype** (бинарные плоскости → `uint8`/bit-pack — 4–32× без сжатия; `float16` для остального), (2) сжимать только межмашинный трафик и только если сеть — узкое место; в `LocalTransport`/shared mem — не сжимать; (3) lz4 по умолчанию, zstd-1 для WAN/медленных каналов; (4) сжимать obs отдельно от мелких векторов. Поток в CLAUDE.md «300 MB/s на 100 воркеров» = 2.4 Gbit/s — на 10 GbE помещается без сжатия.
- **gRPC в Python**: официальные рекомендации (grpc.io/docs/guides/performance): переиспользовать каналы и stubs; keepalive; пул каналов при упоре в лимит параллельных стримов HTTP/2; «Streaming RPCs create extra threads … which makes streaming RPCs **much slower than unary RPCs** in gRPC Python, unlike the other languages»; asyncio-API может ускорить; future-API избегать. Лимит сообщения по умолчанию 4 MB (`grpc.max_receive_message_length`) — чанк 300 KB проходит, батчи — нет. SEED RL использует **стриминговый** gRPC (C++), с батчингом на сервере и unix-сокетами для локальных акторов. То есть «persistent client-streaming», запланированный в открытых пунктах Colosseum, в Python может оказаться **медленнее**, чем unary с батчем нескольких чанков; требует замера (я gRPC сам не мерил — **не проверено**; бенчмарки лежат в `/tmp/bench_scripts` у соседнего агента).
- Альтернативы: ZeroMQ PUSH/PULL (multipart, zero-copy буферы), сырой TCP с length-prefixed кадрами, `moolib` (RPC с автоматическим выбором транспорта shared memory/TCP/Infiniband, README), `torch.distributed`/NCCL — для градиентов multi-GPU и broadcast весов, не для потока траекторий.
- Веса: версионирование, воркер тянет только при смене версии; компактный формат (fp16/`safetensors`, mmap); у SF — копия из общей памяти <1 мс.

### 5.3 Лёрнер

- **pin_memory/prefetch**. PyTorch tutorial (pinmem_nonblock): «calling pin_memory() on a pageable tensor before casting it to GPU should not bring any significant speed-up, on the contrary this call is usually slower than just executing the transfer» (0.4333 мс против 0.3694 мс на 1M элементов); `non_blocking=True` помогает на многих тензорах (13.76 против 18.44 мс на 1000 тензоров); рекомендация — создавать тензоры сразу в pinned-памяти / предвыделять. В Colosseum `_prepare_batch` делает `v.pin_memory().to(device, non_blocking=True)` на каждый тензор каждого батча — по этому источнику это скорее замедление; нужны **предвыделенные pinned staging-буферы** (`torch.empty(..., pin_memory=True)`, `copy_` из них) и фоновый поток/второй CUDA-stream, готовящий следующий батч, пока идёт backward (IMPALA §3: «preparing the next batch of data for the learner while still performing computation»; RLlib `num_gpu_loader_threads=8`; SF `Batcher`).
- **AMP**: PyTorch AMP docs: `scaler.unscale_(optimizer)` **перед** `clip_grad_norm_` (и «unscale_ only once per optimizer per step»); для нескольких оптимизаторов — `scaler.step(optN)` на каждый, `scaler.update()` один раз после всех; `scaler.scale(loss)` для каждого лосса; «Backward passes under autocast are not recommended»; bfloat16 не требует GradScaler (не проверено по этому конкретному тексту). Пропущенный из-за inf/NaN `optimizer.step()` → шаг LR-scheduler после пропуска вызывает предупреждение; счётчики шагов считать по реально выполненным шагам.
- **torch.compile на лёрнере**: даёт выигрыш для крупных сетей при **статических формах** (фиксированный размер батча/чанка, `dynamic=False`; ещё лучше `mode="max-autotune-no-cudagraphs"` не пробовал); при переменном B (неполные батчи) — рекомпиляции; для рекуррентного покадрового цикла компиляция разворачивает T итераций (долгая компиляция).
- **V-trace/GAE: вычисление**. Мои замеры (CPU, T=256, B=64, 1 поток): python-цикл на torch ≈ 2 870 мкс; `torch.jit.script` ≈ 2 040; numpy-цикл ≈ 730; `torch.compile` (цикл развёрнут) ≈ 386 мкс при **компиляции 35 с**. На GPU цикл T=256 — это ~1–2 тысячи запусков маленьких ядер на вызов. Эталонные решения: PufferLib — один CUDA-поток на строку (`puff_advantage`, `src/algo.cu`: «GAE / full truncated IS (V-trace ρ/c): ρ̄ on δ, c̄ on λ product», 16-байтовые векторные загрузки); RLlib — цикл по T, в старом коде «on CPU for better perf», в новом «modern GPUs are quite optimized … leave on GPU»; оригинал IMPALA — на CPU; TorchRL — GAE через `conv1d` с геометрическим рядом `(γλ)^k` (постоянные коэффициенты; для V-trace `c_t` зависят от данных, свёртка не годится — нужен scan/компиляция); rlax — `jax.lax.scan`.
  Рекомендация: считать V-trace один раз за минибатч в **одном** fused-ядре (compile одной функции с фиксированным T или простое numpy/CPU вычисление в потоке подготовки батча), а не внутри autograd-графа.

### 5.4 Прочее

- Не передавать через `mp.Queue` объекты `torch.Tensor`, требующие grad; использовать `inference_mode`-тензоры только там, где лёрнер потом не будет их использовать в autograd (SF различает режимы).
- Профилировать долю: env step / inference / сериализация / ожидание очереди / backward (SF `docs/07-advanced-topics/profiling.md`, метрики `train/valids_fraction`, `version_diff_*`).

---

## 6. Типичные баги в самописных IMPALA/APPO (чек-лист ревьюера)

Каждый пункт — что проверить и как должно быть; отдельно — таблица в разделе 7.

1. **Сдвиг obs/action/reward/done.** Должно быть `(s_t, a_t, logμ_t, V_t, r_t, term_t, trunc_t)` — все с индексом t относятся к одному решению; `V_next` — по `s_{t+1}`. Юнит-тест: детерминированная среда «счётчик» с известными `V`, сверка с ручным расчётом на чанке с done в середине.
2. **Bootstrap не от той сети / не от того состояния.** Конец чанка — по `obs_T` и текущей сети лёрнера (IMPALA/SF); значение от воркера (behavior-сеть, устаревшее) смещает цели. При done на последнем шаге — множитель `(1−term)`.
3. **Truncation трактуется как termination** (bootstrap 0 на тайм-лимите) — занижение `V` у длинных игр; нужно `V(final_obs)` (SB3/TorchRL); при auto-reset `next_obs` — уже первый кадр следующего эпизода.
4. **След V-trace пересекает границу эпизода** (`c_t` не обнулён по done) или, наоборот, обрывается на границе чанка из-за `loss_mask`. Должно: `γ(1−done)·c` в рекурсии.
5. **Утечка hidden между эпизодами**: сброс в воркере и лёрнере должен быть в одном и том же месте (до обработки первого obs нового эпизода); лёрнер должен использовать те же `done` (не сдвинутые на 1). Проверка: в тесте подать на лёрнер одну и ту же последовательность и убедиться, что `log_prob` совпадает с записанным воркером до 1e-5 (при тех же весах) — это универсальный тест на выравнивание и скрытые состояния.
6. **Stored state не совпадает с реальным**: `h_init` записан *до* или *после* обработки первого шага чанка; после сброса по done в начале чанка `h_init` должен быть нулём.
7. **Маска при пересчёте log-prob**: behavior log-prob записан с маскированным распределением; лёрнер пересчитывает с той же маской; энтропия/KL — по маскированному; padding/terminal-строки исключены; fp16 переполнение маски.
8. **Составные действия**: логпроб неиспользуемых голов не включён; ratio — по сумме; хранится сырой семпл (для tanh — `u`).
9. **Ratio относительно «не той» политики**: PPO-clip с ratio к μ при большом lag обнуляет градиенты (всё клипнуто) — вместо «V-trace исправляет» получаем «данные выброшены»; при lag > ~20–30 SGD-шагов мониторить `clipfrac`, `approx_kl`, `version_diff`.
10. **Двойной учёт IS**: `adv` уже содержит `ρ̄_pg=min(ρ̄,π/μ)`, а в surrogate умножается на ratio (так в SF, это норма) — убедиться, что нет **третьего** множителя.
11. **Нормализация advantage** — по минибатчу, но по `valid`-маске; не по всему `[T,B]` с паддингом; `std` с eps.
12. **AMP/GradScaler**: `unscale_` до clip; `scaler.update()` один раз; при нескольких оптимизаторах — по правилу выше; `autocast` вокруг forward, **не** вокруг backward; scheduler.step после **успешного** optimizer.step; bf16 — без scaler.
13. **LR schedule не по тем шагам**: в Colosseum `lr_scheduler.step()` вызывается раз за `train_step` (политику версий), а `setup_lr_schedule(total_steps)` получает «total_steps» — убедиться, что единицы совпадают (обновления, а не env-шаги; `train_step` содержит `num_epochs·minibatches` оптимизаторных шагов).
14. **Энтропия/value loss — редукции**: IMPALA суммирует по T и B (коэффициенты 0.5 / 0.00025 под сумму); PPO-стек — среднее (0.5 / 0.01). Нельзя смешивать коэффициенты из разных конвенций.
15. **Staleness не контролируется**: нет `max_policy_lag`/фильтра, не логируется `policy_version` чанка; несколько эпох+async → lag накапливается (SF: формула).
16. **Value target без нормализации на шкале наград >> 1**: огромные градиенты value, `value_loss` доминирует; нужен return normalization или reward scaling.
17. **Наблюдательная нормализация расходится между воркерами/лёрнером/оппонентами пула**, статистики не в чекпойнте.
18. **Пошаговые игры**: нет закрывающего перехода для того, кто ходил не последним; награда выдана не тому игроку; переход строится по ходам обоих игроков в один `[T]`-ряд (смешиваются перспективы).
19. **Kickstarting для рекуррентных сетей**: KL к учителю считается без hidden/без маски (stateless, на `flat_obs`), что несовместимо с рекуррентной студенческой политикой/маскированными распределениями.
20. **pin_memory на каждый батч**, `.item()` на каждый минибатч (синхронизация GPU; копить на устройстве, `.item()` раз за итерацию), `torch.save` в горячем пути, `lz4` на несжимаемых данных.

---

## 7. Итоговый чек-лист для ревью реализации

Формат: пункт → как правильно → источник. «M» = must-have, «N» = nice-to-have.

### A. APPO / V-trace
| # | Пункт | Как правильно | Источник |
|---|---|---|---|
| A1 M | Ratio PPO | Явно выбрать схему: SF (к μ из воркера) или IMPACT (к target-сети). Смешивать нельзя. При SF — clamp ratio, асимметричный clip `[1/(1+ε),1+ε]` допустим | SF `learner.py::_calculate_losses`; RLlib `appo_torch_learner.py` L168–181 |
| A2 M | Advantage | `ρ̄_pg (r + γ(1−term) v_{next} − V)` из V-trace; GAE не накладывается поверх | IMPALA `vtrace.py`; RLlib `vtrace_torch_v2.py` L160–166 |
| A3 M | λ в V-trace | `c_t = λ·min(c̄, ρ_t)`, λ≈0.9 | IMPALA Remark 2; rlax `vtrace`; Andrychowicz §3.4; Puffer `puff_advantage` |
| A4 M | Два флага конца | `terminated` — обнуляет bootstrap; `done` — обрывает след; при truncation — `V(final_obs)` | TorchRL `vtrace_advantage_estimate`; SB3 `collect_rollouts`; Pardo 1712.00378 |
| A5 M | Bootstrap чанка | `V(obs_T)` пересчитывать лёрнером по текущей сети | IMPALA `build_learner`; SF `_prepare_batch`; RLlib +1ts |
| A6 M | Value loss | MSE к `vs` без value clipping | Andrychowicz C13; Engstrom; Huang #9 |
| A7 M | Max policy lag | Фильтр/маска по `train_step − chunk.version`, логировать lag, масштабировать LR долей валидных | SF `_prepare_batch`, `policy-lag.md`; OpenAI Five §staleness |
| A8 N | Пересчёт на каждом минибатче | Допустимо (SF); альтернатива — зафиксированная target-сеть (RLlib) | SF vs RLlib |
| A9 N | ρ̄, c̄ | 1.0/1.0 по умолчанию; ρ̄ определяет fixed point, c̄ — variance | IMPALA Remark 3 |
| A10 N | Эпохи | 1–4; следить за `clipfrac`/KL; Cleanba: PPO с 16 шагами/роллаут хуже при async | Cleanba §5.3, §6 |

### B. Рекуррентность
| B1 M | Stored state | Сохранять `h_init` на начало чанка (до первого шага), после reset — нули | SF; IMPALA `first_agent_state`; Huang #36 |
| B2 M | Сброс внутри чанка | Сбрасывать `h` и `c` всех слоёв перед обработкой первого obs нового эпизода; та же конвенция у воркера и лёрнера | CleanRL `get_states`; IMPALA `unroll`; SF `build_rnn_inputs` |
| B3 M | Тест соответствия | Лёрнер при тех же весах воспроизводит записанные воркером `log_prob`/`value` | — (контрольный тест) |
| B4 N | Burn-in | Для больших lag/длинных чанков: первые K шагов без лосса | R2D2 |
| B5 N | Truncated BPTT с хранимым состоянием каждые K шагов | Для T=256+; V-trace требует полной последовательности | OpenAI Five (16-шаговые сегменты); SF `recurrence` |
| B6 N | Скорость | Не вызывать `nn.LSTM` по одному шагу, если можно разбить на сегменты между done | SF `rnn_utils.py` |

### C. Нормализации / оптимизация
| C1 M | Obs normalization | Running mean/std, clip, единственный писатель, статистики в чекпойнте/пуле | SF `running_mean_std.py`; RLlib `MeanStdFilter.merge_states`; Five |
| C2 M | Init | Ортогональная, policy 0.01, value 1.0 | Engstrom #3; Huang #2; Andrychowicz C57 |
| C3 M | Grad clip | Global norm; учесть редукцию лосса | Engstrom #9; SF 4.0; RLlib 40 |
| C4 M | Adv norm | По минибатчу по valid | SF; Huang #7 |
| C5 N | Return/value normalization | Running std (SF) или PopArt (разные шкалы) | MAPPO §5.1; PopArt |
| C6 N | LR/entropy schedule | Линейно; единицы шагов одинаковы в scheduler и конфиге | Engstrom #4; Five Table 2 |
| C7 M | Редукция лосса | Не смешивать «sum»- и «mean»-коэффициенты | IMPALA `experiment.py` vs PPO |

### D. Мультиагент и пошаговые игры
| D1 M | Переходы | Только ходящий игрок; время = индекс собственных решений | RLCard `reorganize`; OpenSpiel `dqn.step` |
| D2 M | Закрывающий переход | Терминальная награда и `done` — последнему переходу **каждого** игрока | RLCard; OpenSpiel `tic_tac_toe_qlearner.py` |
| D3 M | Накопление наград | Награда за переход = сумма наград между его ходами | PettingZoo AEC `_cumulative_rewards` |
| D4 M | `collect_mask` | Не копить траектории оппонентов-пула; их ходы — «среда» | Colosseum design, SF `valids` |
| D5 N | FFA | Zero-sum rank reward; выбывшие — `done` + inactive | Five (symmetrize), SF `inactive-agents.md` |
| D6 N | Эгоцентричное наблюдение, случайная рассадка | — | MAPPO, практика |

### E. Маски и действия
| E1 M | Маска хранится и применяется при пересчёте log-prob/entropy/KL | masked logits `−1e8…−1e9`; behavior log-prob маскированный | Huang&Ontañón; SF `action_distributions.py` |
| E2 M | Padding/терминал | Исключить из лосса | RLlib `loss_mask`, SF `valids` |
| E3 M | AMP | Логиты/маска в fp32 | PyTorch AMP |
| E4 M | Непрерывные | Хранить сырой семпл `u`; tanh — с якобианом; clip — только на входе среды | SAC App. C; CleanRL |
| E5 M | Зависимые головы | Маска использования `m_h`; зависимые маски хранить; энтропия/KL только использованных | OpenAI Five App. F |

### F. Производительность
| F1 M | `torch.set_num_threads(1)` + `OMP/MKL_NUM_THREADS=1` в воркерах | SF `rollout_worker.py:70` |
| F2 M | Батчинг inference по средам | измерено: ~10× на сэмпл при B=32 |
| F3 M | Сериализация | pickle-5 / `tobytes`/shared memory, **не** `torch.save`; по FIFO — индексы буферов | измерено; SF paper §3.2 |
| F4 M | Сжатие | Сначала сузить dtype; сжимать только межмашинный трафик; lz4 по умолчанию | измерено |
| F5 M | pin_memory | Предвыделенные pinned-буферы + prefetch | PyTorch pinmem tutorial |
| F6 N | `mp` стратегия | `file_system` при многих чанках в полёте | PyTorch multiprocessing doc; SF `init_torch_runtime` |
| F7 N | gRPC | Unary с батчем может быть быстрее стриминга в Python; переиспользовать каналы; 4 MB лимит | grpc.io performance |
| F8 N | V-trace | Один fused-вызов (compile/numpy) на минибатч | измерено; Puffer `puff_advantage` |
| F9 N | Двойная буферизация | Только при отдельном inference-процессе/async env | SF `double-buffered.md` |

---

## 8. Приложение: наблюдения при быстром чтении кода Colosseum (гипотезы для проверки)

Я не проводил полного ревью; прочитаны `algorithms/appo.py`, `algorithms/vtrace.py`, фрагменты `networks/actor_critic.py`, `worker/rollout_worker.py`. Все пункты — **гипотезы**, подлежащие проверке по чек-листу выше.

1. `compute_vtrace(...)` использует один флаг `dones` и для bootstrap, и для обрыва следа; `rollout_worker.py` ставит `done = terminated or truncated` и `bootstrap_val = 0.0 if done` — truncation не бутстрапится (A4).
2. `bootstrap_value` вычисляется на **воркере** сетью воркера (устаревшая, behavior) и хранится в чанке; в IMPALA/SF — лёрнером по текущей сети (A5).
3. В V-trace отсутствует λ (только `c_bar`) (A3).
4. `compute_loss` пересчитывает V-trace на каждом минибатче по текущей θ — совпадает с SF, значит ratio к μ из воркера согласован; **но** `CLAUDE.md` описывает это как «APPO … как в Sample Factory» — у SF V-trace опционален и по умолчанию выключен.
5. `evaluate_actions_recurrent`: сброс `h*=not_done` **после** шага t при `done_t=1`; эквивалентно CleanRL при условии, что воркер сбрасывает после того же шага (B2/B3 — нужен тест соответствия). Цикл по T шагам с `nn.LSTM` (неоптимально, §5.3).
6. Kickstart: `self._kickstart.compute(self._network, flat_obs)` вызывается на плоских наблюдениях без `hidden`/маски — для рекуррентных/маскированных политик нет согласованности (E1, §6.19).
7. `pin_memory()` на каждый тензор каждого батча (`_prepare_batch`) — по PyTorch tutorial это обычно медленнее (F5).
8. LR scheduler шагает раз за `train_step`; убедиться, что `setup_lr_schedule(total_steps)` считает в тех же единицах (C6, §6.13).
9. `value_loss = F.mse_loss(...)` без множителя 0.5 и `entropy.mean()` — это «mean»-конвенция; коэффициенты `value_loss_coeff`, `entropy_coeff` не переносить из IMPALA (C7).
10. Нормализация advantage считается по всему `[T,B]` минибатча без `valid`-маски — для пошаговых игр с пропусками потребуется маска (D1).

---

## 9. Источники

**Эталонный код**
- Sample Factory — https://github.com/alex-petrenko/sample-factory : `sample_factory/algo/learning/learner.py`, `.../learning/rnn_utils.py`, `.../utils/running_mean_std.py`, `.../utils/action_distributions.py`, `.../utils/torch_utils.py`, `.../sampling/rollout_worker.py`, `sample_factory/cfg/cfg.py`, `docs/07-advanced-topics/{policy-lag,normalizations,action-masking,inactive-agents,double-buffered}.md`, `docs/06-architecture/overview.md`
- RLlib (Ray master) — https://github.com/ray-project/ray/tree/master/python/ray/rllib : `algorithms/appo/torch/appo_torch_learner.py`, `algorithms/appo/appo.py`, `algorithms/appo/appo_learner.py`, `algorithms/appo/appo_torch_policy.py`, `algorithms/impala/torch/vtrace_torch_v2.py`, `connectors/learner/add_one_ts_to_episodes_and_truncate.py`, `connectors/env_to_module/mean_std_filter.py`, `utils/postprocessing/zero_padding.py`
- DeepMind IMPALA — https://github.com/deepmind/scalable_agent : `vtrace.py`, `experiment.py`, `environments.py`
- rlax — https://github.com/google-deepmind/rlax/blob/master/rlax/_src/vtrace.py
- CleanRL — https://github.com/vwxyzjn/cleanrl : `cleanrl/ppo_atari_lstm.py`, `cleanrl/ppo_continuous_action.py`
- PufferLib — https://github.com/PufferAI/PufferLib (tag 3.0: `pufferlib/vector.py`, `pufferlib/pufferl.py`; master: `src/algo.cu`, `config/default.ini`)
- TorchRL — https://github.com/pytorch/rl/blob/main/torchrl/objectives/value/functional.py
- Stable-Baselines3 — https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py
- PettingZoo — https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/utils/env.py
- OpenSpiel — https://github.com/google-deepmind/open_spiel : `open_spiel/python/rl_environment.py`, `open_spiel/python/pytorch/dqn.py`, `open_spiel/python/examples/tic_tac_toe_qlearner.py`
- RLCard — https://github.com/datamllab/rlcard/blob/master/rlcard/utils/utils.py (`reorganize`), `rlcard/envs/env.py` (`run`)
- moolib — https://github.com/facebookresearch/moolib ; SEED RL — https://github.com/google-research/seed_rl

**Статьи и документация**
- IMPALA — https://arxiv.org/abs/1802.01561
- IMPACT (основа RLlib APPO) — https://arxiv.org/abs/1912.00167
- Sample Factory — https://arxiv.org/abs/2006.11751
- Cleanba — https://arxiv.org/abs/2310.00036
- Andrychowicz et al., What Matters In On-Policy RL — https://arxiv.org/abs/2006.05990
- Engstrom et al., Implementation Matters in Deep Policy Gradients — https://arxiv.org/abs/2005.12729
- Huang et al., The 37 Implementation Details of PPO — https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- Huang & Ontañón, A Closer Look at Invalid Action Masking — https://arxiv.org/abs/2006.14171
- OpenAI Five (Dota 2 with Large Scale Deep RL) — https://arxiv.org/abs/1912.06680
- MAPPO — https://arxiv.org/abs/2103.01955
- PopArt: van Hasselt et al. — https://arxiv.org/abs/1602.07714 ; Hessel et al. — https://arxiv.org/abs/1809.04474
- SAC (tanh-squash, Приложение C) — https://arxiv.org/abs/1812.05905
- Pardo et al., Time Limits in RL — https://arxiv.org/abs/1712.00378
- R2D2 — https://openreview.net/forum?id=r1lyTjAqYX (PDF не удалось скачать; детали сверены по Acme https://arxiv.org/abs/2006.00979 и пересказам)
- PettingZoo paper — https://arxiv.org/abs/2009.14471
- EnvPool — https://arxiv.org/abs/2206.10558 ; SEED RL — https://arxiv.org/abs/1910.06591
- PEP 574 (pickle protocol 5) — https://peps.python.org/pep-0574/
- gRPC performance best practices — https://grpc.io/docs/guides/performance/
- PyTorch AMP examples — https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- PyTorch pin_memory/non_blocking tutorial — https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
- PyTorch multiprocessing (sharing strategies) — https://docs.pytorch.org/docs/stable/multiprocessing.html
- AlphaStar (Nature 2019) — https://www.nature.com/articles/s41586-019-1724-z (структура авторегрессивных голов — по памяти, не проверено по тексту)

**Собственные бенчмарки** (скрипты в `/tmp/r3_perf/`: `b_infer.py`, `b_ser.py`, `b_vt.py`, `b_comp.py`; общий venv проекта, torch 2.14.1+cpu, 1 поток, loadavg 18–22 на 8 потоках — относительные величины надёжнее абсолютных).

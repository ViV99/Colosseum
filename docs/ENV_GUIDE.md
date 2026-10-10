# Как написать среду для Colosseum

Этот документ объясняет контракт среды SP2 (`GameSpec` + `MultiAgentEnv` + `StepResult`) на примерах
для каждого типа игры. Все примеры — рабочие демо-среды из `examples/`, их конфиги лежат в
`configs/examples/`. Полное описание контракта — спека
`docs/superpowers/specs/2026-10-08-sp2-game-model-design.md` (блоки 1–3); код контракта —
`src/colosseum/envs/game.py`, `src/colosseum/envs/spaces.py`, проверки — `src/colosseum/envs/contract.py`.

| Тип игры | Демо-среда | Конфиг | Что в ней показано |
|---|---|---|---|
| Соло | `examples/coin_grid` | `coin_grid.yaml` | `GameSpec.solo`, Dict-наблюдение с `uint8`, маски, обрыв по лимиту шагов (`truncated`, `final_obs`) |
| 1v1 пошаговая | `examples/tic_tac_toe` | `tic_tac_toe.yaml` | ходит одно место из двух (`acting`), маски, награды ждущему месту |
| Бот с юнитами | `examples/unit_harvest` | `unit_harvest.yaml` | `Units`, рождение и гибель юнитов, списки сущностей с маской, `Dict`-действие |
| Команда на команду | `examples/team_tag` | `team_tag.yaml` | `GameSpec.teams_of`, локальное окно, `global_state` для критика, выбывший сокомандник |
| FFA с выбыванием | `examples/tron` | `tron.yaml` | `GameSpec.symmetric([2, 3, 4])`, `terminated`, ранги по порядку выбывания |
| 1 vs N, асимметрия | `examples/predator_prey` | `predator_prey.yaml` | две роли с разными пространствами, два агента (`agents.<id>.roles`) |
| Кооператив | `examples/coop_buttons` | `coop_buttons.yaml` | одна команда, исход `score`, `teammates: mixed`, cross-play |
| Соревновательная игра «как на соревновании» | `examples/space_miners` | `space_miners.yaml` | `Units` с `Box`-компонентом, маски внутри юнитов, сущности, Box2D |
| Дерево действий | `examples/composite_action` | `chase.yaml` | `Dict(direction=Discrete, speed=Box)` |
| Скриптовые боты (SP3) | `examples/unit_harvest/bots.py`, `examples/tic_tac_toe/bots.py` | `unit_harvest_league.yaml` | `ScriptedBot` как якорь и учитель; что среда отдаёт ботам через `infos` — раздел 10, лига — `docs/LEAGUE_GUIDE.md` |

## 1. Минимальная среда

Среда — подкласс `colosseum.envs.game.MultiAgentEnv` с атрибутом `spec` и двумя методами:

```python
import gymnasium
import numpy as np

from colosseum.envs.game import GameSpec, MultiAgentEnv, StepResult


class Counter(MultiAgentEnv):
    def __init__(self, length: int = 10) -> None:
        self.length = length
        self.spec = GameSpec.solo(gymnasium.spaces.Box(0.0, 1.0, (1,), np.float32), gymnasium.spaces.Discrete(2))
        self._t = 0

    def _obs(self) -> np.ndarray:
        return np.array([self._t / self.length], np.float32)

    def reset(self, seed: int | None, layout: str) -> StepResult:
        self._t = 0
        return StepResult(acting={0}, obs={0: self._obs()})

    def step(self, actions: dict) -> StepResult:
        self._t += 1
        reward = float(actions[0])                 # +1 за действие 1
        if self._t == self.length:                 # конец по правилам
            return StepResult(acting=set(), obs={}, rewards={0: reward}, episode_over=True)
        return StepResult(acting={0}, obs={0: self._obs()}, rewards={0: reward})
```

Конфиг: `env.env_class: my_game.env.Counter`, `env.kwargs: {length: 10}`. Перед обучением всегда
запускайте `colosseum validate -c my_game.yaml`: он проверяет спеку и конфиг, делает `reset` каждого
включённого варианта партии (`matchmaking.layouts`, по умолчанию все) и до 8 шагов случайными
легальными действиями с проверкой контракта и полной проверкой пространств (`space.contains`), а
затем прогоняет `step` и `unroll` модели каждого агента. Модуль `my_game` должен импортироваться:
текущая папка добавляется в `sys.path` (и в дочерних процессах), поэтому запускайте команды из папки,
где лежит `my_game/`.

## 2. `GameSpec`: роли, места, команды, варианты партии

```python
RoleSpec(observation_space, action_space, global_state_space=None)   # тип места
SeatSpec(role="player", team=0)                                      # одно место варианта
GameSpec(roles={"player": RoleSpec(...)}, layouts={"2p": (SeatSpec("player", 0), SeatSpec("player", 1))})
```

- **Вариант партии** (`layout`) — кортеж мест; место `i` варианта — это ключ `i` во всех словарях
  `StepResult`. Номера команд варианта — ровно `0..T-1`. Имена ролей и вариантов — буквы, цифры, `_`
  и `-` (не с `-` в начале).
- **Тип исхода выводится из числа команд**: одна команда — `score` (соло, кооператив), две — `wdl`,
  три и больше — `rank`. Одна игра может иметь варианты разных типов.
- **Разное число игроков в одной игре** — разные варианты: `GameSpec.symmetric([2, 3, 4], obs, act)`
  даёт `2p`, `3p`, `4p`. Места `n..P_max-1` в варианте `np` — пустые: им нельзя давать наблюдения,
  маски, награды или `terminated`. Доли вариантов при обучении задаёт `matchmaking.layouts`
  (`configs/examples/tron.yaml`).

Хелперы (роль по умолчанию называется `player`):

| Хелпер | Варианты | Пример |
|---|---|---|
| `GameSpec.solo(obs, act, global_state=None)` | `solo` | `coin_grid` |
| `GameSpec.symmetric(n или [n1, n2, ...], obs, act, global_state=None)` | `2p`, `3p`, ... (команда = место) | `tic_tac_toe`, `tron` |
| `GameSpec.teams_of([2, 2], obs, act, global_state=None)` | `2v2`; `[[2, 2], [3, 3]]` — несколько вариантов; `[2, 1, 1]` — `2v1v1`; места нумеруются по командам | `team_tag` |
| `GameSpec.teams_of([2], obs, act, ...)` | `coop2` (одна команда) | `coop_buttons` |

**Асимметричные роли** — `GameSpec` целиком руками (`examples/predator_prey/game.py`):

```python
hunter = RoleSpec(Box(-1, 1, (9,), np.float32), Discrete(5))
prey = RoleSpec(Box(-1, 1, (7,), np.float32), Discrete(9))
spec = GameSpec(roles={"hunter": hunter, "prey": prey},
                layouts={"1v2": (SeatSpec("hunter", 0), SeatSpec("prey", 1), SeatSpec("prey", 1))})
```

Один агент играет только роли с одинаковыми пространствами (наблюдение, действие и `global_state`):
у агента одна сеть, а её входы и выходы задаются пространствами. Роли с разными пространствами —
разные агенты:

```yaml
agents:
  hunter: {roles: [hunter]}
  prey: {roles: [prey]}
```

Если `roles` не указаны, агент играет все роли игры, и тогда у всех ролей должны совпадать
пространства (иначе `ConfigError` с подсказкой). Роли с общим пространством, которые различаются
по смыслу (например, «нападающий» и «защитник» одной сети), кодируйте во входе наблюдения.

## 3. `StepResult`: кто ходит, награды, конец эпизода

```python
StepResult(
    acting={...},            # места, которые ходят на СЛЕДУЮЩЕМ шаге
    obs={seat: obs},         # обязательно для каждого ходящего места
    action_masks={seat: m},  # только для ходящих; нет ключа — всё разрешено
    rewards={seat: r},       # любому живому месту; нет ключа — 0
    terminated={...},        # места, выбывшие на этом шаге
    episode_over=False,
    truncated=False,         # оборван искусственным лимитом, а не по правилам
    final_obs=None,          # при truncated: каждому живому месту
    global_state=None,       # для ролей с global_state_space (раздел 6)
    outcome=None,            # при episode_over (раздел 5)
    infos={},
)
```

- `reset(seed, layout)` возвращает `StepResult` с `acting`, `obs`, `action_masks`, `global_state`;
  наград, `terminated`, `episode_over` и `outcome` нет. `seed` нужно применять: одинаковый сид —
  одинаковый эпизод.
- `step(actions)` получает словарь ровно по местам из предыдущего `acting`.
- Ключи всех словарей и элементы `acting`/`terminated` — целые номера мест.
- **Пошаговые игры**: в `acting` одно место (`tic_tac_toe`). **Одновременные ходы**: все живые места
  (`tron`, `unit_harvest`).
- **Шаг без решений** (`acting` пуст, эпизод не закончен) допустим: воркер вызовет `step({})`. Больше
  `env.max_idle_steps` (по умолчанию 1000) таких шагов подряд — ошибка контракта: так ловится среда,
  которая забыла выставить `episode_over`.
- **Ждущее место не получает шага модели.** Модель места вызывается (и обновляет своё состояние,
  например LSTM) только на ходах этого места, а наблюдения не ходящих мест фреймворк игнорирует, их
  можно не отдавать. Поэтому **всё, что произошло между ходами места и что политике нужно знать**
  (ходы соперника, история, туман войны), должно быть в наблюдении её следующего хода: среда сама
  кладёт туда то, что место должно помнить. Рекуррентное ядро помнит только то, что место видело на
  своих ходах.
- **Награды ждущему месту** разрешены (в `tic_tac_toe` проигравший получает −1, когда ходит победитель):
  они копятся в последнем переходе места; награды до первого хода места уходят в его первый переход.
  **Место, которое за эпизод так и не сходило, теряет свои награды**: их не к чему приписать.
  Первая такая потеря на воркере пишется в `logs/worker-<i>.log` как WARNING с контекстом
  (`worker 0, env 1, seat 3, episode step 40, layout 4p: agent 'a' loses reward -1.0 ...`); дальше
  потери только считаются. Поле `dropped_reward_episodes` записи `system` в `metrics.jsonl` и в WandB
  (`system/dropped_reward_episodes`) — сумма последних счётчиков (эпизоды-места с потерянной
  ненулевой наградой) тех воркеров, что прислали статистику за последние несколько секунд. Поэтому
  оно может уменьшиться, когда воркер подвисает, и показать 0 в последней записи при остановке;
  надёжный сигнал — WARNING в логе воркера. Рейтинги и исходы партий от потерь не страдают: они
  считают все награды. Если потери есть, проверьте, что каждое место ходит в эпизоде хотя бы раз,
  или начисляйте награду на шаге, где место ходит.

### Жизненный цикл места

| Состояние | Что можно | Что нельзя |
|---|---|---|
| пустое (нет в варианте) | ничего | наблюдение, маска, `global_state`, награда, `terminated`, `acting` |
| живое | ходить, ждать, получать награды | — |
| выбывшее (`terminated` раньше) | ничего | ходить, награды, повторный `terminated` |

- **Выбывание** (`terminated={seat}`) значит «return этого места окончателен». Награда в том же
  `StepResult` разрешена (типичный штраф за смерть) и применяется первой (`tron`, пойманная жертва в
  `predator_prey`). Место нельзя выбить и включить в `acting` в одном шаге.
- **Мёртвый юнит или игрок команды, которому ещё положена командная награда в конце,** *не*
  терминируется: среда просто перестаёт включать его в `acting` и продолжает начислять ему награды.
  `terminal` он получит в конце эпизода. Пример — заморозка в `team_tag`: замороженный бот стоит на
  поле, не ходит и получает общую награду команды. Выбирайте `terminated`, только если места больше
  ничего не ждёт.

### Конец эпизода: правила против обрыва

- **По правилам** (победа, выбывание всех, фиксированная длина матча вроде 505 шагов Lux S3):
  `episode_over=True`, `acting` пуст, `truncated=False`. Bootstrap = 0.
- **Искусственный обрыв** (лимит шагов, которого нет в правилах игры): `episode_over=True`,
  `truncated=True` и `final_obs` для **каждого живого места** (и финальный `global_state`, если роль
  его объявила). Лёрнер посчитает V по `final_obs` своей свежей сетью. `coin_grid` — пример: у игры
  нет конца, лимит в 50 шагов — обрыв.

Если сомневаетесь: является ли лимит длины частью игры? Да — это правило (`unit_harvest`, `team_tag`,
`coop_buttons`: фиксированная длина матча). Нет — это обрыв (`coin_grid`). Признак «оставшееся время»
в наблюдении полезен в обоих случаях и решения не меняет: `coin_grid` отдаёт долю оставшихся шагов и всё
равно обрывает эпизод (`truncated`), а `tron` время не показывает вовсе. Подавать его — не правило всех
демо-игр, а решение среды.

## 4. Наблюдения: деревья, `uint8`, списки сущностей

`observation_space` роли: `Box`, `Discrete`, `MultiBinary`, `MultiDiscrete` или вложенный `Dict` из
них. Наблюдение — дерево numpy-массивов той же структуры.

- **Тип данных сохраняется** всюду: среда → буфер → чанк → батч лёрнера. Карта в `uint8` остаётся
  `uint8` (в 4 раза меньше трафика, чем `float32`); к `float` приводит энкодер:
  `obs["grid"].float()` (`examples/coin_grid/models.py`).
- **Список сущностей** — соглашение `Dict(entities=Box(N_max, F), entity_mask=MultiBinary(N_max))`:
  строки без сущности — нули, маска — 0. Энкодер усредняет или считает attention только по маске
  (`masked_mean` в `examples/unit_harvest/models.py`).
- Предупреждение для attention-энкодеров: в служебных слотах чанка (`pad`) фреймворк копирует
  предыдущее наблюдение, но если среда может отдать наблюдение с пустой маской сущностей, энкодер не
  должен давать NaN (softmax по пустому множеству). Усреднение с `clamp(min=1)` безопасно.
- **Перспектива места — забота среды.** В симметричных играх удобно отражать доску так, чтобы каждое
  место «начинало слева» (`unit_harvest`, `team_tag`, `space_miners`), и отражать действия обратно.

### Порядок ключей `gymnasium.spaces.Dict`

`gymnasium.spaces.Dict({...})` из обычного `dict` **сортирует ключи по алфавиту**. От порядка
компонентов зависят раскладка маски юнитов (`action`-маска `Units` склеивает компоненты в этом
порядке) и порядок голов. Чтобы порядок был таким, как написано, передавайте список пар (или
`OrderedDict`):

```python
gymnasium.spaces.Dict([("base", Discrete(2)), ("workers", Units(8, Discrete(5)))])   # порядок как написано
gymnasium.spaces.Dict({"workers": ..., "base": ...})                                  # станет base, workers
```

Так построены все демо-среды (`examples/unit_harvest/game.py`, `examples/space_miners/game.py`).

## 5. Исход матча (`Outcome`)

`outcome` задаётся при `episode_over` и описывает **команды** варианта (в FFA и соло команда = место):

```python
Outcome(team_rank={0: 1.0, 1: 2.0})                # 1 — лучший; ничья — равные ранги; дробные допустимы
Outcome(team_score={0: 7.0, 1: 3.0})               # игровой счёт; ранги выводятся (больше — лучше)
```

- Без `outcome` счёт команды = **среднее** по её местам их наград за эпизод (среднее, а не сумма,
  чтобы команды разного размера сравнивались честно), ранги выводятся из счёта
  (`examples/composite_action`).
- Можно задать оба поля: тогда ранги берутся как есть, а не из счёта (`space_miners`: счёт игры и
  ранги с правилом «кто забил первым» при равном счёте). С одним `team_rank` счёт = среднее наград.
- Ключи — ровно команды варианта.
- Общие места в FFA: `tron` делит места поровну между выбывшими в одном шаге (два места за 3-е и
  4-е — оба 3.5).

## 6. `global_state` и централизованный критик

Если у роли есть `global_state_space`, среда отдаёт `global_state[seat]` каждому ходящему месту этой
роли на каждом шаге (и при `reset`), а при обрыве — каждому живому месту. Он доходит только до
value-пути модели: `networks.critic_encoder_class` (подкласс `BaseCriticEncoder`) превращает его в
вектор, который конкатенируется с признаками ядра перед value-головой. Политика его не видит, поэтому
на соревновании модель работает без него. `critic_encoder_class` без `global_state_space` у ролей
агента — `ConfigError`.

```yaml
networks:
  encoder_class: examples.team_tag.models.TagEncoder          # локальное окно
  critic_encoder_class: examples.team_tag.models.TagCriticEncoder   # вся карта
```

`global_state` хранится в каждом слоте чанка, поэтому он стоит памяти и трафика: замер на `team_tag`
— в `docs/benchmarks.md` (payload чанка ×1.61, шаг лёрнера +29 %). Используйте `uint8` и компактные
представления.

## 7. Действия: деревья, `Units`, маски

`action_space` роли: `Discrete`, `MultiDiscrete`, `Box` (1-D, float), `Units` или вложенный `Dict` из
них. Действие — дерево: `int64` для дискретных частей, `float32` для `Box`. `Box`-действия фреймворк не
обрезает: обрезайте в среде (`np.clip`, `examples/composite_action/game.py`).

### `Units`: один бот, много юнитов

```python
from colosseum.envs.spaces import Units

Units(max_units, per_unit, only_if=None)
# per_unit: Discrete(A) | MultiDiscrete([A1, ...]) | Box(d,) | Dict из Discrete и Box
```

- Действие группы — массив с ведущим измерением `[U]`: `int64[U]` для `Discrete`, `int64[U, C]` для
  `MultiDiscrete`, `float32[U, d]` для `Box`, словарь таких массивов для `Dict`.
- Индекс слота назначает среда; фреймворк не связывает слоты между шагами. Новый юнит — в свободный
  слот, погибший освобождает слот (`unit_harvest`).
- Разные типы юнитов — `Dict` из нескольких `Units`:
  `Dict([("factories", Units(10, Discrete(4))), ("robots", Units(200, MultiDiscrete([5, 64]))), ("global", Discrete(3))])`.
- **GridNet** — частный случай `Units(H * W, ...)`: юнит = клетка. Хелпер
  `colosseum.networks.heads.gridnet_to_units` перекладывает `[B, C, H, W]` в `[B, H*W, C]`.
- **Цель-указатель** — компонент `Discrete(N_targets)` с маской по существующим целям.
- **`only_if={компонент: (родитель, {значения})}`**: компонент учитывается (log-prob, энтропия, KL,
  лосс), только если выбранное значение родителя (дискретный компонент того же юнита) входит в
  множество. Пример: цель `sap` учитывается, только если тип действия = sap.
- **`only_if` есть только у `Units`**: условие задаётся внутри одного юнита, а у обычного `Dict`-действия
  такого параметра нет. Боту без юнитов, которому нужна одна условная голова, подойдёт `Units(1, ...)`
  с этой головой как компонентом.
- Решения одного места раскладываются на K «решающих»: каждый юнит — один, все не юнитовые части
  вместе — ещё один. Режимы `algorithm.ratio_mode` / `unit_trace` / `entropy_reduction` (по умолчанию
  `auto`: `per_unit`, `joint`, `mean_valid` при K > 1; при K = 1 все режимы сводятся к `joint`, кроме явного `unit_trace: none`) и итоги
  эксперимента по K=8 и K=128 — в `docs/benchmarks.md`.

Модель юнитов: энкодер отдаёт `EncoderOutput(latent, aux={"units": [B, U, E]})`, политика собирает
параметры через `UnitsHead` (`examples/unit_harvest/models.py`):

```python
params = {"base": self.base_head(features),
          "workers": self.workers_head(torch.cat([units, context], dim=-1))}
return make_distribution(self.action_spec, params)
```

`make_distribution(action_spec, params)` (`colosseum.networks.heads` или `colosseum.networks.dist`):
для `Dict`-действия `params` повторяют дерево действия; для `Box` — `{"mean": [B, d], "log_std": [d]}`;
для группы `Units` — словарь, который возвращает `UnitsHead`; для действия без `Dict` — параметры
единственной группы без обёртки (`examples/space_miners/models.py`).

### Маски

Маска — дерево, повторяющее дискретные части действия; нет листа — всё разрешено.

| Пространство | Маска |
|---|---|
| `Discrete(n)` | `bool[n]` |
| `MultiDiscrete([A1, ...])` | `bool[A1 + ...]` |
| `Box` | нет |
| `Units` | `{"unit": bool[U], "action": bool[U, сумма размеров дискретных компонентов]}` |

- **Маска — массив `dtype=bool`** (`np.ones(n, bool)`). Маска другого типа (`int`, `float`) — ошибка
  контракта, даже если в ней только 0 и 1: фреймворк не приводит типы молча, потому что `int`-массив
  легко спутать со списком индексов разрешённых действий, а `float` — с вероятностями.
- У ходящего места дискретный лист без единого разрешённого действия — ошибка контракта. Держите
  хотя бы одно «безопасное» действие (стоять, пас).
- В `Units` строка `unit=False` — юнита нет; пустая строка компонента у существующего юнита делает
  этот компонент невалидным без ошибки. Пример маски внутри юнитов — `push` в `space_miners`
  разрешён, только если астероид рядом.
- Модель не применяет маски сама: `ComposedModel` накладывает маску на распределение политики.

## 8. Модель

`networks` в конфиге: либо `model_class` (монолитная модель, подкласс
`colosseum.networks.model.PolicyModel` с методами `step` и `unroll`), либо части `ComposedModel`:

```yaml
networks:
  encoder_class: ...         # obs-дерево -> Tensor [B, D] или EncoderOutput(latent, aux)
  core: null                 # или {class: colosseum.networks.cores.LSTMCore, kwargs: {...}}
  policy_class: ...          # (features, aux) -> Distribution (make_distribution)
  value_class: ...           # features [B, D (+ G)] -> [B]
  critic_encoder_class: null # global_state -> [B, G]
  kwargs: {}                 # энкодеру, критик-энкодеру, головам и model_class (не ядру)
```

Конструкторы получают по имени то, что явно объявляют в сигнатуре: энкодер, критик-энкодер, политика и
`model_class` — `observation_space`, `action_space`, `global_state_space`, `action_spec`; головы
политики и value — `in_dim` (выход ядра, у value плюс выход критик-энкодера); все они ещё получают
`networks.kwargs`. Ядро получает только `input_dim` = `latent_dim` энкодера и свои `core.kwargs`
(пространства и `networks.kwargs` ему не передаются). Поэтому одни и те же классы работают для ролей с разными
размерами (`examples/predator_prey/models.py`). Ядра (`colosseum.networks.cores`): `null` (без
памяти), `LSTMCore` / `GRUCore` (`hidden_size`, `num_layers`), `WindowAttentionCore` (`d_model`,
`window`, `num_heads`, `num_layers`; пример — `configs/examples/tic_tac_toe_attention.yaml`).

Свой `model_class` с нормализаторами, переопределяющий `PolicyModel.update_normalizers(obs, global_state)`, должен принимать `obs=None`: во время прогрева критика (`agents.<id>.init.critic_warmup_steps`, `docs/LEAGUE_GUIDE.md` раздел 8) APPO вызывает `update_normalizers(None, global_state)`, чтобы обновлялись только нормализаторы пути value (глобального состояния), а статистика нормализаторов наблюдений не менялась. Базовая реализация пропускает нормализатор, у которого дерево-источник равно `None`.

## 9. Проверка

```bash
colosseum validate -c configs/examples/<game>.yaml
colosseum train -c configs/examples/<game>.yaml --set training.total_timesteps=20000 --set run.name=smoke
```

Ошибки контракта приходят как `EnvContractError` с контекстом «воркер, среда, место, шаг эпизода,
вариант» (`worker 3, env 1, seat 2, episode step 7, layout 4p: ...`) — по нему видно, какое правило
нарушено. В `validate` контекст начинается с `validate`, порядок тот же, в том числе у полной
проверки пространств (`space.contains`): `validate, seat 2, episode step 7, layout 4p`.
Исключение из `reset`/`step` самой среды `validate` показывает одной строкой `Config error: ...` с
вариантом и шагом.

## 10. `infos` для скриптовых ботов

Скриптовый бот (`colosseum.players.ScriptedBot`, `docs/LEAGUE_GUIDE.md`) получает в `act(obs, mask, info)` наблюдение и маску своего места и `info = StepResult.infos.get(seat)` последнего шага (`None`, если среда ничего не положила). Через `infos` среда отдаёт боту «сырое» состояние, которое нейросети не нужно: координаты, ссылки на объекты движка, словари.

```python
return StepResult(acting={0, 1}, obs=..., action_masks=..., rewards=...,
                  infos={s: {"enemy_base": self._bases[1 - s].copy()} for s in (0, 1)})
```

- Ключ — номер места, значение — что угодно (обычно `dict`). Фреймворк его не проверяет и не копирует; в чанки и BC-данные он не попадает. Читают его только места со скриптовыми ботами (игроки и учителя DAgger), в обучении, `eval` и `record` одинаково.
- `rollout.vec_env: sync`: объект передаётся по ссылке, цены нет. Не меняйте его после `step`: бот может держать ссылку.
- `rollout.vec_env: subprocess`: `StepResult` целиком пиклится из процесса среды в воркер на каждом шаге — вместе с `infos`, даже если ботов в партии нет. Цена — размер `infos`: крупные объекты (вся карта, история) на каждом шаге заметно замедляют воркер. Кладите туда только нужное ботам и компактно (numpy-массивы вместо списков словарей) или включайте их флагом среды в `env.kwargs`, который ставится только в конфигах с ботами.

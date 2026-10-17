# Лига, игроки и warm start

Как задать, с кем играет обучаемый агент, и как начать обучение не с нуля. Всё настраивается в одном конфиге. `colosseum validate -c cfg.yaml` печатает итоговую смесь соперников каждого агента и отчёт `init` — проверяйте им каждый рецепт (раздел 13).

## 1. Понятия

- **Агент** — запись в `agents:`. Поле `kind`:
  - `trainable` (по умолчанию) — обучаемая линия со своим лёрнером;
  - `scripted` — бот на правилах, подкласс `colosseum.players.ScriptedBot` (раздел 3);
  - `frozen` — фиксированные веса из файла: чекпоинт другого запуска, прошлый сабмит, BC-сеть (раздел 4).
- Если в конфиге нет ни одного обучаемого агента (нет секции `agents` или в ней только scripted и frozen), неявно добавляется обучаемый `agent_0` с глобальными настройками. Добавление бота не заставляет переписывать минимальный конфиг.
- **Игрок** — тот, кто сидит за местом: latest обучаемого агента, его снимок `ckpt_v<N>` или scripted / frozen агент. В конфиге и CLI игроков называют только по имени агента; снимки текущего запуска выбирает матчмейкер, а конкретный снимок другого запуска подключается как frozen-агент с `path`.
- **Владелец данных** — обучаемый агент, для которого собирается среда (обучаемые агенты по очереди, в порядке конфига). Данные собирают только места с latest обучаемых агентов; снимки, scripted и frozen агенты не собирают никогда.
- **Якоря** — scripted и frozen агенты, которые играют соперниками (`matchmaking.anchors`).

## 2. Один агент в self-play

Минимальный конфиг (без `agents` и без `matchmaking`) — это один `agent_0` и смесь по умолчанию:

```yaml
matchmaking:
  opponents: {latest: 0.7, snapshots: 0.2, rivals: 0.0, anchors: 0.1}
  pfsp: {weighting: hard, exponent: 2.0, halflife_games: 200}
```

Для каждой команды соперников независимо разыгрывается категория её ядра:
- `latest` — текущие веса владельца;
- `snapshots` — хранимые снимки (свои; в асимметричной игре — и снимки соперника, раздел 6), по PFSP;
- `rivals` — latest других обучаемых агентов (арена, раздел 5);
- `anchors` — якоря (раздел 3).

Пустая сейчас категория (снимков ещё нет, якорей нет) отдаёт свою долю остальным пропорционально. Если пусты все категории с положительной долей (например, `snapshots: 1.0` до первого снимка), ядро берётся из `latest`, а без него — из якорей; такие команды помечаются категорией `fallback`, в лог пишется одно предупреждение на агента и вариант. Ядро садится на одно место команды; остальные места команды заполняются по `teammates` (`self` — ядру). PFSP: для каждой пары «latest владельца против игрока X» в каждом варианте партии ведётся EMA счёта x (победа 1, ничья 0.5; `halflife_games` — полураспад в партиях; до первой игры 0.5), вес кандидата — `hard` (1 − x)^p, `balanced` x(1 − x) или `uniform`.

**Конфиги SP2** работают без правок: `mode`, `self_play_ratio`, `latest_prob`, `pfsp_exponent` переводятся (`spr` = 1 при `mode: self_play`, иначе `self_play_ratio`; `latest = spr · latest_prob`, `snapshots = spr · (1 − latest_prob)`, `rivals = 1 − spr`, `anchors = 0`, `pfsp = {weighting: hard, exponent: pfsp_exponent}`) с одним предупреждением, которое показывает получившуюся секцию. Старые ручки вместе с `opponents` или `pfsp` — ошибка конфига. Недостающие старые ручки берут дефолты SP2 (`mode: self_play`, `self_play_ratio: 0.5`, `latest_prob: 0.5`, `pfsp_exponent: 1.0`), поэтому переведённый конфиг получает `exponent: 1.0`, а не новый дефолт 2. Так же переводятся `checkpoint.pool_size` → `keep_last` и `training.kickstart_*` → секция `kickstart`. Примеры из `configs/examples/` уже записаны в новой форме. Кроме `team_tag.yaml` (в SP3 добавлен якорь `RandomBot`, раздел 3) и нового `unit_harvest_league.yaml`, это точный перевод их формы SP2 (строка-комментарий над `opponents` показывает старые значения): такой пример ведёт себя так же, как его копия SP2, загруженная в SP3. Но поведение лиги по сравнению с SP2 изменилось (дефолтная смесь, PFSP по снимкам, снимки соперника в асимметричных играх, хранение снимков и др.) — полный список в разделе 16.

## 3. Скриптовый бот и якоря

```python
# my_game/bots.py
import numpy as np
from colosseum.players import ScriptedBot

class Greedy(ScriptedBot):
    def __init__(self, aggr: float = 0.5):            # kwargs из конфига
        self.aggr = aggr

    def reset(self, *, role, seat, layout, rng):     # в начале каждого эпизода, где бот сидит за этим местом
        self.rng = rng                               # засеян из сида эпизода, номера места и имени агента (без training.seed — случайно)

    def act(self, obs, mask, info):                   # numpy-деревья (mask = None, если среда не дала маску); info = StepResult.infos[seat] или None
        legal = np.flatnonzero(mask)
        return int(legal[0] if self.rng.random() < self.aggr else self.rng.choice(legal))
```

- Экземпляр создаётся на тройку (агент, среда, место) при первом появлении бота за этим местом и живёт между эпизодами; память эпизода бот держит в `self`. `self.game_spec` (`GameSpec`) выставляет фреймворк до первого `reset`.
- Действие проходит ту же проверку легальности, что и действие сети. Нелегальное действие или исключение в боте — `PlayerError` с контекстом «воркер, среда, место, шаг эпизода, вариант, агент».
- `info` — `StepResult.infos[seat]` среды: туда среда кладёт «сырое» состояние для ботов (`docs/ENV_GUIDE.md`, раздел 10).
- Встроенный `colosseum.players.RandomBot` — случайное легальное действие.

```yaml
agents:
  main: {}                                   # kind: trainable
  greedy:
    kind: scripted
    class: my_game.bots.Greedy
    kwargs: {aggr: 0.7}
    roles: [player]                          # по умолчанию — все роли игры (пространства ролей могут различаться)
  random:
    kind: scripted
    class: colosseum.players.RandomBot

matchmaking:
  opponents: {latest: 0.6, snapshots: 0.2, rivals: 0.0, anchors: 0.2}
  anchors: {greedy: 2.0, random: 1.0}        # веса внутри категории anchors
```

`anchors: null` (по умолчанию) — все scripted и frozen агенты конфига с равными весами; `anchors: []` — никого (например, бот нужен только как учитель kickstart); список имён — равные веса. Якорь садится ядром команды соперников; места его команды, роли которых он играет, тоже получают его (`teammates: self`). Пример: `configs/examples/team_tag.yaml` — якорь `RandomBot` против пассивной политики «тянуть на ничью»; `configs/examples/unit_harvest_league.yaml` — бот игры и `RandomBot`.

## 4. Замороженный агент (прошлый сабмит)

```yaml
agents:
  main: {}
  prev_sub:
    kind: frozen
    path: runs/sub12/checkpoints/main/ckpt_v9000   # папка чекпоинта: роли, сигнатура и networks из meta.json
  bc_net:
    kind: frozen
    path: bc.pt                                    # .pt: сеть — глобальная networks с этим override
    networks: {kwargs: {hidden: 128}}
    roles: [player]                                # для .pt; по умолчанию все роли с одинаковыми пространствами
```

Frozen-агент может быть якорем, источником `init.from` (раздел 8), учителем kickstart и игроком в `eval` / `record` / `tournament` по имени. Архитектура своя (не обязана совпадать с обучаемым агентом). Для папки чекпоинта `roles` и `networks` указывать нельзя. Чекпоинты SP2 подходят: их `meta.json` содержит сеть, роли и сигнатуру. `networks` и `roles` frozen-записи относятся только к её собственному `path`: `-a bc_net=other.pt` в `eval`, `record` (`--player` / `--against`) и `tournament` их не наследует — такой `.pt` строится с глобальной `networks` и всеми ролями игры (для имени обучаемого агента, как раньше, — с его архитектурой и ролями).

## 5. Лига из нескольких агентов

```yaml
agents:
  small: {networks: {kwargs: {hidden: 64}}}
  large: {networks: {kwargs: {hidden: 256}}, algorithm: {learning_rate: 1.0e-4}}

matchmaking:
  opponents: {latest: 0.4, snapshots: 0.2, rivals: 0.3, anchors: 0.1}
```

`rivals` — latest другого обучаемого агента, выбранного по PFSP; эти места тоже собирают данные (арена: один шаг среды кормит двух лёрнеров). `snapshots` — снимки всех агентов, играющих роли команды соперников. Пример: `configs/examples/tic_tac_toe_multi.yaml`.

## 6. Асимметричная игра

В `predator_prey` агент `hunter` играет роль охотника, `prey` — жертвы (`agents.<id>.roles`). Для владельца-охотника команда жертв играется только агентами-жертвами, поэтому:
- `latest` — latest жертвы (выбор по PFSP, если жертв несколько), а доля `rivals` прибавляется к `latest`;
- `snapshots` — снимки жертвы: охотник встречает и прошлые версии соперника;
- `anchors` — якоря с ролью жертвы, например `prey_bot: {kind: scripted, class: ..., roles: [prey]}`.

Проверка конфига требует, чтобы каждую роль каждого варианта играл хотя бы один обучаемый агент или якорь владельца.

## 7. Кооператив со скриптовым напарником

- **Роль, которую агент не играет.** Если в команде есть место роли, которой нет у ядра (например, `pilot` и `gunner`, а обучаемый агент играет только `pilot`), место получает latest обучаемого агента с этой ролью, а если такого нет — якорь владельца с этой ролью (по весам якорей):

```yaml
agents:
  pilot: {roles: [pilot]}
  gunner_bot: {kind: scripted, class: my_game.bots.Gunner, roles: [gunner]}
```

- **Смешанные напарники.** С `teammates: mixed` каждое место роли ядра достаётся ядру с вероятностью `teammate_self_prob`, иначе — равномерно одному из кандидатов: latest других обучаемых агентов с этой ролью, снимки агента ядра, якоря владельца с этой ролью. Так обучаемый агент учится играть и с ботом-напарником. Пример формы: `configs/examples/coop_buttons.yaml`.

## 8. Конвейер: `record` → `bc` → `init` → kickstart

Весь конвейер идёт по одному конфигу. Скопируйте `configs/examples/unit_harvest_league.yaml` в `harvest.yaml` и заранее назовите в нём выход BC — файла ещё нет:

```yaml
agents:
  main:
    init: {from: bc_net, critic_warmup_steps: 30}   # RL стартует с BC-весов
    kickstart: {teacher: greedy}                     # бот продолжает учить (DAgger)
  bc_net: {kind: frozen, path: bc.pt}                # появится на шаге 2
  # greedy и random — как в примере
```

```bash
# 1. Бот играет сам с собой; все его решения — данные BC (папка на роль + record.json)
colosseum record -c harvest.yaml --player greedy --num-matches 300 --output data/greedy --seed 0
# 2. BC-сеть агента main
colosseum bc -c harvest.yaml --agent main --data data/greedy --output bc.pt --epochs 5 --seed 0
# 3. RL с BC-весов: прогрев критика, kickstart от бота (DAgger), лига с якорями
colosseum train -c harvest.yaml --set run.name=harvest-league
# 4. Оценка: обученный агент против бота, random и BC-сети
NEW=$(ls -d runs/harvest-league/checkpoints/main/ckpt_v* | sort -V | tail -1)
colosseum eval -c harvest.yaml -a trained=$NEW -a greedy -a random -a bc_net --num-matches 100 --deterministic
```

`record`, `bc` и `eval` проверяют только то, чем пользуются: среду, модели обучаемых агентов и scripted / frozen агентов, которых сажают за стол (`--player`, `--against`, `-a <имя>`). Источники `init`, учителя kickstart, прогрев критика, матчмейкинг (якоря, смесь, свой матчмейкер) и остальные frozen-агенты им не нужны и не проверяются, поэтому `bc_net` и `init.from` могут указывать на ещё не созданный `bc.pt`. `validate` и `train` проверяют всё, и ошибки в этих частях покажут они. Пример можно использовать и без правок, задав тёплый старт в командной строке: `colosseum train -c configs/examples/unit_harvest_league.yaml --set agents.main.init.from=bc.pt --set agents.main.init.critic_warmup_steps=30 --set agents.main.kickstart.teacher=greedy`; `--set` принимают и `record`, `bc`, `eval`.

- **`record`.** `--player` / `--against` — имя scripted или frozen агента либо `name=path` (папка чекпоинта или `.pt`). Без `--against` игрок занимает все места, и записываются все места. Варианты, которые игрок не может заполнить один (не играет какую-то их роль), без `--layout` пропускаются с одной INFO-строкой в логе; ошибка конфига с подсказкой «добавьте `--against`» — только если не осталось ни одного варианта или такой вариант задан явно через `--layout`. С `--against` каждый соперник даёт пару с игроком (ротация как в `eval`, `--num-matches` на пару и вариант), записываются только места игрока. Эпизод места лежит в файле непрерывно, `dones` корректны; `record.json` описывает запись и содержит сводку исходов.
- **`bc`.** `--data` можно повторять; для папки с `record.json` берутся подпапки ролей агента. `--seed` фиксирует начальные веса и порядок минибатчей.
- **`init`** (секция агента или глобальная): `from` — `.pt`, папка чекпоинта, папка запуска (последний чекпоинт агента с тем же id) или имя frozen-агента; грузятся только веса (версия политики 0, новый оптимизатор). `strict: false` грузит тензоры с совпавшими именем и формой и перечисляет остальные в логе и в выводе `validate`. При `training.resume_from` resume важнее: восстановленный агент игнорирует `init` (запись в лог).
- **Прогрев критика** `critic_warmup_steps: N`: первые N шагов обучения обновляется только value-путь (value-голова и `critic_encoder`), лоссы политики, энтропии и kickstart выключены, статистика нормализаторов наблюдений заморожена — политика лёрнера побитово остаётся BC-политикой, пока критик догоняет; нормализатор глобального состояния критика (путь value) продолжает учиться, чтобы входы критика не «прыгнули» в конце прогрева. Воркеры, как и в SP2, играют своей случайной инициализацией до первой синхронизации весов (первые секунды; V-trace это учитывает), после неё — BC-политикой.
- **Kickstart** (`kickstart: {teacher, lambda, decay_steps, kl}`): `teacher` — имя frozen или scripted агента либо путь (`.pt` или папка чекпоинта). Нейросетевой учитель — KL по решающим (`kl: forward | reverse`); голый путь к `.pt` строится с архитектурой ученика (как в SP2), а frozen-агент (для `.pt` — с его `networks`) и папка чекпоинта приносят свою архитектуру; рекуррентный учитель — только с той же раскладкой состояния, что у ученика. Скриптовый учитель — DAgger: на каждом ходе собирающего места воркер спрашивает бота, его действие идёт в чанк, лёрнер добавляет `λ · mean(−log π(a_учителя))`. λ затухает линейно за `decay_steps` шагов обучения, начиная после прогрева. Имя обучаемого агента учителем быть не может: нужный снимок подключается как frozen-агент с `path`.

Сравнение конвейера с обучением с нуля на равном бюджете — `docs/benchmarks.md`, «Конвейер против обучения с нуля». На `unit_harvest` (3 сида, 160 000 шагов сред) выигрыша нет: обе политики упираются в один потолок — почти одни ничьи с ботом игры (средний счёт против него 0.483 у конвейера и 0.492 с нуля) и 100 % побед над `random`, а конвейер стоит ≈ 129 с против ≈ 92 с. Эта игра слишком проста для тёплого старта; он должен окупаться на играх, где обучение с нуля за бюджет не доходит до уровня бота. Замер дефолтов APPO при `Units` (`ratio_mode: auto` = `per_unit`, `unit_trace: auto` = `joint`) по правилу на нескольких сидах их не поменял — см. `docs/benchmarks.md`, «Дефолты APPO при `Units`»; на играх с множеством юнитов `ratio_mode: joint` можно задать явно.

## 9. Расписания долей

Любая доля `opponents` и любой вес якоря — число или кусочно-линейное расписание по глобальным env-шагам запуска (до первой точки и после последней значение постоянное):

```yaml
matchmaking:
  opponents:
    latest: {0: 0.8, 2_000_000: 0.5}
    snapshots: {0: 0.0, 2_000_000: 0.3}
    rivals: 0.0
    anchors: {0: 0.2, 2_000_000: 0.2}
  anchors:
    greedy: {0: 1.0, 1_000_000: 0.2}    # бот важен в начале
    random: 1.0
```

Проверка конфига требует, чтобы в каждой точке расписания у каждой команды соперников была хотя бы одна структурно непустая категория с положительной долей.

## 10. Override на агента

`agents.<id>.matchmaking` переопределяет `opponents`, `anchors`, `pfsp`, `layouts`, `teammates`, `teammate_self_prob`; `shuffle_seats` и `matchmaker_class` — только глобальные. Слияние с глобальной секцией: `opponents` и `pfsp` — по ключам (заданная доля заменяет глобальную, остальные доли остаются; расписание доли заменяется целиком), а `anchors`, `layouts` и `teammates` / `teammate_self_prob` заменяют глобальное значение целиком (например, `layouts: {4p: 1}` агента не добавляется к глобальному `{2p: 1}`, а заменяет его). `init` и `kickstart` устроены как остальные секции агента (`networks`, `algorithm`, `learner`): глобальная секция — дефолт для всех обучаемых агентов, `agents.<id>.init` / `.kickstart` сливаются с ней по ключам. Доли `opponents` — относительные веса: они нормируются по заполнимым категориям (итог показывает `validate`). `--set` не заходит внутрь карты якорей или расписания: задавайте значение целиком, например `--set 'matchmaking.anchors={greedy: 2}'` (в кавычках: в значении есть пробел).

```yaml
agents:
  main: {}
  explorer:
    matchmaking:
      opponents: {latest: 0.3, snapshots: 0.6, rivals: 0.1, anchors: 0.0}
      pfsp: {weighting: balanced}
```

## 11. Свой матчмейкер

```python
# my_game/league.py
from colosseum.core.types import FIXED_NETWORK_ID, LATEST_NETWORK_ID, SOURCE_OWNER, Lineup, SeatAssignment
from colosseum.league import BaseMatchmaker, MatchmakerContext


class BotFirst(BaseMatchmaker):
    """Первые 500 000 шагов — только против бота greedy, потом — против последнего снимка владельца."""

    def __init__(self, context: MatchmakerContext) -> None:
        super().__init__(context)

    def lineup_for(self, owner: str) -> Lineup:
        me = SeatAssignment(owner, LATEST_NETWORK_ID, collect=True, source=SOURCE_OWNER)
        snapshots = self.context.snapshots(owner)
        if self.context.env_steps() < 500_000 or not snapshots:
            other = SeatAssignment("greedy", FIXED_NETWORK_ID, collect=False, source="anchors")
        else:
            other = SeatAssignment(owner, snapshots[-1], collect=False, source="snapshots")
        return Lineup("2p", [me, other] if self.context.rng.random() < 0.5 else [other, me])
```

```yaml
matchmaking:
  matchmaker_class: my_game.league.BotFirst
```

`MatchmakerContext` даёт `spec`, агентов с видом и ролями (`agents`), обучаемых агентов (`trainable`), итоговый `matchmaking` каждого агента, `snapshots(agent)`, `pfsp_score(layout, owner, (agent, network))`, `env_steps()`, `anchors(owner)` и общий `rng`. `on_result(result)` получает каждый `MatchResult`. Фреймворк проверяет каждый состав (вариант, число мест, роли, существование игроков и снимков, `collect` только у latest обучаемых); нарушение — ошибка с именем класса и составом.

Проверка конфига и с `matchmaker_class` проверяет встроенную секцию `opponents` (каждая команда соперников должна быть заполнима хотя бы одной категорией с положительной долей), хотя свой матчмейкер её не читает: оставляйте её выполнимой — например, по умолчанию или `opponents: {latest: 1.0, snapshots: 0.0, rivals: 0.0, anchors: 0.0}`.

## 12. Хранение снимков

```yaml
checkpoint:
  interval: 1000        # снимок каждые N шагов обучения
  keep_last: 20         # последние 20 (бывший pool_size)
  keep_every: 10        # и каждый 10-й (версия кратна interval · keep_every); 0 = выкл.
  keep_top_k: 0         # SP4: > 0 — ещё k лучших по нижней границе рейтинга (нужен match_log)
  top_k_quantile: 0.1
  top_k_min_games: 200
  max_snapshots: 100    # SP4: потолок на агента; null — без потолка; не задан и keep_last > 100 — keep_last
  save_optimizer: true
```

Финальный снимок текущего запуска не вытесняется никогда; финальные снимки запусков, из которых этот возобновлён, становятся обычными снимками. `trainer_state.pt` есть только у снимков в окне `keep_last` (resume нужен только с последних). Удалённый снимок пропадает из кандидатов матчмейкера и PFSP-статистики, воркеры выгружают его модель. Resume из папки запуска (`training.resume_from: runs/<run>`) переносит хранимые снимки в новый запуск (hardlink или копия), так что пул соперников переживает resume; из папки чекпоинта или `.pt` пул начинается пустым. Resume из снимка без `trainer_state.pt` (вне окна `keep_last`, например снимок `keep_every`) восстанавливает только веса и версию политики: оптимизатор новый, прогрев критика (`critic_warmup_steps`) проходит ещё раз, а λ kickstart затухает с начала; если это не нужно, задайте для такого resume `init.critic_warmup_steps: 0` (глобально или у агента) и нужную секцию `kickstart`. Перенесённые hardlink-снимки — те же файлы, что в старом запуске: не редактируйте их на месте ни в одном из двух запусков (удалять папки можно). С SP4 `keep_top_k` хранит k снимков с наибольшей нижней границей BT-рейтинга (с защитой новых снимков до `top_k_min_games` игр), закреплённые снимки (`colosseum pin`, `tournament --pin`) не вытесняются никогда, а `max_snapshots` ограничивает число снимков агента, и при превышении первыми уходят снимки только по `keep_every`. Правило целиком, порядок вытеснения и закрепление — [`ARENA_GUIDE.md`](ARENA_GUIDE.md), раздел 5.

## 13. Что печатает `validate`

```bash
colosseum validate -c configs/examples/unit_harvest_league.yaml
```

```text
  OK: agent 'main' (trainable)
  OK: agent 'greedy' (scripted)
  OK: agent 'random' (scripted)
  agent 'main': opponents by pfsp hard (exponent 2, half-life 200 games), teammates self
    2p, team 0: step 0: latest 0.50, snapshots 0.20, rivals 0.00, anchors 0.30
    2p, team 1: step 0: latest 0.50, snapshots 0.20, rivals 0.00, anchors 0.30
    anchors: greedy 1, random 1
  match log: on (detail window 512 MB, min free disk 2048 MB, compaction every 60 s)
  ratings: Bradley-Terry over train and pulse matches, refit every 60 s, anchor 'random' (the first RandomBot agent); bootstrap 200
  agent 'main': snapshots: newest 10 (keep_last, with trainer state); every 1000th version (keep_every 10); the final one of this run; pinned; top-k off; at most 100 per agent (over the cap: keep_every snapshots oldest first)
  agent 'greedy' (scripted examples.unit_harvest.bots.HarvestBot): roles ['player']; played 16 decisions in layouts ['2p']
  agent 'random' (scripted colosseum.players.RandomBot): roles ['player']; played 16 decisions in layouts ['2p']
Config is valid.
```

С warm start из README (после `record` и `bc`) добавляется строка `init`:

```bash
colosseum validate -c configs/examples/unit_harvest_league.yaml --set agents.main.init.from=bc.pt \
  --set agents.main.init.critic_warmup_steps=30 --set agents.main.kickstart.teacher=greedy
```

```text
  OK: agent 'main' (trainable)
  OK: agent 'greedy' (scripted)
  OK: agent 'random' (scripted)
  agent 'main': opponents by pfsp hard (exponent 2, half-life 200 games), teammates self
    2p, team 0: step 0: latest 0.50, snapshots 0.20, rivals 0.00, anchors 0.30
    2p, team 1: step 0: latest 0.50, snapshots 0.20, rivals 0.00, anchors 0.30
    anchors: greedy 1, random 1
  match log: on (detail window 512 MB, min free disk 2048 MB, compaction every 60 s)
  ratings: Bradley-Terry over train and pulse matches, refit every 60 s, anchor 'random' (the first RandomBot agent); bootstrap 200
  agent 'main': snapshots: newest 10 (keep_last, with trainer state); every 1000th version (keep_every 10); the final one of this run; pinned; top-k off; at most 100 per agent (over the cap: keep_every snapshots oldest first)
  agent 'main': init from bc.pt (strict): 20 tensors
  agent 'greedy' (scripted examples.unit_harvest.bots.HarvestBot): roles ['player']; played 16 decisions in layouts ['2p']
  agent 'random' (scripted colosseum.players.RandomBot): roles ['player']; played 16 decisions in layouts ['2p']
Config is valid.
```

Для каждого обучаемого агента и варианта партии — доли категорий для каждой команды соперников (`team N` — команда, против которой играет владелец) после перераспределения пустых в начальной и конечной точке расписаний (здесь расписаний нет, поэтому только `step 0`) и список якорей с весами; настройки журнала матчей и рейтингов (якорь шкалы) и итоговое правило хранения снимков каждого агента (SP4, [`ARENA_GUIDE.md`](ARENA_GUIDE.md)); с `evaluation.enabled: true` — строка `pulse:` с полем, раскладками и числом матчей на снимок; для скриптовых агентов — роли и сколько решений бот сделал в пробной игре; для `init` — откуда берутся веса и (при `strict: false`) какие тензоры не загрузились.

## 14. Метрики и рейтинги

- Scripted и frozen агенты — отдельные сущности в ELO, матрице win-rate и `role_win_rates` (`ratings.json`): «main против greedy» видно сразу. В этих таблицах SP2/SP3 снимки по-прежнему засчитываются своему агенту; онлайн-ELO — индикатор прогресса, не рейтинг для отбора.
- **Рейтинги снимков (SP4).** Из журнала матчей (`runs/<name>/matches.db`) считается рейтинг Bradley–Terry с доверительными интервалами для каждой сущности: версии агента (`main@v1200`), scripted и frozen агенты — по имени. Живой рейтинг пересчитывается во время обучения раз в `ratings.refit_interval_sec` и пишется в `ratings.json` → `bt`, в консоль (`bt <рейтинг> ± <половина 95 % интервала>` последней корзины агента) и в записи `ratings` файла `metrics.jsonl`; отчёт и дашборд (`colosseum report`, `colosseum dashboard`) показывают его так: пока запуск идёт — живой рейтинг из `ratings.json` (интервалы Лапласа), после остановки — подгонка по журналу с bootstrap-интервалами. Ноль шкалы — якорь (по умолчанию первый `RandomBot` конфига). Сущности, шкала, группы ролей и как читать интервалы — [`ARENA_GUIDE.md`](ARENA_GUIDE.md), раздел 3.
- PFSP-таблица (счёт и число игр latest владельца против каждого игрока) — в `ratings.json`, по вариантам. Записи `ratings` в `metrics.jsonl` с SP4 несут вместо неё сводку `pfsp_summary` по владельцу (`players`, `min_score`, `max_score`); полная таблица — только в `ratings.json`.
- ELO, win-rate, `wr_vs_past` и PFSP с SP4 переживают resume из папки запуска: состояние лиги пишется в `runs/<name>/league_state.json` на каждом такте метрик и в конце запуска, и новый запуск загружает файл родителя (у папки SP3 его нет — статистика начинается пустой, INFO в логе). Состояние своего матчмейкера (`on_result`) не сохраняется.
- Доли **сыгранных** эпизодов по категориям и якорям, по агентам и вариантам — в `metrics.jsonl` (записи `episodes`, поле `opponents.<вариант>`: `teams` — число команд соперников, разыгранных для агента как владельца данных; `categories` — доли `latest` / `snapshots` / `rivals` / `anchors` / `fallback`; `anchors` — доля каждого якоря среди всех команд) и в WandB (`episodes/<agent>/opponents/<вариант>/...`). Пример из прогона конвейера README (запись сокращена до этих полей):

```json
{"kind": "episodes", "agent": "main", "env_steps": 95488, "opponents": {"2p": {"teams": 256, "categories": {"latest": 0.421875, "snapshots": 0.3046875, "rivals": 0.0, "anchors": 0.2734375, "fallback": 0.0}, "anchors": {"greedy": 0.125, "random": 0.1484375}}}}
```

  Сыгранные доли могут отличаться от заданных (разыгранных): состав применяется в конце эпизода и держится до следующего обновления составов (`rollout.match_refresh_interval_sec`), а обновление заменяет ещё не сыгранные; короткие партии при этом представлены чаще. Пример: в `team_tag` разыгранная доля якоря 0.2, а сыгранная ≈ 0.36 — партии против случайной команды короткие (`docs/benchmarks.md`, «team_tag: доля якоря»). Место, чей снимок был удалён до начала партии и которое поэтому сыграло latest, всё равно засчитывается в `snapshots`.
- В W/D/L записей `episodes` у соперников есть тип `anchor` (рядом с `latest`, `past`, `arena`).

## 15. Ограничения

- Распределённый режим (`run-learner` / `run-workers`) — только latest-веса обучаемых агентов (объём SP2) и только игры, где каждый агент играет все роли включённых вариантов: scripted и frozen агенты, `init`, прогрев критика, свой матчмейкер, override `matchmaking` и `kickstart` на агента и учитель kickstart, кроме одного глобального `.pt` с архитектурой ученика, там — ошибка конфига с пометкой SP5; любая смесь соперников сводится к «только latest», а `teammates: mixed` игнорируется (одно предупреждение). Журнал матчей и рейтинги снимков распределённые роли не пишут; `evaluation.enabled: true` и `checkpoint.keep_top_k > 0` там — ошибка конфига с пометкой SP5, а `max_snapshots` работает, вытесняя снимки от старых к новым (FIFO, без рейтингов).
- Чего нет в SP4 ([`ARENA_GUIDE.md`](ARENA_GUIDE.md) описывает то, что есть): распределённого турнира, оценки во время обучения (pulse) и журнала матчей на других машинах — до SP5; турниров Swiss / king-of-the-hill, SPRT-гейта сабмита и α-Rank / Nash averaging (при циклах — матрица результатов и худший соперник); модели ничьих Davidson и точного Plackett–Luce для FFA (ничья = 0.5, FFA — по парам команд); выбора top-k по разнообразию стратегий (top-k — по нижней границе рейтинга).
- Рекуррентный учитель kickstart с другой раскладкой состояния, батчевый API скриптовых ботов, эксплойтеры — позже, по необходимости (свой матчмейкер закрывает экзотику).

## 16. Совместимость с SP2

Конфиги и чекпоинты SP2 загружаются и работают (раздел 2, раздел 4): старые ручки переводятся, а не отвергаются. Перевод конфига точный, но поведение лиги отличается от SP2 — SP3 сменил дефолты и правила матчмейкера:
- **Конфиг без ручек матчмейкинга** получает новую смесь `{latest: 0.7, snapshots: 0.2, rivals: 0, anchors: 0.1}` с PFSP `hard`, `exponent: 2` (без якорей их доля перераспределяется: когда снимки есть, фактически latest ≈ 0.78, snapshots ≈ 0.22); в SP2 было 0.5 / 0.5 (latest / чекпоинт) с равномерным выбором чекпоинта.
- **Снимки выбираются по PFSP**, а не равномерно — и в переведённых конфигах SP2 (`pfsp_exponent` становится `pfsp.exponent`). Счёт для PFSP — EMA с полураспадом `halflife_games` (200 партий) по сетевым id, а не накопленные win rate рейтингов SP2 (арена SP2).
- **Асимметричные игры встречают снимки соперника.** Например, в `predator_prey` (и копия SP2, и пример v3: `snapshots: 0.2`) охотник примерно в 20 % своих сред играет против снимков жертвы; в SP2 команду, роли которой владелец не играет, всегда занимал latest другого агента.
- **`mode: self_play` с несколькими обучаемыми агентами** теперь встречает и снимки других агентов (играющих роли команды соперников), а не только свои.
- **`mode: league` разыгрывается на каждую команду соперников**, а не один раз на матч: в вариантах с тремя и более командами состав соперников другой, чем в SP2.
- **Хранение снимков:** FIFO `pool_size` становится `keep_last`, плюс `keep_every: 10` по умолчанию (каждый 10-й снимок хранится всегда, без `trainer_state`) и финальный снимок; диск растёт примерно на `шаги обучения / (checkpoint.interval · keep_every)` снимков (версия снимка — номер шага обучения, а не шаг сред). Resume из папки запуска переносит пул снимков.
- **Один обучаемый агент с `mode: league` и `self_play_ratio: 0.0`** переводится в `opponents: {latest: 0, snapshots: 0, rivals: 1, anchors: 0}`. Других обучаемых агентов нет, категория `rivals` пуста, и проверка конфига отклоняет его (ни одной заполнимой категории с положительной долей); SP2 в этом случае молча играл self-play. Запишите `opponents` явно, например `{latest: 0.8, snapshots: 0.2, rivals: 0.0, anchors: 0.0}`.
- **Путь учителя kickstart** — файл `.pt` или папка чекпоинта. SP2 принимал любой файл весов (например, `teacher.pth`); такой файл нужно переименовать в `.pt`.

Чтобы вернуть смесь SP2 для конфига без ручек, задайте её явно: `opponents: {latest: 0.5, snapshots: 0.5, rivals: 0, anchors: 0}` и `pfsp: {weighting: uniform}`; `checkpoint.keep_every: 0` возвращает чистый FIFO (финальный снимок текущего запуска всё равно хранится).

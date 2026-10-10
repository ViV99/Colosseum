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

**Конфиги SP2** работают без правок: `mode`, `self_play_ratio`, `latest_prob`, `pfsp_exponent` переводятся (`spr` = 1 при `mode: self_play`, иначе `self_play_ratio`; `latest = spr · latest_prob`, `snapshots = spr · (1 − latest_prob)`, `rivals = 1 − spr`, `anchors = 0`, `pfsp = {weighting: hard, exponent: pfsp_exponent}`) с одним предупреждением, которое показывает получившуюся секцию. Старые ручки вместе с `opponents` или `pfsp` — ошибка конфига. Так же переводятся `checkpoint.pool_size` → `keep_last` и `training.kickstart_*` → секция `kickstart`. Примеры из `configs/examples/` уже записаны в новой форме — это точный перевод их формы SP2 (строка-комментарий над `opponents` показывает старые значения), поведение не изменилось. Два отличия от SP2 — в разделе 16.

## 3. Скриптовый бот и якоря

```python
# my_game/bots.py
import numpy as np
from colosseum.players import ScriptedBot

class Greedy(ScriptedBot):
    def __init__(self, aggr: float = 0.5):            # kwargs из конфига
        self.aggr = aggr

    def reset(self, *, role, seat, layout, rng):     # в начале каждого эпизода, где бот сидит за этим местом
        self.rng = rng                               # засеян из сида эпизода и номера места

    def act(self, obs, mask, info):                   # numpy-деревья; info = StepResult.infos[seat] или None
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

Frozen-агент может быть якорем, источником `init.from` (раздел 8), учителем kickstart и игроком в `eval` / `record` по имени. Архитектура своя (не обязана совпадать с обучаемым агентом). Для папки чекпоинта `roles` и `networks` указывать нельзя. Чекпоинты SP2 подходят: их `meta.json` содержит сеть, роли и сигнатуру.

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

```bash
# 1. Бот играет сам с собой; все его решения — данные BC (папка на роль + record.json)
colosseum record -c configs/examples/unit_harvest_league.yaml --player greedy --num-matches 300 --output data/greedy --seed 0
# 2. BC-сеть агента main
colosseum bc -c configs/examples/unit_harvest_league.yaml --agent main --data data/greedy --output bc.pt --epochs 5
# 3. RL с BC-весов: прогрев критика, kickstart от бота (DAgger), лига с якорями
colosseum train -c configs/examples/unit_harvest_league.yaml --set run.name=harvest-league \
  --set agents.main.init.from=bc.pt --set agents.main.init.critic_warmup_steps=30 \
  --set agents.main.kickstart.teacher=greedy
```

- **`record`.** `--player` / `--against` — имя scripted или frozen агента либо `name=path` (папка чекпоинта или `.pt`). Без `--against` игрок занимает все места, и записываются все места. Варианты, которые игрок не может заполнить один (не играет какую-то их роль), без `--layout` пропускаются с одной INFO-строкой в логе; ошибка конфига с подсказкой «добавьте `--against`» — только если не осталось ни одного варианта или такой вариант задан явно через `--layout`. С `--against` каждый соперник даёт пару с игроком (ротация как в `eval`, `--num-matches` на пару и вариант), записываются только места игрока. Эпизод места лежит в файле непрерывно, `dones` корректны; `record.json` описывает запись и содержит сводку исходов.
- **`bc`.** `--data` можно повторять; для папки с `record.json` берутся подпапки ролей агента.
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

`agents.<id>.matchmaking` (deep-merge на глобальную секцию) переопределяет `opponents`, `anchors`, `pfsp`, `layouts`, `teammates`, `teammate_self_prob`; `shuffle_seats` и `matchmaker_class` — только глобальные. Так же устроены `init` и `kickstart`: глобальная секция — дефолт для всех обучаемых агентов, `agents.<id>.init` / `.kickstart` переопределяют.

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
  keep_every: 10        # и каждый 10-й навсегда (версия кратна interval · keep_every); 0 = выкл.
  save_optimizer: true
```

Финальный снимок не удаляется никогда. `trainer_state.pt` есть только у снимков в окне `keep_last` (resume нужен только с последних). Удалённый снимок пропадает из кандидатов матчмейкера и PFSP-статистики, воркеры выгружают его модель. Resume из папки запуска (`training.resume_from: runs/<run>`) переносит хранимые снимки в новый запуск (hardlink или копия), так что пул соперников переживает resume; из папки чекпоинта или `.pt` пул начинается пустым. Перенесённые hardlink-снимки — те же файлы, что в старом запуске: не редактируйте их на месте ни в одном из двух запусков (удалять папки можно). Лучшие снимки по рейтингу (top-k) — в SP4; до тех пор держите их через `keep_every` или как frozen-агентов.

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
  agent 'main': init from bc.pt (strict): 20 tensors
  agent 'greedy' (scripted examples.unit_harvest.bots.HarvestBot): roles ['player']; played 16 decisions in layouts ['2p']
  agent 'random' (scripted colosseum.players.RandomBot): roles ['player']; played 16 decisions in layouts ['2p']
Config is valid.
```

Для каждого обучаемого агента и варианта партии — доли категорий для каждой команды соперников (`team N` — команда, против которой играет владелец) после перераспределения пустых в начальной и конечной точке расписаний (здесь расписаний нет, поэтому только `step 0`) и список якорей с весами; для скриптовых агентов — роли и сколько решений бот сделал в пробной игре; для `init` — откуда берутся веса и (при `strict: false`) какие тензоры не загрузились.

## 14. Метрики и рейтинги

- Scripted и frozen агенты — отдельные сущности в ELO, матрице win-rate и `role_win_rates` (`ratings.json`): «main против greedy» видно сразу. Снимки засчитываются своему агенту (рейтинги снимков — SP4).
- PFSP-таблица (счёт и число игр latest владельца против каждого игрока) — в `ratings.json`, по вариантам.
- Доли **сыгранных** эпизодов по категориям и якорям, по агентам и вариантам — в `metrics.jsonl` (записи `episodes`, поле `opponents.<вариант>`: `teams` — число команд соперников, разыгранных для агента как владельца данных; `categories` — доли `latest` / `snapshots` / `rivals` / `anchors` / `fallback`; `anchors` — доля каждого якоря среди всех команд) и в WandB (`episodes/<agent>/opponents/<вариант>/...`). Пример из прогона конвейера README (запись сокращена до этих полей):

```json
{"kind": "episodes", "agent": "main", "env_steps": 95488, "opponents": {"2p": {"teams": 256, "categories": {"latest": 0.421875, "snapshots": 0.3046875, "rivals": 0.0, "anchors": 0.2734375, "fallback": 0.0}, "anchors": {"greedy": 0.125, "random": 0.1484375}}}}
```

  Сыгранные доли могут отличаться от заданных (разыгранных): состав применяется в конце эпизода и держится до следующего обновления составов (`rollout.match_refresh_interval_sec`), а обновление заменяет ещё не сыгранные; короткие партии при этом представлены чаще. Пример: в `team_tag` разыгранная доля якоря 0.2, а сыгранная ≈ 0.36 — партии против случайной команды короткие (`docs/benchmarks.md`, «team_tag: доля якоря»). Место, чей снимок был удалён до начала партии и которое поэтому сыграло latest, всё равно засчитывается в `snapshots`.
- В W/D/L записей `episodes` у соперников есть тип `anchor` (рядом с `latest`, `past`, `arena`).

## 15. Ограничения

- Распределённый режим (`run-learner` / `run-workers`) — только self-play на latest (объём SP2): scripted и frozen агенты, `init`, прогрев критика, свой матчмейкер, override `matchmaking` и `kickstart` на агента там — ошибка конфига с пометкой SP5; любая смесь соперников сводится к «только latest» с предупреждением.
- Top-k снимков, рейтинги снимков, OpenSkill / Bradley–Terry, `colosseum tournament` — SP4.
- Рекуррентный учитель kickstart с другой раскладкой состояния, батчевый API скриптовых ботов, эксплойтеры — позже, по необходимости (свой матчмейкер закрывает экзотику).

## 16. Совместимость с SP2

Конфиги и чекпоинты SP2 работают (раздел 2, раздел 4), с двумя отличиями:
- **Один обучаемый агент с `mode: league` и `self_play_ratio: 0.0`** переводится в `opponents: {latest: 0, snapshots: 0, rivals: 1, anchors: 0}`. Других обучаемых агентов нет, категория `rivals` пуста, и проверка конфига отклоняет его (ни одной заполнимой категории с положительной долей); SP2 в этом случае молча играл self-play. Запишите `opponents` явно, например `{latest: 0.8, snapshots: 0.2, rivals: 0.0, anchors: 0.0}`.
- **Путь учителя kickstart** — файл `.pt` или папка чекпоинта. SP2 принимал любой файл весов (например, `teacher.pth`); такой файл нужно переименовать в `.pt`.

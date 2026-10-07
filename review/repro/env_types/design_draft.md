## 3. Предложение по редизайну

Цель — минимальный набор изменений, после которого все 5 типов игр становятся first-class, при сохранении обратной совместимости (старый `BaseEnv` работает через адаптер).

### 3.1 Env API v2: роли, команды, места, acting-set (D1, **M**, prerequisite для D3–D9)

```python
# colosseum/envs/spec.py
@dataclass(frozen=True)
class RoleSpec:
    name: str                              # "player", "hunter", "prey", ...
    observation_space: gym.Space           # Box | Dict (см. D2)
    action_space: gym.Space                # Discrete | Box | Dict | MultiDiscrete(nd) | UnitActions (см. D4)

@dataclass(frozen=True)
class SeatSpec:
    seat: int                              # 0..num_seats-1
    role: str
    team: int                              # команда; FFA: team == seat; solo: 0

class OutcomeKind(str, Enum):
    SCORE = "score"        # solo / score-attack: только число
    WDL = "wdl"            # win/draw/loss между командами
    RANK = "rank"          # FFA / ranked teams: ранги 1..N, ничьи разрешены

@dataclass(frozen=True)
class GameSpec:
    roles: dict[str, RoleSpec]
    seats: tuple[SeatSpec, ...]
    outcome_kind: OutcomeKind
    symmetric: bool = True                 # все места взаимозаменяемы → можно ротировать места
    max_episode_steps: int | None = None

    @property
    def num_seats(self) -> int: ...
    def teams(self) -> dict[int, list[int]]: ...
    def seats_of_role(self, role: str) -> list[int]: ...

@dataclass
class Outcome:
    kind: OutcomeKind
    seat_score: dict[int, float] | None = None   # «сырой» счёт игры (solo, Halite)
    team_rank: dict[int, int] | None = None      # 1 = лучший; WDL = ранги {1,2} или {1,1}

@dataclass
class StepResult:
    obs: dict[int, Obs]                 # наблюдения для мест, которые действуют на СЛЕДУЮЩЕМ шаге
    rewards: dict[int, float]           # награды ВСЕМ местам (в т.ч. неактивным/выбывшим на этом шаге)
    terminated: dict[int, bool]         # per-seat: место закончило (выбыло или игра окончена)
    truncated: dict[int, bool]
    acting: frozenset[int]              # кто действует на следующем шаге (turn-based, «ход раз в N тиков»)
    action_masks: dict[int, Mask]       # только для acting-мест
    infos: dict[int, dict]
    outcome: Outcome | None = None      # заполнено ⇔ эпизод окончен (все места terminated/truncated)

class MultiAgentEnv(ABC):
    spec: GameSpec
    def reset(self, seed: int | None = None) -> StepResult: ...
    def step(self, actions: dict[int, Action]) -> StepResult:    # actions только для acting-мест
        ...

class LegacyEnvAdapter(MultiAgentEnv):
    """Оборачивает текущий BaseEnv: одна роль, team=seat, acting=info['active'] или все,
    outcome из info['outcome'|'rank'] или из суммарной награды."""
```

Семантика (из PettingZoo AEC/Parallel и OpenSpiel): эпизод окончен, когда все места terminated/truncated; выбывшее место перестаёт появляться в `acting`; награды места накапливаются, пока оно не действует.

### 3.2 Seat-трекер в воркере: корректные reward/done для неактивных и выбывших мест (D3, **M**, зависит от D1; quick-fix-версия поверх текущего API — **S**)

```python
class SeatTrajectory:
    """Буфер одного (env, seat). Переход (s_t, a_t) «открыт», пока место снова не начнёт действовать
    или не завершится. Все награды, пришедшие в промежутке, суммируются в r_t."""
    def on_act(self, obs, action, logp, value, mask, hidden) -> Optional[TrajectoryChunk]:
        # закрывает предыдущий открытый переход; если буфер полон — запечатывает chunk
        # с bootstrap = value (V текущего состояния того же места) и возвращает его
    def on_reward(self, r: float) -> None          # += к открытому переходу
    def on_terminal(self, truncated: bool, bootstrap: float | None) -> Optional[TrajectoryChunk]:
        # done=1 на открытом переходе, сброс hidden; chunk не обязан быть полным (padding+valid-mask)
```

Ключевые свойства: (1) ни одна награда не теряется; (2) `done` ставится на последнем собственном переходе места, даже если игра закончилась на чужом ходу; (3) инференс и обновление LSTM-состояния — только для acting-мест (тот же порядок шагов, что видит learner); (4) выбывание = `terminated[seat]`, без сброса всего env.

Для «быстрого фикса» в текущем `rollout_worker.py` (без API v2): `pending_reward[e,p]` и запечатывание буфера только при следующем активном шаге/конце эпизода; при `done` и неактивном слоте — `buf._rewards[cursor-1] += r; buf._dones[cursor-1] = 1`.

### 3.3 Структурированные наблюдения (D2, **M**, независимо)

```python
TensorTree = torch.Tensor | dict[str, "TensorTree"]
class ObsSpec:                                   # из gym.Space: Box, Dict, MultiBinary (вложенные Dict)
    @classmethod
    def from_space(cls, space) -> "ObsSpec": ...
    def allocate(self, leading: tuple[int, ...]) -> dict[str, np.ndarray]: ...
    def stack(self, obs_list) -> dict[str, np.ndarray]: ...
def tree_map(fn, tree: TensorTree) -> TensorTree: ...
```
`TrajectoryChunk.observations: TensorTree`; `BaseEncoder.forward(obs: TensorTree)`; `VectorEnv`, `RolloutBuffer`, `_prepare_batch`, сериализация gRPC, нормализация — через `tree_map`. Entity-list = `Dict(entities=Box(N_max, F), entity_mask=MultiBinary(N_max))` — паддинг + маска становятся конвенцией, а не хаком.

### 3.4 Per-unit действия (D4, **M**, независимо; D5 зависит от D4)

```python
class UnitActions(gym.spaces.Space):
    """max_units юнитов × под-действие (Discrete(A) | MultiDiscrete([A1..Ac])). Также GridNet: max_units = H*W."""
    def __init__(self, max_units: int, per_unit: gym.spaces.Discrete | gym.spaces.MultiDiscrete): ...

class FactorizedCategorical(Distribution):
    """logits [B, U, A] (или список по компонентам), mask [B, U, A], unit_mask [B, U].
    Юнит без валидных действий/несуществующий → logp=0, entropy=0, без NaN."""
    def log_prob(self, a: Tensor) -> Tensor              # [B]   сумма по живым юнитам (joint)
    def log_prob_per_unit(self, a: Tensor) -> Tensor     # [B, U]
    def entropy(self, reduce: Literal["sum", "mean"] = "mean") -> Tensor
    def unit_mask(self) -> Tensor                        # [B, U]
```
- `ActionSpec`: нативный тип `"units"` / nd-`MultiDiscrete` с `action_shape=(U,)|(U,C)|(H,W,C)`, dtype int64, маска формы `(U, A)`; никаких Python-циклов по компонентам; исправить порядок компонент (числовой, а не строковый).
- Политика должна видеть per-unit фичи, а не только `latent`:
```python
@dataclass
class EncoderOutput:
    latent: Tensor                    # [B, D] → RNN → value / глобальные головы
    aux: dict[str, Tensor]            # напр. unit_emb [B, U, E], spatial [B, C, H, W]
class BasePolicy(nn.Module):
    def forward(self, latent: Tensor, aux: dict[str, Tensor] | None = None) -> Distribution: ...
```
`ActorCriticNetwork.forward`: если encoder вернул `EncoderOutput`, RNN применяется к `latent`, `aux` передаётся в policy как есть (совместимо со старыми энкодерами, возвращающими Tensor).

### 3.5 APPO для многих юнитов (D5, **S**, зависит от D4)

`AlgorithmConfig.ratio_mode: Literal["joint", "per_unit"] = "joint"`:
- `joint` — как сейчас (корректно для малого числа компонент).
- `per_unit` — PPO-clip применяется к ratio каждого юнита с общим (командным) advantage, loss усредняется по живым юнитам; для V-trace ρ/c берётся `exp(mean_u log_ratio_u)` (или `clamp(joint)` — опция). Энтропия — среднее по живым юнитам.
- Опционально (позже, **M**): per-unit reward/value (`rewards: [T, U]`) для credit assignment.

### 3.6 Исходы, MatchResult v2 и рейтинги (D6 **S**, D7 **S-M**; зависят от D1)

```python
@dataclass
class SeatRecord:
    seat: int; role: str; team: int
    agent_id: str; network_id: str
    total_reward: float; score: float | None; rank: int | None   # team rank
@dataclass
class MatchResultV2:
    match_id: str; game_kind: OutcomeKind; seats: list[SeatRecord]; episode_length: int
```
(решает коллизию ключей `agent:network`).

```python
class BaseRating(ABC):
    def update(self, teams: list[list[str]], ranks: list[int]) -> None: ...   # команды × ранги (ничьи = равные ранги)
    def get(self, entity: str) -> tuple[float, float]: ...                    # (mu, sigma)
class TeamElo(BaseRating): ...          # 1v1 и командные; рейтинг команды = среднее участников
class PlackettLuceRating(BaseRating):   # FFA/N-команд (OpenSkill / TrueSkill-подобный), без внешних зависимостей
    ...
class ScoreTracker:                     # solo: EMA/среднее/CI очков на agent/checkpoint
    def update(self, entity: str, score: float) -> None: ...
```
Правило: пары сравниваются только между РАЗНЫМИ командами; одно и то же базовое значение-сущность (agent:network) внутри команды не сравнивается само с собой. Entity для рейтинга — `agent_id:network_id` (чекпоинты тоже получают рейтинг → нормальная PFSP по истории).

### 3.7 Матчмейкинг с ролями, командами и ротацией мест (D8, **M-L**; зависит от D1, D6)

```python
@dataclass
class Lineup:                   # кто играет за одну команду
    team: int
    members: dict[int, PlayerSlot]          # seat -> slot

class OpponentSampler(ABC):
    def sample(self, focus: str, role: str, k: int, game: GameSpec) -> list[str]:  # entity ids
        ...                                   # PFSP / uniform / latest / scripted

class BaseMatchmaker(ABC):
    def generate(self, game: GameSpec, focus_agents: list[str], n: int) -> list[MatchConfig]: ...

class LeagueMatchmaker(BaseMatchmaker):
    """1) выбирает focus-агента (round-robin/пропорционально по всем trainable, а не только agents[0]);
       2) выбирает команду/роль для focus с учётом allowed_roles агента;
       3) всю команду focus заполняет им самим (latest, collect=True) или (опция) teammates из пула;
       4) каждой из остальных команд выбирает ОДНУ сущность (PFSP по рейтингу/WR) и заполняет ею все места команды
          (FFA: N-1 независимых выборов);
       5) применяет перестановку мест: random или balanced (пары матчей с переставленными сторонами)."""

class SoloMatchmaker(BaseMatchmaker): ...   # num_seats == 1: нет соперников, чекпоинты не нужны
```
Конфиг: `matchmaking.class`, `matchmaking.seat_permutation: random|balanced|fixed`, `agents.<id>.roles: [hunter]`, `matchmaking.teammates: self|pool`. Coordinator получает матчмейкер по dotted path (сейчас жёстко зашит).

### 3.8 Асимметричные роли (D9, **L**; зависит от D1, D2, D8)

- Буферы и сети строятся per (agent, role): `network_factories[(agent_id, role)]`, `ActionSpec/ObsSpec` per role.
- Два варианта: (а) отдельный trainable-агент на роль (свой learner) — просто и изолированно; (б) один агент с несколькими головами/энкодерами по роли (`MultiRoleActorCritic(role -> ActorCritic)`), один learner, chunk помечается `role`.
- Рейтинг: отдельный по ролям (hunter-Elo против prey-Elo) + матрица WR role×role.

### 3.9 Eval v2 (D10, **M**; зависит от D1, D6, частично D7)

| Тип игры | Метрики |
|---|---|
| Solo | mean/std/95% CI score, длина эпизода, success-rate, распределение очков |
| 1v1 | WR + Wilson CI, раздельно по месту (первый/второй ход), ничьи; balanced seat schedule |
| FFA | средний ранг ± CI, top-1 rate, pairwise «beats»-матрица, PL/TrueSkill μ±σ |
| Teams | команда = фиксированный lineup (agent на всю команду), WR команд, перестановки сторон |
| Asym | матрица WR role×role (hunter_ckpt × prey_ckpt), по каждой роли отдельно |

Плюс: поддержка recurrent-сетей (hidden per (env, seat)), scripted-агентов, per-agent архитектур из CLI (`--agent name:path.pt:config.yaml`).

### 3.10 Метрики обучения (D11, **S**, независимо)

Монитор уже получает `MatchResult` — логировать в WandB: `episode_return`, `episode_length`, `score` (solo), WR/рейтинг по агентам и по месту, средний ранг (FFA), долю выбываний.

### 3.11 Сводка: порядок работ и оценки

| ID | Работа | Effort | Зависит от |
|---|---|---|---|
| Q1 | Quick-fixes: reward/done для неактивных мест (ET-01), коллизия ключей результата (ET-04), порядок компонент MultiDiscrete (ET-07), all-False маска (ET-08), recurrent в eval (ET-10), `env.num_players` vs config (ET-17), логирование эпизодов (ET-12), solo-eval (ET-11) | S (каждый), ~M суммарно | — |
| D1 | GameSpec/RoleSpec/SeatSpec/StepResult + LegacyEnvAdapter | M | — |
| D2 | Dict/entity наблюдения (ObsSpec, TensorTree) | M | — |
| D3 | SeatTrajectory в воркере (per-seat termination, acting-set) | M | D1 |
| D4 | UnitActions + FactorizedCategorical + nd ActionSpec + EncoderOutput.aux | M | (D2 желательно) |
| D5 | APPO ratio_mode=per_unit | S | D4 |
| D6 | Outcome + MatchResultV2 | S | D1 |
| D7 | Рейтинги: TeamElo, Plackett-Luce, ScoreTracker | S-M | D6 |
| D8 | LeagueMatchmaker: команды/роли/ротация мест/все focus-агенты; pluggable | M-L | D1, D6, (D7) |
| D9 | Асимметричные роли: per-role spaces/сети/буферы | L | D1, D2, D8 |
| D10 | Eval v2 по типам игр | M | D1, D6, (D7) |
| D11 | Логирование эпизодных метрик | S | — |

Рекомендуемый порядок: Q1 → D1+D6 → D3 → D11 → D4+D5 → D2 → D7 → D8 → D10 → D9.

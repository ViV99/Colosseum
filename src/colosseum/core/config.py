"""Pydantic v2 configuration models for the Colosseum framework (config v2, SP2).

Every section of the training pipeline (algorithm, environment, network,
rollout, learner, matchmaking, checkpointing, metrics, transport, run) has its own
model with sensible defaults.  The top-level :class:`ColosseumConfig` combines
them all and can be loaded from a YAML file via :func:`load_config`.

Changes from SP1: ``env.num_players``, ``training.phase`` and the ``self_play``
section are gone (the game structure comes from the env's ``GameSpec``); new are
``env.max_idle_steps``, ``matchmaking``, ``checkpoint.interval`` / ``pool_size``,
``agents.<id>.roles``, ``algorithm.ratio_mode`` / ``unit_trace`` /
``entropy_reduction`` and ``networks.critic_encoder_class``; ``rollout.chunk_length``
is at least 2.

SP3: ``agents.<id>.kind`` (trainable / scripted / frozen) and the implicit trainable ``agent_0``.
SP3: ``checkpoint.keep_last`` / ``keep_every`` (``pool_size`` is read as ``keep_last``).
SP3: ``init`` and ``kickstart`` sections (global + per agent); ``training.kickstart_*`` are translated
into ``kickstart``.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import logging
import re
import types
import typing
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from colosseum.core.errors import ConfigError
from colosseum.league.schedule import ScheduleValue, parse_schedule

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ids used as path components
# ---------------------------------------------------------------------------

_PATH_COMPONENT_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


def check_path_component(value: str, what: str) -> str:
    """Return ``value`` if it is safe as one path component, else raise ConfigError.

    Agent ids and checkpoint ids name directories (``<checkpoints>/<agent_id>/<ckpt_id>``).
    Allowed: letters, digits, ``_``, ``.`` and ``-``, not starting with ``.`` or ``-``.
    This rejects empty ids, ``.``/``..``, hidden names, separators and absolute paths.
    """
    if not isinstance(value, str) or _PATH_COMPONENT_RE.fullmatch(value) is None:
        raise ConfigError(
            f"Invalid {what} {value!r}: use letters, digits, '_', '.' and '-' "
            f"(not starting with '.' or '-'); it is used as a directory name"
        )
    return value


# The metrics record kinds / global metric namespaces (``colosseum.metrics.jsonl.METRIC_KINDS``):
# an agent with one of these ids would collide with them in metrics.jsonl and WandB (T6.4).
RESERVED_AGENT_IDS = frozenset({"ratings", "system", "episodes", "train"})


def check_agent_id(value: str) -> str:
    """Return ``value`` if it is a valid agent id, else raise ConfigError.

    An agent id is a safe path component (:func:`check_path_component`) without ``.``,
    so every agent is addressable as ``--set agents.<id>.<section>.<key>=...``, and not
    one of :data:`RESERVED_AGENT_IDS`.
    """
    check_path_component(value, "agent id")
    if "." in value:
        raise ConfigError(f"Invalid agent id {value!r}: '.' is not allowed (it separates --set path parts)")
    if value in RESERVED_AGENT_IDS:
        raise ConfigError(f"Invalid agent id {value!r}: reserved for global metrics")
    return value


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LRSchedule(str, Enum):
    """Supported learning-rate schedules."""

    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"


class TransportMode(str, Enum):
    """Communication backend between workers and learners."""

    LOCAL = "local"  # multiprocessing / shared memory (single machine)
    GRPC = "grpc"  # distributed via gRPC


# ---------------------------------------------------------------------------
# Section configs
# ---------------------------------------------------------------------------


class StrictModel(BaseModel):
    """Base for every config model: unknown keys are errors, not silently ignored (R5-07). Fields with an
    alias (``class``, ``from``, ``lambda``) accept their field name too (a ``model_dump()`` without aliases
    validates again)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# Matchmaking keys an agent may override (spec block 5); shuffle_seats and matchmaker_class stay global.
AGENT_MATCHMAKING_KEYS = frozenset({"opponents", "anchors", "pfsp", "layouts", "teammates", "teammate_self_prob"})
# SP2 matchmaking knobs: accepted only in the raw global matchmaking section, translated, never stored.
SP2_MATCHMAKING_KNOBS = ("mode", "self_play_ratio", "latest_prob", "pfsp_exponent")
_SP2_MATCHMAKING_DEFAULTS = {"mode": "self_play", "self_play_ratio": 0.5, "latest_prob": 0.5, "pfsp_exponent": 1.0}
# SP2 kickstart knobs of the training section -> keys of the top-level kickstart section.
SP2_KICKSTART_KNOBS = {"kickstart_teacher": "teacher", "kickstart_lambda": "lambda",
                       "kickstart_decay_steps": "decay_steps", "kickstart_kl": "kl"}

# A share or an anchor weight: a number or {step: value} points over the run's global env steps.
ScheduleField = Annotated[ScheduleValue, BeforeValidator(parse_schedule)]


class AlgorithmConfig(StrictModel):
    """Hyperparameters for the RL algorithm."""

    name: str = "appo"
    algorithm_class: str = Field(
        default="colosseum.algorithms.appo.APPO",
        description="Dotted import path to algorithm class.",
    )
    gamma: float = Field(default=0.99, ge=0.0, le=1.0, description="Discount factor.")
    vtrace_lambda: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="V-trace lambda: trace coefficients c_t = lambda * min(c_bar, rho_t). "
                    "1.0 = plain V-trace; ~0.9-0.95 trades bias for lower variance "
                    "(on-policy it equals GAE(lambda)). GAE itself is not used.",
    )
    eps_clip: float = Field(default=0.2, gt=0.0, description="PPO clipping epsilon.")
    value_loss_coeff: float = Field(default=0.5, ge=0.0, description="Coefficient for value-function loss.")
    entropy_coeff: float = Field(default=0.01, ge=0.0, description="Entropy bonus coefficient.")
    max_grad_norm: float = Field(default=0.5, gt=0.0, description="Max gradient norm for clipping.")
    num_epochs: int = Field(default=1, ge=1, description="PPO epochs per batch of data.")
    minibatch_chunks: int = Field(
        default=0,
        ge=0,
        description="Minibatch size measured in trajectory CHUNKS (each chunk is a full "
                    "[T]-step sequence; minibatching is over the batch dimension B, not over "
                    "timesteps, so recurrent sequences stay intact). 0 = use all chunks as one "
                    "batch (the typical APPO setting).",
    )
    vtrace_rho_bar: float = Field(default=1.0, gt=0.0, description="V-trace truncation for importance weights (rho).")
    vtrace_c_bar: float = Field(default=1.0, gt=0.0, description="V-trace truncation for trace-cutting (c).")
    learning_rate: float = Field(default=3e-4, gt=0.0, description="Initial learning rate.")
    lr_schedule: LRSchedule = Field(
        default=LRSchedule.LINEAR,
        description="LR schedule over training progress = env steps so far / training.total_timesteps.",
    )
    use_torch_compile: bool = Field(
        default=False,
        description="Compile V-trace with torch.compile. Adds ~1-3s startup latency.",
    )
    normalize_advantages: bool = Field(
        default=True,
        description="Normalize advantages per minibatch for stable PPO training.",
    )
    use_amp: bool = Field(default=False, description="Enable automatic mixed precision training.")
    amp_dtype: str = Field(default="float16", description="AMP dtype: 'float16' or 'bfloat16'.")
    ratio_mode: Literal["auto", "joint", "per_unit"] = Field(
        default="auto",
        description="Policy loss over deciders: 'joint' = PPO clip on the joint ratio, advantage times the "
                    "clipped scalar rho; 'per_unit' = ratio and clip per decider with a shared advantage and no "
                    "rho factor, mean over valid deciders. 'auto' = per_unit with Units actions, else joint.",
    )
    unit_trace: Literal["auto", "joint", "geo_mean", "none"] = Field(
        default="auto",
        description="Scalar rho for the V-trace targets: 'joint' = exp(sum of decider log-ratios), 'geo_mean' = "
                    "exp(mean), 'none' = rho = c = 1. 'auto' = joint (also with Units: units-experiment "
                    "ruling, docs/benchmarks.md).",
    )
    entropy_reduction: Literal["auto", "mean_valid", "sum"] = Field(
        default="auto",
        description="How entropy and the kickstart KL are reduced over deciders. 'auto' = sum with "
                    "ratio_mode joint, mean_valid with per_unit.",
    )


class EnvConfig(StrictModel):
    """Environment specification."""

    env_class: str = Field(
        ...,
        description="Dotted import path to a MultiAgentEnv class (e.g. 'examples.tic_tac_toe.game.TicTacToe').",
    )
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Extra kwargs forwarded to the env constructor.")
    max_idle_steps: int = Field(
        default=1000, ge=1,
        description="Max steps in a row without acting seats and without episode_over before the env is "
                    "reported as broken (EnvContractError).",
    )


class CoreConfig(StrictModel):
    """Core (trunk) between encoder and heads: ``{class: <dotted path>, kwargs: {...}}``."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_path: str = Field(
        ..., alias="class",
        description="Dotted path to a colosseum.networks.cores.Core subclass "
                    "(e.g. 'colosseum.networks.cores.LSTMCore').",
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs for the core constructor (input_dim is passed automatically).",
    )


class NetworkConfig(StrictModel):
    """Model specification: a monolithic ``model_class`` or encoder + core + heads."""

    model_config = ConfigDict(extra="forbid")

    model_class: str | None = Field(
        default=None,
        description="Dotted path to a PolicyModel subclass. When set, encoder/core/heads must be omitted.",
    )
    encoder_class: str | None = Field(default=None, description="Dotted path to the encoder class.")
    core: CoreConfig | None = Field(
        default=None, description="Optional core between encoder and heads (null = stateless NoCore).",
    )
    policy_class: str | None = Field(default=None, description="Dotted path to the policy head class.")
    value_class: str | None = Field(default=None, description="Dotted path to the value head class.")
    critic_encoder_class: str | None = Field(
        default=None,
        description="Optional dotted path to a BaseCriticEncoder: the value head then sees core features plus "
                    "the encoded global_state (centralized critic). Composed models only; the agent's role must "
                    "declare a global_state_space.",
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra kwargs forwarded to the model (or encoder and head) constructors.",
    )

    @model_validator(mode="after")
    def _check_model_spec(self) -> NetworkConfig:
        parts = {
            "encoder_class": self.encoder_class,
            "core": self.core,
            "policy_class": self.policy_class,
            "value_class": self.value_class,
            "critic_encoder_class": self.critic_encoder_class,
        }
        if self.model_class:
            extra = [name for name, value in parts.items() if value is not None]
            if extra:
                raise ValueError(
                    f"networks.model_class is set, so {', '.join(extra)} must be omitted"
                )
            return self
        missing = [n for n in ("encoder_class", "policy_class", "value_class") if not parts[n]]
        if missing:
            raise ValueError(
                "networks: set either model_class, or all of encoder_class, policy_class, "
                f"value_class (missing: {', '.join(missing)})"
            )
        return self


class RolloutConfig(StrictModel):
    """Worker / rollout collection settings."""

    chunk_length: int = Field(
        default=256, ge=2,
        description="Slots per trajectory chunk (S >= 2): decisions (act), bootstrap observations (boot) and "
                    "padding (pad).",
    )
    num_workers: int = Field(default=4, ge=1, description="Number of worker processes.")
    envs_per_worker: int = Field(default=8, ge=1, description="Vectorised envs per worker process.")
    weight_sync_interval_sec: float = Field(
        default=5.0,
        ge=0.0,
        description="How often (seconds) workers pull fresh weights from the weight store.",
    )
    vec_env: Literal["sync", "subprocess"] = Field(
        default="sync",
        description="Vectorised-env backend per worker: 'sync' (envs stepped sequentially in "
                    "the worker process) or 'subprocess' (envs stepped in parallel child "
                    "processes — better for CPU-heavy envs).",
    )
    subproc_workers: int | None = Field(
        default=None,
        description="Number of child processes for the 'subprocess' vec_env (defaults to "
                    "min(envs_per_worker, cpu_count)). Ignored for 'sync'.",
    )
    torch_threads: int = Field(
        default=1,
        ge=1,
        description="torch intra-op threads per worker process, set at process start "
                    "(inter-op threads are always 1). SubprocessVectorEnv children always "
                    "use 1 thread.",
    )
    match_refresh_interval_sec: float = Field(
        default=30.0,
        ge=0.0,
        description="How often (seconds) the coordinator re-generates match assignments and "
                    "pushes them (plus any new checkpoints) to workers, so self-play vs. "
                    "historical checkpoints and PFSP opponent selection update during training. "
                    "0 disables runtime refresh (static matchmaking).",
    )


class LearnerConfig(StrictModel):
    """Learner process settings."""

    device: str = Field(default="auto", description="Torch device string ('auto', 'cuda:0', 'cpu').")
    queue_size: int = Field(default=64, ge=1, description="Max trajectory chunks buffered in the learner queue.")
    batch_chunks: int = Field(default=16, ge=1, description="Number of chunks aggregated into one training batch.")
    weight_push_interval: int = Field(
        default=5, ge=1,
        description="Push weights to workers every N training steps. Each push clones the full "
                    "state_dict to CPU, and workers only pull every weight_sync_interval_sec, so "
                    "pushing every single step is wasteful; a small value keeps weights fresh "
                    "without per-step serialization overhead.",
    )
    pin_memory: bool = Field(
        default=False,
        description="Pin batch tensors for faster CPU-to-GPU transfer. Only effective with CUDA.",
    )
    torch_threads: int | None = Field(
        default=None,
        ge=1,
        description="torch threads per learner process. None = auto: 2 on CUDA; on CPU "
                    "max(1, (cpu_count - num_workers * rollout.torch_threads) // num_learners).",
    )


class TrainingConfig(StrictModel):
    """Top-level training loop settings."""

    total_timesteps: int = Field(default=10_000_000, ge=1, description="Total env timesteps before training ends.")
    seed: int | None = Field(default=None, description="Global random seed for reproducibility.")
    resume_from: str | None = Field(
        default=None,
        description="Resume each trainable agent before training from: a checkpoint dir (containing "
                    "model.pt; weights, trainer state and policy_version), a previous run dir "
                    "(containing checkpoints/; each agent's latest checkpoint there), or a .pt "
                    "state_dict (e.g. a BC output; weights only, policy_version 0). Policy versions "
                    "and the global env-step counter continue from the checkpoint.",
    )


class OpponentShares(StrictModel):
    """Shares of the opponent categories, drawn independently for every opposing team (spec block 5).

    Each is a number or a schedule ``{env_step: value}``. Empty categories pass their share to the
    others in proportion; ``rivals`` counts as ``latest`` for a team the owner cannot play.
    """

    latest: ScheduleField = Field(default=0.7, description="The owner's latest weights (self-play); for a team "
                                                            "the owner cannot play, another agent's latest by PFSP.")
    snapshots: ScheduleField = Field(default=0.2, description="Stored snapshots of the agents that play the team "
                                                               "(own, the opponent's in asymmetric games, others'), "
                                                               "by PFSP.")
    rivals: ScheduleField = Field(default=0.0, description="Latest weights of the other trainable agents that play "
                                                            "the team (arena; they collect too), by PFSP.")
    anchors: ScheduleField = Field(default=0.1, description="The owner's anchors (scripted / frozen agents) that "
                                                             "play the team, by their weights.")


class PfspConfig(StrictModel):
    """Prioritized fictitious self-play over candidates of a category (spec block 5)."""

    weighting: Literal["hard", "balanced", "uniform"] = Field(
        default="hard", description="hard: (1 - x)^exponent; balanced: x(1 - x); uniform: 1 (x = the owner's EMA "
                                    "score against the candidate; floor 1e-6).")
    exponent: float = Field(default=2.0, ge=0.0, description="Exponent of the hard weighting.")
    halflife_games: float = Field(default=200.0, gt=0.0, description="EMA half-life of the PFSP score, in games.")


class MatchmakingConfig(StrictModel):
    """How the coordinator builds lineups (spec block 5).

    ``opponents``, ``anchors``, ``pfsp``, ``layouts``, ``teammates`` and ``teammate_self_prob`` may be
    overridden per agent (``agents.<id>.matchmaking``); ``shuffle_seats`` and ``matchmaker_class``
    are global only. The SP2 knobs (``mode``, ``self_play_ratio``, ``latest_prob``,
    ``pfsp_exponent``) are translated from the raw global section (``translate_sp2_matchmaking``).
    """

    opponents: OpponentShares = Field(default_factory=OpponentShares)
    anchors: list[str] | dict[str, ScheduleField] | None = Field(
        default=None,
        description="Scripted / frozen agents the owner meets as anchors: null = every scripted and frozen agent "
                    "(weight 1 each); [] = none; a list of names (weight 1 each) or {name: weight or schedule}.",
    )
    pfsp: PfspConfig = Field(default_factory=PfspConfig)
    layouts: dict[str, float] = Field(
        default_factory=dict,
        description="Layout weights, e.g. {2p: 0.5, 4p: 0.5}. Empty = every layout with equal weight. Only "
                    "layouts with a seat for the data owner's role are drawn.",
    )
    teammates: Literal["self", "mixed"] = Field(
        default="self",
        description="'self' = the team core takes every seat of the team it can play; 'mixed' = each such "
                    "seat goes to the core with probability teammate_self_prob, else to another candidate.",
    )
    teammate_self_prob: float = Field(
        default=0.5, ge=0.0, le=1.0, description="With teammates: mixed, probability that a seat goes to the core.",
    )
    shuffle_seats: bool = Field(
        default=True,
        description="Permute teams with equal role composition and seats of the same role within a team (global).",
    )
    matchmaker_class: str | None = Field(
        default=None,
        description="Dotted path to a colosseum.league.BaseMatchmaker subclass that replaces the built-in "
                    "mixture (global).",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_sp2_knobs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            knobs = [k for k in SP2_MATCHMAKING_KNOBS if k in data]
            if knobs:
                raise ValueError(
                    f"{knobs} are SP2 matchmaking knobs: they are accepted only in the global matchmaking section "
                    f"of the input config (translated into opponents and pfsp); write opponents/pfsp here"
                )
        return data

    @field_validator("layouts")
    @classmethod
    def _check_layout_weights(cls, layouts: dict[str, float]) -> dict[str, float]:
        bad = {name: weight for name, weight in layouts.items() if not weight > 0}
        if bad:
            raise ValueError(f"matchmaking.layouts weights must be > 0, got {bad}")
        return layouts

    @field_validator("anchors")
    @classmethod
    def _check_anchor_names(cls, anchors: Any) -> Any:
        names = list(anchors) if anchors is not None else []
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"matchmaking.anchors: names must be non-empty strings, got {names}")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"matchmaking.anchors lists {duplicates} more than once")
        return anchors

    @field_validator("matchmaker_class")
    @classmethod
    def _check_matchmaker_class(cls, path: str | None) -> str | None:
        if path is not None and "." not in path:
            raise ValueError(f"matchmaking.matchmaker_class must be a dotted path 'module.Class', got {path!r}")
        return path


def translate_sp2_matchmaking(raw: dict) -> dict:
    """Pure translation of SP2 matchmaking knobs in a raw global section (spec block 5).

    Without knobs: a deep copy. With at least one: missing knobs take SP2's defaults
    (``mode: self_play``, ``self_play_ratio: 0.5``, ``latest_prob: 0.5``, ``pfsp_exponent: 1.0``);
    ``spr = 1`` under ``self_play``, else ``self_play_ratio``; ``latest = spr * latest_prob``,
    ``snapshots = spr * (1 - latest_prob)``, ``rivals = 1 - spr``, ``anchors = 0``, each rounded to
    12 decimals; ``pfsp = {weighting: hard, exponent: pfsp_exponent}``. Knobs together with
    ``opponents`` or ``pfsp``, or a knob out of range, raise ConfigError.
    """
    present = [k for k in SP2_MATCHMAKING_KNOBS if k in raw]
    if not present:
        return copy.deepcopy(raw)
    clash = [k for k in ("opponents", "pfsp") if k in raw]
    if clash:
        raise ConfigError(
            f"matchmaking: the SP2 knobs {present} cannot be combined with {clash}; drop the knobs and set "
            f"matchmaking.opponents / matchmaking.pfsp only"
        )
    knobs = {**_SP2_MATCHMAKING_DEFAULTS, **{k: raw[k] for k in present}}

    def number(name: str, low: float, high: float | None) -> float:
        value = knobs[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not (
                low <= value and (high is None or value <= high)):
            bound = f"[{low}, {high}]" if high is not None else f">= {low}"
            raise ConfigError(f"matchmaking.{name}={value!r} (SP2 knob) must be a number {bound}")
        return float(value)

    if knobs["mode"] not in ("self_play", "league"):
        raise ConfigError(f"matchmaking.mode={knobs['mode']!r} (SP2 knob) must be 'self_play' or 'league'")
    self_play_ratio = number("self_play_ratio", 0.0, 1.0)        # checked even where mode makes it unused
    spr = 1.0 if knobs["mode"] == "self_play" else self_play_ratio
    latest_prob = number("latest_prob", 0.0, 1.0)
    exponent = number("pfsp_exponent", 0.0, None)
    out = {k: copy.deepcopy(v) for k, v in raw.items() if k not in SP2_MATCHMAKING_KNOBS}
    shares = {"latest": spr * latest_prob, "snapshots": spr * (1.0 - latest_prob), "rivals": 1.0 - spr,
              "anchors": 0.0}
    out["opponents"] = {name: round(share, 12) for name, share in shares.items()}  # 0.4 * 0.75 -> 0.3, not 0.30..04
    out["pfsp"] = {"weighting": "hard", "exponent": exponent}
    return out


def translate_sp2_kickstart(raw: dict) -> dict:
    """Move SP2's ``training.kickstart_*`` of a raw config into a top-level ``kickstart`` section.

    Returns ``raw`` itself when there is nothing to translate, else a new dict (inputs are not
    mutated). Old knobs together with a top-level ``kickstart`` raise ConfigError.
    """
    training = raw.get("training")
    if not isinstance(training, dict):
        return raw
    present = [k for k in SP2_KICKSTART_KNOBS if k in training]
    if not present:
        return raw
    if raw.get("kickstart") is not None:
        raise ConfigError(
            f"training.{', training.'.join(present)} (SP2 knobs) cannot be combined with a top-level kickstart "
            f"section; move them into kickstart: {{teacher, lambda, decay_steps, kl}}"
        )
    out = dict(raw)
    out["training"] = {k: v for k, v in training.items() if k not in SP2_KICKSTART_KNOBS}
    out["kickstart"] = {SP2_KICKSTART_KNOBS[k]: copy.deepcopy(training[k]) for k in present}
    return out


def merge_matchmaking(base: dict, override: dict) -> dict:
    """A per-agent matchmaking override on the global section (spec block 5).

    ``opponents`` and ``pfsp`` merge key by key (one share's schedule is replaced whole, never merged
    point by point); every other key (``anchors``, ``layouts``, ...) replaces the global value.
    Inputs are not mutated.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in ("opponents", "pfsp") and isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **copy.deepcopy(value)}
        else:
            out[key] = copy.deepcopy(value)
    return out


class CheckpointConfig(StrictModel):
    """Snapshot storage (SP3 spec block 4). Snapshots live in the run dir (``<run>/checkpoints/``).

    Kept: the newest ``keep_last`` (with ``trainer_state.pt``), every snapshot whose version is a multiple of
    ``interval * keep_every`` (``model.pt`` + ``meta.json``), and every final snapshot. ``pool_size`` (SP2) is
    read as ``keep_last`` with a warning.
    """

    interval: int = Field(
        default=1000, ge=1,
        description="Save a checkpoint every N TRAINING steps (optimizer updates / policy versions), not env "
                    "steps (was self_play.checkpoint_interval).",
    )
    keep_last: int = Field(
        default=20, ge=1,
        description="The newest N snapshots per agent are kept, with trainer_state.pt (was pool_size, a FIFO pool).",
    )
    keep_every: int = Field(
        default=10, ge=0,
        description="Also keep every snapshot whose version is a multiple of interval * keep_every (model.pt and "
                    "meta.json only); 0 = off.",
    )
    save_optimizer: bool = Field(
        default=True,
        description="Whether checkpoints include the trainer state (optimizer, LR progress, AMP scaler, "
                    "kickstart, counters) as trainer_state.pt. Without it a resume restores weights and "
                    "policy_version only.",
    )

    @model_validator(mode="before")
    @classmethod
    def _translate_pool_size(cls, data: Any) -> Any:
        """SP2's ``pool_size`` is ``keep_last`` (one warning); both together are an error."""
        if not isinstance(data, dict) or "pool_size" not in data:
            return data
        if "keep_last" in data:
            raise ValueError("checkpoint.pool_size (SP2) and checkpoint.keep_last are both set; keep only keep_last "
                             "(pool_size is its old name)")
        data = dict(data)
        data["keep_last"] = data.pop("pool_size")
        logger.warning(f"checkpoint.pool_size is the SP2 name of checkpoint.keep_last; read as keep_last: "
                       f"{data['keep_last']} (rename it in the config)")
        return data


class InitConfig(StrictModel):
    """Warm start of a trainable agent's weights (spec block 6). The top-level section is the default of
    every trainable agent; ``agents.<id>.init`` deep-merges onto it."""

    from_: str | None = Field(
        default=None, alias="from",
        description=".pt state_dict | checkpoint dir (model.pt, role signature checked) | run dir (the agent's "
                    "latest checkpoint there) | name of a frozen agent. Weights only: policy version 0, a fresh "
                    "optimizer, counters from zero. Ignored for agents restored by training.resume_from.",
    )
    strict: bool = Field(
        default=True,
        description="false: load only the tensors whose name and shape match; the rest are listed by validate "
                    "and in the log (no matching tensor at all is an error).",
    )
    critic_warmup_steps: int = Field(
        default=0, ge=0,
        description="The first N learner train steps update only the value path (PolicyModel.value_parameters()): "
                    "policy, entropy and kickstart losses off, observation-normalizer statistics frozen (the "
                    "critic's global-state normalizers keep learning), kickstart decay starts afterwards.",
    )


class KickstartConfig(StrictModel):
    """A decaying pull of the student towards a teacher (spec block 6). The top-level section is the default
    of every trainable agent; ``agents.<id>.kickstart`` deep-merges onto it."""

    teacher: str | None = Field(
        default=None,
        description="Name of a frozen or scripted agent, or a path: a .pt (the student's architecture) or a "
                    "checkpoint dir (its own architecture from meta.json). None = no kickstart.",
    )
    lambda_: float = Field(
        default=1.0, ge=0.0, alias="lambda",
        description="Initial weight of the kickstart term; decays linearly to 0 over decay_steps train steps "
                    "(after the critic warm-up).",
    )
    decay_steps: int = Field(default=50_000, ge=1, description="Train steps over which lambda decays to 0.")
    kl: Literal["forward", "reverse"] = Field(
        default="forward",
        description="Neural teachers: 'forward' = KL(teacher || student), 'reverse' = KL(student || teacher). "
                    "Scripted teachers use the label loss -log pi(a_teacher) instead.",
    )


class MetricsConfig(StrictModel):
    """Logging and metrics settings."""

    use_wandb: bool = Field(default=False, description="Enable Weights & Biases logging.")
    wandb_project: str = Field(default="colosseum", description="WandB project name.")
    wandb_entity: str | None = Field(default=None, description="WandB entity (team or user).")
    log_interval: int = Field(default=10, ge=1, description="Log metrics every N training steps.")
    console_interval_sec: float = Field(
        default=10.0, ge=0.0,
        description="Seconds between console progress lines and episodes/system/ratings records.",
    )


class RunConfig(StrictModel):
    """Where a training run writes its outputs: ``<dir>/<name>/``."""

    name: str | None = Field(
        default=None,
        description="Run name (one path component); default '<config_stem>-<YYYYmmdd-HHMMSS>'. "
                    "An explicit name whose run dir already exists is an error.",
    )
    dir: str = Field(default="runs", description="Parent directory of all runs.")

    @field_validator("name")
    @classmethod
    def _check_name(cls, name: str | None) -> str | None:
        if name is not None:
            try:
                check_path_component(name, "run name")
            except ConfigError as e:
                raise ValueError(str(e)) from e
        return name


class TransportConfig(StrictModel):
    """Communication backend settings.

    ``mode`` and ``grpc_port`` are unused since SP1; kept for SP5 (distribution). Distributed
    roles take their ports as command-line flags.
    """

    mode: TransportMode = Field(
        default=TransportMode.LOCAL,
        description="Transport backend to use. Unused since SP1; kept for SP5 (distribution).",
    )
    grpc_port: int = Field(
        default=50051, ge=1, le=65535,
        description="Port for gRPC services. Unused since SP1; kept for SP5 (distribution).",
    )
    grpc_max_message_mb: int = Field(
        default=64,
        ge=1,
        description="Max gRPC message size in MiB.",
    )


# ---------------------------------------------------------------------------
# Agents (SP3 spec block 1): trainable, scripted and frozen
# ---------------------------------------------------------------------------


AgentKind = Literal["trainable", "scripted", "frozen"]

# Without any trainable agent in ``agents`` this trainable agent (global settings) exists implicitly.
IMPLICIT_AGENT_ID = "agent_0"


def _check_roles_list(roles: list[str] | None) -> list[str] | None:
    if roles is not None:
        if not roles:
            raise ValueError("roles must not be empty (omit it to play every role)")
        duplicates = sorted({r for r in roles if roles.count(r) > 1})
        if duplicates:
            raise ValueError(f"roles lists {duplicates} more than once")
    return roles


class _AgentEntry(StrictModel):
    """Base of the three agent kinds: an unknown field is named together with the kind's fields."""

    @model_validator(mode="before")
    @classmethod
    def _name_unknown_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        fields = cls.model_fields
        allowed = set(fields) | {f.alias for f in fields.values() if f.alias}
        unknown = sorted(str(key) for key in data if key not in allowed)
        if unknown:
            kind = data.get("kind", "trainable")
            names = sorted(f.alias or name for name, f in fields.items())
            hint = ""
            if kind == "trainable" and {"class", "kwargs"} & set(unknown):
                hint = "; a rule-based bot needs kind: scripted"
            elif kind == "trainable" and "path" in unknown:
                hint = "; fixed weights from a file need kind: frozen"
            raise ValueError(f"unknown field(s) {unknown} for a {kind} agent (its fields: {names}){hint}")
        return data


class TrainableAgent(_AgentEntry):
    """A trainable agent (``kind: trainable``, the default): partial per-agent overrides.

    The section overrides are deep-merged onto the global sections before validation (R5-08), so an
    override that sets only ``learning_rate`` keeps every other global algorithm value.

    Caveat for ``networks``: the merge is key by key, so an override that switches to
    ``model_class`` still inherits the global ``encoder_class`` / ``policy_class`` /
    ``value_class`` (and ``core``) unless it sets them to ``null``, and an override that
    swaps only ``encoder_class`` still inherits the global ``networks.kwargs`` (set
    ``kwargs`` explicitly if the new classes take different arguments).
    """

    kind: Literal["trainable"] = "trainable"
    roles: list[str] | None = Field(
        default=None,
        description="Roles this agent plays (all must have the same spaces). Omitted = every role of the game, "
                    "which then must all have the same spaces.",
    )
    networks: dict[str, Any] | None = None
    algorithm: dict[str, Any] | None = None
    learner: dict[str, Any] | None = None
    matchmaking: dict[str, Any] | None = Field(default=None, description="Per-agent matchmaking override.")
    init: dict[str, Any] | None = Field(default=None, description="Per-agent warm start (init) override.")
    kickstart: dict[str, Any] | None = Field(default=None, description="Per-agent kickstart override.")

    @field_validator("roles")
    @classmethod
    def _check_roles(cls, roles: list[str] | None) -> list[str] | None:
        return _check_roles_list(roles)

    @field_validator("matchmaking")
    @classmethod
    def _check_matchmaking_override(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return value
        knobs = sorted(set(value) & set(SP2_MATCHMAKING_KNOBS))
        if knobs:
            raise ValueError(f"matchmaking: {knobs} are SP2 knobs, accepted only in the global matchmaking "
                             f"section; set opponents / pfsp in agents.<id>.matchmaking")
        global_only = sorted(set(value) & {"shuffle_seats", "matchmaker_class"})
        if global_only:
            raise ValueError(f"matchmaking: {global_only} are global only; set them in the top-level "
                             f"matchmaking section")
        unknown = sorted(set(value) - AGENT_MATCHMAKING_KEYS)
        if unknown:
            raise ValueError(f"matchmaking: unknown keys {unknown}; an agent may override "
                             f"{sorted(AGENT_MATCHMAKING_KEYS)}")
        return value


# SP2 name of the per-agent override model.
AgentOverride = TrainableAgent


class ScriptedAgent(_AgentEntry):
    """A rule-based bot (``kind: scripted``): a ``colosseum.players.ScriptedBot`` subclass built with ``kwargs``."""

    kind: Literal["scripted"]
    class_path: str = Field(..., alias="class",
                            description="Dotted path to a colosseum.players.ScriptedBot subclass.")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Constructor kwargs of the bot.")
    roles: list[str] | None = Field(
        default=None, description="Roles the bot plays (their spaces may differ). Omitted = every role of the game.",
    )

    @field_validator("class_path")
    @classmethod
    def _check_class_path(cls, value: str) -> str:
        if "." not in value:
            raise ValueError(f"class must be a dotted path 'package.module.Class', got {value!r}")
        return value

    @field_validator("roles")
    @classmethod
    def _check_roles(cls, roles: list[str] | None) -> list[str] | None:
        return _check_roles_list(roles)


class FrozenAgent(_AgentEntry):
    """Fixed weights from a file (``kind: frozen``): a checkpoint dir (roles and networks from its
    ``meta.json``) or a ``.pt`` state_dict (``networks``: partial override onto the global networks;
    ``roles``: default every role of the game, which then must share one set of spaces)."""

    kind: Literal["frozen"]
    path: str = Field(..., min_length=1, description="Checkpoint dir or .pt state_dict.")
    networks: dict[str, Any] | None = Field(default=None, description="Only with a .pt path.")
    roles: list[str] | None = Field(default=None, description="Only with a .pt path.")

    @field_validator("roles")
    @classmethod
    def _check_roles(cls, roles: list[str] | None) -> list[str] | None:
        return _check_roles_list(roles)


AgentEntry = Annotated[TrainableAgent | ScriptedAgent | FrozenAgent, Field(discriminator="kind")]

_AGENT_SECTIONS = ("networks", "algorithm", "learner", "matchmaking", "init", "kickstart")


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``. Non-dict values replace."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class BCConfig(StrictModel):
    """Offline behavioral cloning (``colosseum bc``)."""

    seq_len: int = Field(
        default=64, ge=1,
        description="Window length (transitions) for stateful models; ignored by stateless "
                    "ones. CLI --seq-len overrides it.",
    )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class ColosseumConfig(StrictModel):
    """Top-level configuration combining every section.

    Can be loaded from a YAML file via :func:`load_config`.
    """

    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    env: EnvConfig
    networks: NetworkConfig
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    learner: LearnerConfig = Field(default_factory=LearnerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    matchmaking: MatchmakingConfig = Field(default_factory=MatchmakingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    init: InitConfig = Field(default_factory=InitConfig)
    kickstart: KickstartConfig = Field(default_factory=KickstartConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    bc: BCConfig = Field(default_factory=BCConfig)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    agents: dict[str, AgentEntry] = Field(
        default_factory=dict,
        description="Agents by id. kind: trainable (default; partial overrides of networks / algorithm / "
                    "learner and roles), scripted (a ScriptedBot class with kwargs and roles) or frozen (fixed "
                    "weights from a checkpoint dir or a .pt). Without a trainable agent an implicit trainable "
                    "'agent_0' with the global settings exists.",
    )

    @model_validator(mode="before")
    @classmethod
    def _translate_sp2_matchmaking_knobs(cls, data: Any) -> Any:
        """SP2 knobs in the raw global matchmaking section -> opponents + pfsp, with one warning."""
        if not isinstance(data, dict):
            return data
        raw = data.get("matchmaking")
        if not isinstance(raw, dict) or not any(k in raw for k in SP2_MATCHMAKING_KNOBS):
            return data
        translated = translate_sp2_matchmaking(raw)
        logger.warning(
            f"SP2 matchmaking knobs {[k for k in SP2_MATCHMAKING_KNOBS if k in raw]} translated to "
            f"opponents={translated['opponents']}, pfsp={translated['pfsp']}; write these in the config "
            f"(the old knobs are not kept in config.resolved.yaml)"
        )
        return {**data, "matchmaking": translated}

    @model_validator(mode="before")
    @classmethod
    def _translate_sp2_kickstart_knobs(cls, data: Any) -> Any:
        """SP2's training.kickstart_* -> the top-level kickstart section, with one warning."""
        if not isinstance(data, dict):
            return data
        translated = translate_sp2_kickstart(data)
        if translated is not data:
            logger.warning(
                f"training.kickstart_* are SP2 knobs; translated to kickstart: {translated['kickstart']} (write that "
                f"section instead; the old keys are not kept in config.resolved.yaml)"
            )
        return translated

    @field_validator("agents", mode="before")
    @classmethod
    def _default_kind(cls, value: Any) -> Any:
        """``null`` = a trainable agent with no overrides; an entry without ``kind`` is trainable."""
        if not isinstance(value, dict):
            return value
        out = {}
        for agent_id, entry in value.items():
            if entry is None:
                entry = {}
            if isinstance(entry, dict) and "kind" not in entry:
                entry = {**entry, "kind": "trainable"}
            out[agent_id] = entry
        return out

    @field_validator("agents")
    @classmethod
    def _check_agent_ids(cls, agents: dict[str, Any]) -> dict[str, Any]:
        for agent_id in agents:
            try:
                check_agent_id(agent_id)
            except ConfigError as e:
                raise ValueError(str(e)) from e
        return agents

    @model_validator(mode="after")
    def _validate_agent_overrides(self) -> ColosseumConfig:
        if self._has_implicit_agent() and IMPLICIT_AGENT_ID in self.agents:
            raise ValueError(
                f"agents.{IMPLICIT_AGENT_ID} is a {self.agents[IMPLICIT_AGENT_ID].kind} agent, but the config has no "
                f"trainable agent, so '{IMPLICIT_AGENT_ID}' is the implicit trainable agent; rename the "
                f"{self.agents[IMPLICIT_AGENT_ID].kind} agent or add a trainable agent"
            )
        for agent_id, entry in self.agents.items():
            if entry.kind != "trainable":
                continue
            try:
                self.get_agent_config(agent_id)
            except ValidationError as e:
                raise ValueError(f"agents.{agent_id}: invalid override:\n{e}") from None
        return self

    def _has_implicit_agent(self) -> bool:
        return not any(entry.kind == "trainable" for entry in self.agents.values())

    def agent_ids(self) -> list[str]:
        """Every agent in config order; the implicit trainable ``agent_0`` (if any) comes first."""
        ids = list(self.agents)
        return [IMPLICIT_AGENT_ID, *ids] if self._has_implicit_agent() else ids

    def _require_known_agent(self, agent_id: str) -> None:
        """Raise ConfigError unless ``agent_id`` is a configured agent or the implicit ``agent_0``."""
        if agent_id in self.agents or (agent_id == IMPLICIT_AGENT_ID and self._has_implicit_agent()):
            return
        if not self.agents:
            raise ConfigError(f"Unknown agent '{agent_id}': without an 'agents' section the only agent is "
                              f"'{IMPLICIT_AGENT_ID}'")
        raise ConfigError(f"Unknown agent '{agent_id}'. Known agents: {self.agent_ids()}")

    def agent_entry(self, agent_id: str) -> TrainableAgent | ScriptedAgent | FrozenAgent:
        """The agent's config entry (the implicit ``agent_0`` is ``TrainableAgent()``)."""
        self._require_known_agent(agent_id)
        entry = self.agents.get(agent_id)
        return entry if entry is not None else TrainableAgent()

    def agent_kind(self, agent_id: str) -> AgentKind:
        return self.agent_entry(agent_id).kind

    def get_agent_config(self, agent_id: str) -> ColosseumConfig:
        """Effective config of one TRAINABLE agent: global sections deep-merged with its overrides."""
        entry = self.agent_entry(agent_id)
        if entry.kind != "trainable":
            raise ConfigError(
                f"agent '{agent_id}' is a {entry.kind} agent: only trainable agents have a training config "
                f"(networks / algorithm / learner)"
            )
        data = self.model_dump(by_alias=True)
        for section in _AGENT_SECTIONS:
            part = getattr(entry, section)
            if part:
                merge = merge_matchmaking if section == "matchmaking" else deep_merge
                data[section] = merge(data[section], part)
        data["agents"] = {}
        return ColosseumConfig.model_validate(data)

    def agent_roles(self, agent_id: str) -> list[str] | None:
        """``agents.<id>.roles`` of any kind (None when omitted).

        Call it on the top-level config, not on a ``get_agent_config()`` result (which has ``agents == {}``).
        """
        roles = self.agent_entry(agent_id).roles
        return None if roles is None else list(roles)

    def get_trainable_agent_ids(self) -> list[str]:
        """Trainable agent ids in config order (owner rotation order); ``["agent_0"]`` without any."""
        ids = [agent_id for agent_id, entry in self.agents.items() if entry.kind == "trainable"]
        return ids or [IMPLICIT_AGENT_ID]

    def fixed_agent_ids(self) -> list[str]:
        """Scripted and frozen agent ids in config order."""
        return [agent_id for agent_id, entry in self.agents.items() if entry.kind != "trainable"]


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


_NUMBER_RE = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")


def parse_override_value(raw: str, key: str | None = None) -> Any:
    """Parse a ``--set key=value`` value with YAML semantics.

    ``null`` gives None, lists and dicts are YAML, and an unquoted number includes
    ``1e-4`` (which plain YAML 1.1 would keep as a string). A quoted value stays a
    string (``'"123"'`` -> ``"123"``), and so does a date-like value. Malformed YAML
    raises ConfigError naming ``key`` (when given) and the raw value.
    """
    text = raw.strip()
    if not text:
        return None
    if _NUMBER_RE.fullmatch(text):
        return int(text) if "." not in text and "e" not in text.lower() else float(text)
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        where = f"--set {key}={raw}" if key is not None else f"override value {raw!r}"
        reason = str(e).splitlines()[0] if str(e) else type(e).__name__
        raise ConfigError(f"Cannot parse {where}: invalid YAML ({reason})") from e
    if isinstance(value, (datetime.date, datetime.datetime)):
        return raw
    return value


def _unwrap_optional(tp: Any) -> Any:
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _model_members(tp: Any) -> list[type[BaseModel]] | None:
    """The config models an annotation stands for: a model, an optional model or a union of models
    (``Annotated`` discriminated unions included, e.g. agent entries); None for anything else."""
    if typing.get_origin(tp) is typing.Annotated:
        tp = typing.get_args(tp)[0]
    tp = _unwrap_optional(tp)
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return [tp]
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        members = [a for a in typing.get_args(tp) if a is not type(None)]
        if members and all(isinstance(m, type) and issubclass(m, BaseModel) for m in members):
            return members
    return None


def _check_override_path(parts: list[str]) -> None:
    """Walk the schema of ColosseumConfig along ``parts``; raise ConfigError on an unknown key.

    At a union of models (an agent entry) a key is valid if one of the members has it.
    """
    tp: Any = ColosseumConfig
    for i, part in enumerate(parts):
        where = ".".join(parts[: i + 1])
        members = _model_members(tp)
        if members is not None:
            found = None
            for model in members:
                name = next((n for n, f in model.model_fields.items() if part in (n, f.alias)), None)
                if name is not None:
                    found = model.model_fields[name].annotation
                    break
            if found is None:
                valid = sorted({f.alias or n for model in members for n, f in model.model_fields.items()})
                raise ConfigError(f"Unknown config key '{where}' (valid keys here: {valid})")
            tp = found
            continue
        tp = _unwrap_optional(tp)
        if typing.get_origin(tp) is dict:
            tp = typing.get_args(tp)[1]
        elif tp is Any:
            return  # free-form dict (env.kwargs, agent override bodies): checked at validation
        else:
            raise ConfigError(f"Cannot set '{'.'.join(parts)}': '{'.'.join(parts[:i])}' is not a section")


# Old (SP2) knobs that ``--set`` accepts although they are not in the schema: the config models translate
# them (one warning each) and never store them.
LEGACY_OVERRIDE_KEYS: set[str] = {"checkpoint.pool_size"}
LEGACY_OVERRIDE_KEYS.update(f"matchmaking.{knob}" for knob in SP2_MATCHMAKING_KNOBS)
LEGACY_OVERRIDE_KEYS.update(f"training.{knob}" for knob in SP2_KICKSTART_KNOBS)


def apply_overrides(data: dict, overrides: dict[str, Any]) -> dict:
    """Return a copy of raw config ``data`` with dotted-path ``overrides`` applied.

    Missing intermediate sections are created (e.g. ``agents.alpha.algorithm``). An
    unknown path raises ConfigError. Values are validated later by ``model_validate``.
    """
    out = copy.deepcopy(data)
    for key, value in overrides.items():
        parts = key.split(".")
        if not all(parts):
            raise ConfigError(f"Malformed override key '{key}'")
        if key not in LEGACY_OVERRIDE_KEYS:
            _check_override_path(parts)
        node = out
        for part in parts[:-1]:
            child = node.get(part)
            if child is None:
                child = node[part] = {}
            elif not isinstance(child, dict):
                raise ConfigError(f"Cannot set '{key}': '{part}' holds a {type(child).__name__}, not a section")
            node = child
        node[parts[-1]] = value
    return out


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ColosseumConfig:
    """Read YAML, apply ``--set`` overrides, validate. Any problem raises ConfigError."""
    path = Path(path)
    try:
        with path.open("r") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as e:
        raise ConfigError(f"Cannot read config {path}: {e}") from e
    if overrides:
        raw = apply_overrides(raw, overrides)
    try:
        return ColosseumConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"Invalid config {path}:\n{e}") from e


def config_hash(config: ColosseumConfig) -> str:
    """Short stable hash of a resolved config (stored in every checkpoint's meta.json)."""
    payload = json.dumps(config.model_dump(mode="json", by_alias=True), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

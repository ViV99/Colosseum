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
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import re
import types
import typing
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from colosseum.core.errors import ConfigError

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
    """Base for every config model: unknown keys are errors, not silently ignored (R5-07)."""

    model_config = ConfigDict(extra="forbid")


class AlgorithmConfig(StrictModel):
    """Hyperparameters for the RL algorithm."""

    name: str = "appo"
    algorithm_class: str = Field(
        default="colosseum.sp2.algorithms.appo.APPO",
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
                    "exp(mean), 'none' = rho = c = 1. 'auto' = joint without Units, geo_mean with Units.",
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
    kickstart_teacher: str | None = Field(
        default=None,
        description="Path to a frozen teacher .pt state_dict (e.g. a BC model), built from this "
                    "agent's networks config. When set, a decaying KL term between teacher and "
                    "student is added to the RL loss (direction: kickstart_kl). None disables it.",
    )
    kickstart_lambda: float = Field(
        default=1.0, ge=0.0, description="Initial weight of the kickstart KL term (decays to 0).",
    )
    kickstart_decay_steps: int = Field(
        default=50_000, ge=1, description="Training steps over which the kickstart lambda decays to 0.",
    )
    kickstart_kl: Literal["forward", "reverse"] = Field(
        default="forward",
        description="Kickstart KL direction: 'forward' = KL(teacher || student) (Kickstarting / "
                    "AlphaStar / VPT, mode-covering); 'reverse' = KL(student || teacher).",
    )


class MatchmakingConfig(StrictModel):
    """How the coordinator builds lineups (layout, match type, team cores, teammates, seats)."""

    mode: Literal["self_play", "league"] = Field(
        default="self_play",
        description="'self_play' = every match is self-play (self_play_ratio is treated as 1); "
                    "'league' = self-play with probability self_play_ratio, else an arena match.",
    )
    layouts: dict[str, float] = Field(
        default_factory=dict,
        description="Layout weights, e.g. {2p: 0.5, 4p: 0.5}. Empty = every layout with equal weight. Only "
                    "layouts with a seat for the data owner's role are drawn.",
    )
    self_play_ratio: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Probability of a self-play match in 'league' mode (the rest are arena matches).",
    )
    pfsp_exponent: float = Field(default=1.0, ge=0.0, description="Exponent p in PFSP priority: f(wr) = (1 - wr)^p.")
    latest_prob: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Self-play: probability that an opposing team core is the owner's latest policy "
                    "(else one of its checkpoints).",
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
        description="Permute teams with equal role composition and seats of the same role within a team.",
    )

    @field_validator("layouts")
    @classmethod
    def _check_layout_weights(cls, layouts: dict[str, float]) -> dict[str, float]:
        bad = {name: weight for name, weight in layouts.items() if not weight > 0}
        if bad:
            raise ValueError(f"matchmaking.layouts weights must be > 0, got {bad}")
        return layouts


class CheckpointConfig(StrictModel):
    """Checkpoint settings. Checkpoints are stored in the run dir (``<run>/checkpoints/``)."""

    interval: int = Field(
        default=1000, ge=1,
        description="Save a checkpoint every N TRAINING steps (optimizer updates / policy versions), not env "
                    "steps (was self_play.checkpoint_interval).",
    )
    pool_size: int = Field(
        default=20, ge=1, description="Max checkpoints kept in the FIFO pool per agent (was self_play.pool_size).",
    )
    save_optimizer: bool = Field(
        default=True,
        description="Whether checkpoints include the trainer state (optimizer, LR progress, AMP scaler, "
                    "kickstart, counters) as trainer_state.pt. Without it a resume restores weights and "
                    "policy_version only.",
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

    ``mode`` and ``grpc_port`` are unused in SP1; kept for SP5 (distribution). Distributed
    roles take their ports as command-line flags.
    """

    mode: TransportMode = Field(
        default=TransportMode.LOCAL,
        description="Transport backend to use. Unused in SP1; kept for SP5 (distribution).",
    )
    grpc_port: int = Field(
        default=50051, ge=1, le=65535,
        description="Port for gRPC services. Unused in SP1; kept for SP5 (distribution).",
    )
    grpc_max_message_mb: int = Field(
        default=64,
        ge=1,
        description="Max gRPC message size in MiB.",
    )


# ---------------------------------------------------------------------------
# Per-agent config overrides
# ---------------------------------------------------------------------------


class AgentOverride(StrictModel):
    """Per-agent overrides as partial dicts.

    They are deep-merged onto the global section before validation (R5-08), so an
    override that sets only ``learning_rate`` keeps every other global algorithm value.

    Caveat for ``networks``: the merge is key by key, so an override that switches to
    ``model_class`` still inherits the global ``encoder_class`` / ``policy_class`` /
    ``value_class`` (and ``core``) unless it sets them to ``null``, and an override that
    swaps only ``encoder_class`` still inherits the global ``networks.kwargs`` (set
    ``kwargs`` explicitly if the new classes take different arguments).
    """

    networks: dict[str, Any] | None = None
    algorithm: dict[str, Any] | None = None
    learner: dict[str, Any] | None = None
    roles: list[str] | None = Field(
        default=None,
        description="Roles this agent plays (all must have the same spaces). Omitted = every role of the game, "
                    "which then must all have the same spaces.",
    )

    @field_validator("roles")
    @classmethod
    def _check_roles(cls, roles: list[str] | None) -> list[str] | None:
        if roles is not None:
            if not roles:
                raise ValueError("roles must not be empty (omit it to play every role)")
            duplicates = sorted({r for r in roles if roles.count(r) > 1})
            if duplicates:
                raise ValueError(f"roles lists {duplicates} more than once")
        return roles


_AGENT_SECTIONS = ("networks", "algorithm", "learner")


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
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    bc: BCConfig = Field(default_factory=BCConfig)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    agents: dict[str, AgentOverride] = Field(
        default_factory=dict,
        description="Per-agent config overrides. Keys are agent IDs. "
                    "Empty = single agent_0 using global config.",
    )

    @field_validator("agents", mode="before")
    @classmethod
    def _null_agent_means_no_override(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: ({} if v is None else v) for k, v in value.items()}
        return value

    @field_validator("agents")
    @classmethod
    def _check_agent_ids(cls, agents: dict[str, AgentOverride]) -> dict[str, AgentOverride]:
        for agent_id in agents:
            try:
                check_agent_id(agent_id)
            except ConfigError as e:
                raise ValueError(str(e)) from e
        return agents

    @model_validator(mode="after")
    def _validate_agent_overrides(self) -> ColosseumConfig:
        for agent_id in self.agents:
            try:
                self.get_agent_config(agent_id)
            except ValidationError as e:
                raise ValueError(f"agents.{agent_id}: invalid override:\n{e}") from None
        return self

    def _require_known_agent(self, agent_id: str) -> None:
        """Raise ConfigError unless ``agent_id`` is a configured agent (or ``agent_0`` without ``agents``)."""
        if self.agents and agent_id not in self.agents:
            raise ConfigError(f"Unknown agent '{agent_id}'. Known agents: {sorted(self.agents)}")
        if not self.agents and agent_id != "agent_0":
            raise ConfigError(
                f"Unknown agent '{agent_id}': without an 'agents' section the only agent is 'agent_0'"
            )

    def get_agent_config(self, agent_id: str) -> ColosseumConfig:
        """Effective config of one agent: global sections deep-merged with its override."""
        self._require_known_agent(agent_id)
        data = self.model_dump(by_alias=True)
        override = self.agents.get(agent_id)
        if override is not None:
            for section in _AGENT_SECTIONS:
                part = getattr(override, section)
                if part:
                    data[section] = deep_merge(data[section], part)
        data["agents"] = {}
        return ColosseumConfig.model_validate(data)

    def agent_roles(self, agent_id: str) -> list[str] | None:
        """``agents.<id>.roles`` (None when omitted: the agent plays every role).

        Call it on the top-level config, not on a ``get_agent_config()`` result (which has ``agents == {}``).
        """
        self._require_known_agent(agent_id)
        override = self.agents.get(agent_id)
        return None if override is None or override.roles is None else list(override.roles)

    def get_trainable_agent_ids(self) -> list[str]:
        """Return list of trainable agent IDs.

        If agents dict is empty, defaults to ``["agent_0"]`` (single-agent mode).
        """
        if not self.agents:
            return ["agent_0"]
        return list(self.agents.keys())


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


def _check_override_path(parts: list[str]) -> None:
    """Walk the schema of ColosseumConfig along ``parts``; raise ConfigError on an unknown key."""
    tp: Any = ColosseumConfig
    for i, part in enumerate(parts):
        tp = _unwrap_optional(tp)
        where = ".".join(parts[: i + 1])
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            fields = tp.model_fields
            name = next((n for n, f in fields.items() if part in (n, f.alias)), None)
            if name is None:
                raise ConfigError(f"Unknown config key '{where}' (valid keys here: {sorted(fields)})")
            tp = fields[name].annotation
        elif typing.get_origin(tp) is dict:
            tp = typing.get_args(tp)[1]
        elif tp is Any:
            return  # free-form dict (env.kwargs, agent override bodies): checked at validation
        else:
            raise ConfigError(f"Cannot set '{'.'.join(parts)}': '{'.'.join(parts[:i])}' is not a section")


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

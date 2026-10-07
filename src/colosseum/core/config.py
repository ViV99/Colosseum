"""Pydantic v2 configuration models for the Colosseum framework.

Every section of the training pipeline (algorithm, environment, network,
rollout, learner, self-play, checkpointing, metrics, transport) has its own
model with sensible defaults.  The top-level :class:`ColosseumConfig` combines
them all and can be loaded from a YAML file via :func:`load_config`.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TrainingPhase(str, Enum):
    """High-level training phase."""

    BC = "bc"
    SELF_PLAY = "self_play"
    LEAGUE = "league"


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


class AlgorithmConfig(BaseModel):
    """Hyperparameters for the RL algorithm."""

    name: str = "appo"
    algorithm_class: str = Field(
        default="colosseum.algorithms.appo.APPO",
        description="Dotted import path to algorithm class.",
    )
    gamma: float = Field(default=0.99, ge=0.0, le=1.0, description="Discount factor.")
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0, description="GAE lambda.")
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
    lr_schedule: LRSchedule = Field(default=LRSchedule.LINEAR, description="LR schedule type.")
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


class EnvConfig(BaseModel):
    """Environment specification."""

    env_class: str = Field(
        ...,
        description="Dotted import path to the environment class (e.g. 'examples.tic_tac_toe.env.TicTacToeEnv').",
    )
    num_players: int = Field(default=2, ge=1, description="Number of player slots per match.")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Extra kwargs forwarded to the env constructor.")


class CoreConfig(BaseModel):
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


class NetworkConfig(BaseModel):
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


class RolloutConfig(BaseModel):
    """Worker / rollout collection settings."""

    chunk_length: int = Field(default=256, ge=1, description="Timesteps per trajectory chunk (T).")
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
    match_refresh_interval_sec: float = Field(
        default=30.0,
        ge=0.0,
        description="How often (seconds) the coordinator re-generates match assignments and "
                    "pushes them (plus any new checkpoints) to workers, so self-play vs. "
                    "historical checkpoints and PFSP opponent selection update during training. "
                    "0 disables runtime refresh (static matchmaking).",
    )


class LearnerConfig(BaseModel):
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


class TrainingConfig(BaseModel):
    """Top-level training loop settings."""

    phase: TrainingPhase = Field(default=TrainingPhase.SELF_PLAY, description="Current training phase.")
    total_timesteps: int = Field(default=10_000_000, ge=1, description="Total env timesteps before training ends.")
    seed: int | None = Field(default=None, description="Global random seed for reproducibility.")
    resume_from: str | None = Field(
        default=None,
        description="Resume each trainable agent's network (and optimizer, if available) from "
                    "this checkpoint before training. Either a path to a .pt state_dict (e.g. a "
                    "BC output) or a checkpoint id in the checkpoint dir (e.g. 'ckpt_v100').",
    )
    kickstart_teacher: str | None = Field(
        default=None,
        description="Path to a frozen teacher .pt state_dict (e.g. a BC model). When set, a "
                    "decaying KL(student || teacher) term is added to the RL loss (online BC / "
                    "kickstarting). None disables kickstarting.",
    )
    kickstart_lambda: float = Field(
        default=1.0, ge=0.0, description="Initial weight of the kickstart KL term (decays to 0).",
    )
    kickstart_decay_steps: int = Field(
        default=50_000, ge=1, description="Training steps over which the kickstart lambda decays to 0.",
    )


class SelfPlayConfig(BaseModel):
    """Self-play and PFSP / league settings."""

    checkpoint_interval: int = Field(
        default=1000,
        ge=1,
        description="Save a new checkpoint to the self-play pool every N TRAINING steps "
                    "(optimizer updates / policy versions), NOT env steps. Note one training "
                    "step consumes chunk_length * batch_chunks env steps, so pick a value well "
                    "below total_timesteps / (chunk_length * batch_chunks) to actually fill the "
                    "self-play pool.",
    )
    pool_size: int = Field(default=20, ge=1, description="Max checkpoints kept in the FIFO pool per agent.")
    latest_prob: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of sampling the latest policy as an opponent (vs. a historical checkpoint).",
    )
    self_play_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of matches that are solo self-play (rest are arena matches).",
    )
    pfsp_exponent: float = Field(
        default=1.0,
        ge=0.0,
        description="Exponent p in PFSP priority: f(wr) = (1 - wr)^p.",
    )


class CheckpointConfig(BaseModel):
    """Checkpoint storage settings."""

    dir: str = Field(default="checkpoints", description="Directory for saving checkpoints.")
    save_optimizer: bool = Field(default=True, description="Whether to include optimizer state in checkpoints.")


class MetricsConfig(BaseModel):
    """Logging and metrics settings."""

    use_wandb: bool = Field(default=False, description="Enable Weights & Biases logging.")
    wandb_project: str = Field(default="colosseum", description="WandB project name.")
    wandb_entity: str | None = Field(default=None, description="WandB entity (team or user).")
    log_interval: int = Field(default=10, ge=1, description="Log metrics every N training steps.")


class TransportConfig(BaseModel):
    """Communication backend settings."""

    mode: TransportMode = Field(default=TransportMode.LOCAL, description="Transport backend to use.")
    grpc_port: int = Field(default=50051, ge=1, le=65535, description="Port for gRPC services (when mode='grpc').")
    grpc_max_message_mb: int = Field(
        default=64,
        ge=1,
        description="Max gRPC message size in MiB.",
    )


# ---------------------------------------------------------------------------
# Per-agent config overrides
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Per-agent overrides. Fields that are None inherit from global config."""

    networks: NetworkConfig | None = None
    algorithm: AlgorithmConfig | None = None
    learner: LearnerConfig | None = None


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class ColosseumConfig(BaseModel):
    """Top-level configuration combining every section.

    Can be loaded from a YAML file via :func:`load_config`.
    """

    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    env: EnvConfig
    networks: NetworkConfig
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    learner: LearnerConfig = Field(default_factory=LearnerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    self_play: SelfPlayConfig = Field(default_factory=SelfPlayConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    agents: dict[str, AgentConfig] = Field(
        default_factory=dict,
        description="Per-agent config overrides. Keys are agent IDs. "
                    "Empty = single agent_0 using global config.",
    )

    def get_agent_config(self, agent_id: str) -> ColosseumConfig:
        """Return an effective config for a specific agent.

        Creates a copy where networks/algorithm/learner are overridden
        by any agent-specific values.
        """
        if agent_id not in self.agents:
            return self.model_copy(deep=True)

        overrides = self.agents[agent_id]
        data = self.model_dump()

        if overrides.networks is not None:
            data["networks"] = overrides.networks.model_dump()
        if overrides.algorithm is not None:
            data["algorithm"] = overrides.algorithm.model_dump()
        if overrides.learner is not None:
            data["learner"] = overrides.learner.model_dump()

        data["agents"] = {}
        return ColosseumConfig.model_validate(data)

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


def load_config(path: str | Path) -> ColosseumConfig:
    """Read a YAML file and return a fully validated :class:`ColosseumConfig`.

    Args:
        path: Filesystem path to the YAML configuration file.

    Returns:
        A validated ``ColosseumConfig`` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        pydantic.ValidationError: If the YAML content fails validation.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = Path(path)
    with path.open("r") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    return ColosseumConfig.model_validate(raw)

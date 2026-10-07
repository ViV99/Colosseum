"""Dynamic class import and instantiation utilities.

The registry module provides a thin convenience layer so that configuration
files can reference Python classes by their fully-qualified dotted path
(e.g. ``"examples.tic_tac_toe.env.TicTacToeEnv"``) and the framework can
import and instantiate them at runtime without hard-coded imports.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from colosseum.core.config import ColosseumConfig
    from colosseum.networks.actor_critic import ActorCriticNetwork


def import_class(dotted_path: str) -> type:
    """Import and return a class from a fully-qualified dotted path.

    Args:
        dotted_path: A string of the form ``"package.module.ClassName"``.

    Returns:
        The class object.

    Raises:
        ValueError: If *dotted_path* does not contain at least one ``'.'``
            separating a module path from a class name.
        ModuleNotFoundError: If the module cannot be imported.
        AttributeError: If the module does not contain the requested name.
    """
    if "." not in dotted_path:
        raise ValueError(
            f"dotted_path must be in the form 'module.ClassName', got: {dotted_path!r}"
        )

    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if not isinstance(cls, type):
        raise TypeError(
            f"{dotted_path!r} resolved to {type(cls).__name__}, not a class"
        )

    return cls


def instantiate(dotted_path: str, **kwargs: Any) -> Any:
    """Import a class from *dotted_path* and return a new instance.

    This is a convenience wrapper around :func:`import_class` that also calls
    the constructor with the provided keyword arguments.

    Args:
        dotted_path: Fully-qualified class path.
        **kwargs: Arguments forwarded to the class constructor.

    Returns:
        An instance of the imported class.
    """
    cls = import_class(dotted_path)
    return cls(**kwargs)


def build_network(config: ColosseumConfig) -> ActorCriticNetwork:
    """Build an ActorCriticNetwork from config (encoder + policy + value + optional recurrent).

    This is the single source of truth for network construction, used by
    launcher, CLI, and any other entry point.

    When ``config.networks.recurrent_type`` is set to ``"lstm"`` or ``"gru"``,
    an RNN module is created with ``input_size = encoder.latent_dim`` and
    inserted between the encoder and the policy/value heads.  The policy and
    value heads must accept ``recurrent_hidden_size`` as their input dimension.
    """
    import torch.nn as nn

    from colosseum.networks.actor_critic import ActorCriticNetwork

    encoder = import_class(config.networks.encoder_class)(**config.networks.kwargs)
    policy = import_class(config.networks.policy_class)(**config.networks.kwargs)
    value = import_class(config.networks.value_class)(**config.networks.kwargs)

    recurrent = None
    if config.networks.recurrent_type is not None:
        rnn_type = config.networks.recurrent_type.lower()
        input_size = encoder.latent_dim
        hidden_size = config.networks.recurrent_hidden_size
        num_layers = config.networks.recurrent_num_layers
        if rnn_type == "lstm":
            recurrent = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        elif rnn_type == "gru":
            recurrent = nn.GRU(input_size, hidden_size, num_layers, batch_first=False)
        else:
            raise ValueError(
                f"Unsupported recurrent_type: {config.networks.recurrent_type!r}. "
                f"Use 'lstm', 'gru', or None."
            )

    return ActorCriticNetwork(encoder, policy, value, recurrent=recurrent)


def validate_config(config: ColosseumConfig) -> None:
    """Build the env + network and run a dummy forward/act to catch mismatches early.

    Surfaces the common "policy/value head input dim doesn't match
    encoder.latent_dim (or recurrent_hidden_size)" error — which the README warns
    about — with a clear message *before* spawning processes, instead of an opaque
    shape error deep inside a worker. Raises ValueError on any structural problem.
    """
    import numpy as np
    import torch

    from colosseum.core.action_spec import ActionSpec

    env = import_class(config.env.env_class)(**config.env.kwargs)
    try:
        try:
            net = build_network(config)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Failed to build network from config: {e}") from e

        spec = ActionSpec.from_space(env.action_space)
        sample = env.observation_space.sample()
        obs = torch.as_tensor(np.asarray(sample), dtype=torch.float32).unsqueeze(0)
        hidden = net.initial_hidden(1) if net.is_recurrent else None

        try:
            actions, log_probs, values, _ = net.act(obs, hidden=hidden)
        except Exception as e:  # noqa: BLE001
            hint = ""
            if net.is_recurrent:
                hint = (
                    " (recurrent: policy/value heads must accept "
                    f"recurrent_hidden_size={config.networks.recurrent_hidden_size}, "
                    "not encoder.latent_dim)"
                )
            else:
                hint = (
                    f" (policy/value heads must accept encoder.latent_dim="
                    f"{net.encoder.latent_dim})"
                )
            raise ValueError(
                f"Network forward pass failed during validation{hint}: {e}"
            ) from e

        # Action shape sanity check against the action spec.
        expected = spec.action_shape
        got = tuple(actions.shape[1:])
        if expected and got != expected:
            raise ValueError(
                f"Policy produced actions of shape {got} but the action space "
                f"expects {expected}. Check your policy head / distribution."
            )
        if values.shape[0] != 1:
            raise ValueError(
                f"Value head returned shape {tuple(values.shape)}; expected a "
                f"scalar per batch element (shape [B])."
            )
    finally:
        env.close()

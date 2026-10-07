"""Dynamic class import and instantiation utilities.

The registry module provides a thin convenience layer so that configuration
files can reference Python classes by their fully-qualified dotted path
(e.g. ``"examples.tic_tac_toe.env.TicTacToeEnv"``) and the framework can
import and instantiate them at runtime without hard-coded imports.
"""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any

from colosseum.core.errors import ConfigError

if TYPE_CHECKING:
    from colosseum.core.config import ColosseumConfig
    from colosseum.networks.actor_critic import ActorCriticNetwork
    from colosseum.networks.model import PolicyModel


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


def _accepts_in_dim(cls: type) -> bool:
    """True if ``cls.__init__`` declares an explicit ``in_dim`` parameter."""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    return "in_dim" in params


def _build_head(dotted_path: str, in_dim: int, kwargs: dict[str, Any]) -> Any:
    """Instantiate a head, passing ``in_dim`` when its constructor accepts it."""
    cls = import_class(dotted_path)
    head_kwargs = dict(kwargs)
    if _accepts_in_dim(cls):
        head_kwargs["in_dim"] = in_dim
    return cls(**head_kwargs)


def build_model(config: ColosseumConfig) -> PolicyModel:
    """Build the agent's :class:`PolicyModel` from ``config.networks``.

    - ``model_class`` set: ``import_class(model_class)(**networks.kwargs)``; it must
      be a ``PolicyModel``.
    - otherwise a :class:`ComposedModel`: ``encoder(**kwargs)``, then
      ``core(input_dim=encoder.latent_dim, **core.kwargs)`` (``NoCore`` when
      ``core`` is null), then the heads with ``in_dim=core.output_dim`` if their
      constructor accepts ``in_dim``, plus ``**kwargs``.
    """
    from colosseum.networks.composed import ComposedModel
    from colosseum.networks.cores import Core, NoCore
    from colosseum.networks.model import PolicyModel

    net = config.networks
    if net.model_class:
        model = import_class(net.model_class)(**net.kwargs)
        if not isinstance(model, PolicyModel):
            raise ConfigError(
                f"networks.model_class {net.model_class!r} must subclass "
                f"colosseum.networks.model.PolicyModel, got {type(model).__name__}"
            )
        return model

    encoder = import_class(net.encoder_class)(**net.kwargs)
    latent_dim = encoder.latent_dim
    if net.core is None:
        core = NoCore(latent_dim)
    else:
        core = import_class(net.core.class_path)(input_dim=latent_dim, **net.core.kwargs)
        if not isinstance(core, Core):
            raise ConfigError(
                f"networks.core.class {net.core.class_path!r} must subclass "
                f"colosseum.networks.cores.Core, got {type(core).__name__}"
            )
    policy = _build_head(net.policy_class, core.output_dim, net.kwargs)
    value = _build_head(net.value_class, core.output_dim, net.kwargs)
    return ComposedModel(encoder, core, policy, value)


def build_network(config: ColosseumConfig) -> ActorCriticNetwork:
    """TRANSITIONAL: the legacy ``ActorCriticNetwork`` built from the new ``networks`` schema.

    Kept only until every caller uses :func:`build_model`; deleted together with
    ``networks/actor_critic.py``. Supports ``core`` = null, ``NoCore``,
    ``LSTMCore`` and ``GRUCore``.
    """
    import torch.nn as nn

    from colosseum.networks.actor_critic import ActorCriticNetwork
    from colosseum.networks.cores import GRUCore, LSTMCore, NoCore

    net = config.networks
    if net.model_class:
        raise ConfigError("networks.model_class is not supported by the legacy ActorCriticNetwork path")
    encoder = import_class(net.encoder_class)(**net.kwargs)
    out_dim = encoder.latent_dim
    recurrent = None
    if net.core is not None:
        core_cls = import_class(net.core.class_path)
        hidden = int(net.core.kwargs.get("hidden_size", 128))
        layers = int(net.core.kwargs.get("num_layers", 1))
        if issubclass(core_cls, LSTMCore):
            recurrent = nn.LSTM(out_dim, hidden, layers)
            out_dim = hidden
        elif issubclass(core_cls, GRUCore):
            recurrent = nn.GRU(out_dim, hidden, layers)
            out_dim = hidden
        elif not issubclass(core_cls, NoCore):
            raise ConfigError(f"core {net.core.class_path!r} is not supported by the legacy ActorCriticNetwork path")
    policy = _build_head(net.policy_class, out_dim, net.kwargs)
    value = _build_head(net.value_class, out_dim, net.kwargs)
    return ActorCriticNetwork(encoder, policy, value, recurrent=recurrent)


def _check_state(state: Any, batch: int, where: str) -> None:
    from colosseum.networks.state import tree_leaves

    for leaf in tree_leaves(state):
        if leaf.dim() == 0 or leaf.shape[0] != batch:
            raise ConfigError(
                f"{where}: every state tensor must have the batch dimension first "
                f"(expected {batch}), got a leaf of shape {tuple(leaf.shape)}"
            )


def validate_config(config: ColosseumConfig) -> None:
    """Build the env and the model and exercise ``step``/``unroll`` on dummy data.

    Raises :class:`ConfigError` with a precise message on any structural problem
    (bad class, head/core dimension mismatch, wrong value shape, state without a
    batch dim, action/mask size mismatch) before any process is spawned.
    """
    import numpy as np
    import torch

    from colosseum.core.action_spec import ActionSpec
    from colosseum.networks.composed import ComposedModel
    from colosseum.networks.distributions import Distribution

    try:
        env = import_class(config.env.env_class)(**config.env.kwargs)
    except Exception as e:
        raise ConfigError(f"Failed to create env {config.env.env_class!r}: {type(e).__name__}: {e}") from e
    try:
        try:
            model = build_model(config)
        except ConfigError:
            raise
        except Exception as e:
            raise ConfigError(f"Failed to build model from config.networks: {type(e).__name__}: {e}") from e

        hint = ""
        if isinstance(model, ComposedModel):
            hint = (
                f" Policy/value heads receive core.output_dim={model.core.output_dim} features "
                f"(encoder.latent_dim={model.encoder.latent_dim}); give the heads an `in_dim` "
                f"constructor argument or size them to match."
            )

        spec = ActionSpec.from_space(env.action_space)
        obs_dict, info_dict = env.reset(seed=0)
        sample = torch.as_tensor(np.asarray(obs_dict[0], dtype=np.float32))
        B, T = 2, 3
        obs = sample.unsqueeze(0).repeat(B, *([1] * sample.dim()))
        mask = None
        raw_mask = info_dict.get(0, {}).get("action_mask") if isinstance(info_dict, dict) else None
        if raw_mask is not None:
            if isinstance(raw_mask, dict):
                flat_mask = spec.flatten_mask(raw_mask)
            else:
                flat_mask = np.asarray(raw_mask, dtype=bool)
            if flat_mask.shape != (spec.flat_mask_size,):
                raise ConfigError(
                    f"env action_mask has shape {flat_mask.shape}, but the action space needs "
                    f"({spec.flat_mask_size},)"
                )
            mask = torch.as_tensor(flat_mask).unsqueeze(0).repeat(B, 1)

        model.eval()
        with torch.no_grad():
            state0 = model.initial_state(B)
            _check_state(state0, B, "initial_state(2)")
            try:
                out = model.step(obs, state0, mask)
            except Exception as e:
                raise ConfigError(
                    f"model.step failed on a dummy batch with obs shape {tuple(obs.shape)}: "
                    f"{type(e).__name__}: {e}.{hint}"
                ) from e
            if not isinstance(out.dist, Distribution):
                raise ConfigError(
                    f"the policy head must return a colosseum Distribution "
                    f"(colosseum.networks.distributions), got {type(out.dist).__name__}"
                )
            if tuple(out.value.shape) != (B,):
                raise ConfigError(
                    f"value must have shape [B]=({B},), got {tuple(out.value.shape)}. "
                    f"Squeeze the last dim in the value head."
                )
            _check_state(out.state, B, "step() state")
            actions = out.dist.sample()
            expected = (B, *spec.action_shape)
            if tuple(actions.shape) != expected:
                raise ConfigError(
                    f"policy produced actions of shape {tuple(actions.shape)}, but the action space "
                    f"expects {expected}. Check the policy head / distribution."
                )

            dones = torch.zeros(T, B, dtype=torch.bool)
            dones[1, 0] = True
            obs_seq = obs.unsqueeze(0).repeat(T, *([1] * obs.dim()))
            mask_seq = None if mask is None else mask.unsqueeze(0).repeat(T, 1, 1)
            try:
                unrolled = model.unroll(obs_seq, state0, dones, mask_seq)
                log_probs = unrolled.dist.log_prob(actions.repeat(T, *([1] * (actions.dim() - 1))))
            except Exception as e:
                raise ConfigError(
                    f"model.unroll failed on a dummy [T={T}, B={B}] batch: {type(e).__name__}: {e}"
                ) from e
            if tuple(unrolled.value.shape) != (T * B,) or tuple(log_probs.shape) != (T * B,):
                raise ConfigError(
                    f"unroll must return time-major [T*B]=({T * B},) values/log-probs, got "
                    f"{tuple(unrolled.value.shape)} / {tuple(log_probs.shape)}"
                )
    finally:
        env.close()

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
        model_cls = import_class(net.model_class)
        if not issubclass(model_cls, PolicyModel):
            raise ConfigError(
                f"networks.model_class {net.model_class!r} must subclass "
                f"colosseum.networks.model.PolicyModel, got {model_cls.__name__}"
            )
        return model_cls(**net.kwargs)

    encoder = import_class(net.encoder_class)(**net.kwargs)
    latent_dim = encoder.latent_dim
    if net.core is None:
        core = NoCore(latent_dim)
    else:
        core_cls = import_class(net.core.class_path)
        if not issubclass(core_cls, Core):
            raise ConfigError(
                f"networks.core.class {net.core.class_path!r} must subclass "
                f"colosseum.networks.cores.Core, got {core_cls.__name__}"
            )
        core = core_cls(input_dim=latent_dim, **net.core.kwargs)
    policy = _build_head(net.policy_class, core.output_dim, net.kwargs)
    value = _build_head(net.value_class, core.output_dim, net.kwargs)
    return ComposedModel(encoder, core, policy, value)


def _check_state_batch_dim(state_a: Any, state_b: Any, batches: tuple[int, int], where: str) -> None:
    """The same state built for two batch sizes must differ only in dim 0 of every leaf.

    Comparing two batch sizes catches layer-first layouts (``[L, B, H]``) that a
    single batch size cannot, e.g. ``num_layers == B``.
    """
    from colosseum.networks.state import tree_leaves

    leaves_a, leaves_b = tree_leaves(state_a), tree_leaves(state_b)
    if len(leaves_a) != len(leaves_b):
        raise ConfigError(
            f"{where}: the state structure depends on the batch size "
            f"({len(leaves_a)} tensors for B={batches[0]}, {len(leaves_b)} for B={batches[1]})"
        )
    for i, (a, b) in enumerate(zip(leaves_a, leaves_b, strict=True)):
        if a.dim() == 0 or b.dim() == 0 or a.shape[0] != batches[0] or b.shape[0] != batches[1] \
                or a.shape[1:] != b.shape[1:]:
            raise ConfigError(
                f"{where}: every state tensor must have the batch dimension first; state tensor #{i} "
                f"has shape {tuple(a.shape)} for B={batches[0]} and {tuple(b.shape)} for B={batches[1]} "
                f"(expected ({batches[0]}, ...) and ({batches[1]}, ...) with identical other dims). "
                f"Store RNN states as [B, num_layers, H], not nn.LSTM/nn.GRU's native [num_layers, B, H]."
            )


def _check_state_payload(state: Any) -> None:
    """The model state must survive the inter-process payload round trip.

    Chunks carry ``initial_state`` to the learner as numpy over ``mp.Queue``
    (pickle) or gRPC (:func:`colosseum.transport.serialization.pack_payload`).
    Both rebuild namedtuples by module path, so e.g. a namedtuple class created
    inside a function would only fail later, in a queue feeder thread.
    """
    from colosseum.networks.state import slice_batch, state_from_numpy, state_to_numpy
    from colosseum.transport.serialization import pack_payload, unpack_payload

    if state is None:
        return
    try:
        state_from_numpy(unpack_payload(*pack_payload(state_to_numpy(slice_batch(state, 0)))))
    except (TypeError, ValueError) as e:
        raise ConfigError(f"initial_state cannot be sent between processes: {e}") from e


def _make_env(config: ColosseumConfig) -> Any:
    """Instantiate ``config.env``; any failure becomes a ConfigError."""
    try:
        return import_class(config.env.env_class)(**config.env.kwargs)
    except Exception as e:
        raise ConfigError(f"Failed to create env {config.env.env_class!r}: {type(e).__name__}: {e}") from e


def _close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def check_env_num_players(config: ColosseumConfig, env: Any = None) -> None:
    """``env.num_players`` in the config must equal the env's own ``num_players`` (R5-16).

    Pass an already built ``env`` to reuse it (the caller keeps ownership); otherwise
    one is built from the config and closed here.
    """
    owned = env is None
    if owned:
        env = _make_env(config)
    try:
        actual = int(env.num_players)
    finally:
        if owned:
            _close_env(env)
    if actual != config.env.num_players:
        raise ConfigError(
            f"env.num_players={config.env.num_players} but {config.env.env_class}.num_players={actual}"
        )


def validate_config(config: ColosseumConfig) -> None:
    """Build the env and the model and exercise ``step``/``unroll`` on dummy data.

    Raises :class:`ConfigError` with a precise message on any structural problem
    (bad class, head/core dimension mismatch, wrong value shape, state without a
    batch dim or that cannot be sent between processes, action/mask size
    mismatch) before any process is spawned.
    """
    import numpy as np
    import torch

    from colosseum.core.action_spec import ActionSpec
    from colosseum.networks.composed import ComposedModel
    from colosseum.networks.distributions import Distribution

    env = _make_env(config)
    try:
        check_env_num_players(config, env)
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

        try:
            spec = ActionSpec.from_space(env.action_space)
        except Exception as e:
            raise ConfigError(
                f"Unsupported env action_space {env.action_space!r}: {type(e).__name__}: {e}"
            ) from e
        try:
            obs_dict, info_dict = env.reset(seed=0)
            sample = torch.as_tensor(np.asarray(obs_dict[0], dtype=np.float32))
        except Exception as e:
            raise ConfigError(
                f"env.reset(seed=0) of {config.env.env_class!r} failed or returned no observation "
                f"for player 0: {type(e).__name__}: {e}"
            ) from e
        B, B_ALT, T = 2, 3, 3
        obs = sample.unsqueeze(0).repeat(B, *([1] * sample.dim()))
        obs_alt = sample.unsqueeze(0).repeat(B_ALT, *([1] * sample.dim()))
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
        mask_alt = None if mask is None else mask[:1].repeat(B_ALT, 1)

        model.eval()
        with torch.no_grad():
            try:
                state0 = model.initial_state(B)
                state0_alt = model.initial_state(B_ALT)
            except Exception as e:
                raise ConfigError(f"model.initial_state(B) failed: {type(e).__name__}: {e}") from e
            _check_state_batch_dim(state0, state0_alt, (B, B_ALT), "initial_state")
            _check_state_payload(state0)
            try:
                out = model.step(obs, state0, mask)
                out_alt = model.step(obs_alt, state0_alt, mask_alt)
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
            _check_state_batch_dim(out.state, out_alt.state, (B, B_ALT), "step() state")
            try:
                actions = out.dist.sample()
            except Exception as e:
                raise ConfigError(
                    f"sampling from the policy distribution {type(out.dist).__name__} failed: "
                    f"{type(e).__name__}: {e}"
                ) from e
            expected = (B, *spec.action_shape)
            if tuple(actions.shape) != expected:
                raise ConfigError(
                    f"policy produced actions of shape {tuple(actions.shape)}, but the action space "
                    f"expects {expected}. Check the policy head / distribution."
                )
            # Same flat shape is not enough: head order, category counts and kinds must match.
            try:
                spec.check_distribution(out.dist)
            except ValueError as exc:
                raise ConfigError(
                    f"networks: policy distribution does not match the action space: {exc}"
                ) from exc

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
        _close_env(env)

"""``validate_config``: everything that can be checked before a run starts (spec block 9).

Re-exported as ``colosseum.core.registry.validate_config`` (callers use that name, so a test
can monkeypatch it there).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from colosseum.core.config import ColosseumConfig
from colosseum.core.errors import ConfigError, EnvContractError, PlayerError
from colosseum.core.registry import build_model, env_spec, make_env
from colosseum.core.specs import ActionSpec, ObsSpec
from colosseum.core.tree import (
    tree_index,
    tree_leaves,
    tree_map,
    tree_same_structure,
    tree_stack,
    tree_to_numpy,
    tree_to_torch,
)
from colosseum.core.types import SLOT_ACT, SLOT_BOOT, SLOT_PAD
from colosseum.envs.game import GameSpec, RoleSpec, StepResult
from colosseum.networks.dist import Distribution

VALIDATE_STEPS = 8  # random legal steps per enabled layout
VALIDATE_LINEUPS = 16  # lineups drawn per trainable agent from a custom matchmaker

# Synthetic chunk of the model check, time-major [S=4, B=2], spelled like the chunk v2 tests
# (A open ACT, T terminal ACT, R truncation BOOT, B chunk-end BOOT, P PAD):
#   column 0: A T P P   (an episode end, then padding)
#   column 1: A R A B   (a truncation, a new episode, a chunk-end bootstrap)
_CHUNK_KINDS = ((SLOT_ACT, SLOT_ACT), (SLOT_ACT, SLOT_BOOT), (SLOT_PAD, SLOT_ACT), (SLOT_PAD, SLOT_BOOT))
_CHUNK_RESET_AFTER = ((False, False), (True, True), (True, False), (True, False))


@dataclass
class ValidationReport:
    """What ``colosseum validate`` prints after the per-agent OK lines: the effective behaviour that is not
    visible in the config (fixed players; later the opponent mix and init reports)."""

    lines: list[str] = field(default_factory=list)


def _seeded(space: Any, rng: np.random.Generator) -> Any:
    space.seed(int(rng.integers(2**31 - 1)))
    return space


def random_legal_action(role: RoleSpec, mask: Any, rng: np.random.Generator) -> Any:
    """A uniformly random legal action of ``role`` (numpy, env format) under its normalized ``mask``.

    Discrete parts pick among the legal values; box parts and units are drawn by their spaces'
    ``sample`` (units with ``mask=`` the group's ``{"unit", "action"}`` mask), seeded from ``rng``.
    """
    spec = ActionSpec.from_space(role.action_space)
    values: dict[tuple[str, ...], Any] = {}
    for group in spec.groups:
        group_mask = spec.group_mask(mask, group)
        if group.kind == "discrete":
            legal = np.flatnonzero(group_mask) if group_mask is not None else np.arange(group.nvec[0])
            values[group.path] = np.asarray(rng.choice(legal), dtype=np.int64)
        elif group.kind == "multi_discrete":
            picks, offset = [], 0
            for n in group.nvec:
                row = group_mask[offset:offset + n] if group_mask is not None else np.ones(n, dtype=bool)
                picks.append(rng.choice(np.flatnonzero(row)))
                offset += n
            values[group.path] = np.asarray(picks, dtype=np.int64)
        elif group.kind == "box":
            space = role.action_space
            for key in group.path:
                space = space[key]
            values[group.path] = np.asarray(_seeded(space, rng).sample())
        else:
            values[group.path] = _seeded(group.units, rng).sample(mask=group_mask)
    if not spec.is_dict:
        return values[spec.groups[0].path]
    action: dict = {}
    for path, value in values.items():
        node = action
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return action


def _contains(space: Any, value: Any, what: str, where: str) -> None:
    if space is not None and not space.contains(value):
        raise EnvContractError(
            f"{where}: {what} is not in the role's space {space!r}; check its dtype, shape and bounds"
        )


def _check_spaces(spec: GameSpec, layout: str, result: StepResult, step: int) -> None:
    """Full ``space.contains`` checks of what the pipeline uses: the acting seats' observations and
    global states, and on truncation the final observations and global states (the BOOT slots).
    The context follows the global order "seat, episode step, layout"."""

    def where(seat: int) -> str:
        return f"validate, seat {seat}, episode step {step}, layout {layout}"

    global_state = result.global_state or {}
    for seat in sorted(result.acting):
        role = spec.roles[spec.role_of(layout, seat)]
        _contains(role.observation_space, result.obs[seat], "observation", where(seat))
        if role.global_state_space is not None:
            _contains(role.global_state_space, global_state[seat], "global_state", where(seat))
    if result.truncated:
        for seat, obs in (result.final_obs or {}).items():
            role = spec.roles[spec.role_of(layout, seat)]
            _contains(role.observation_space, obs, "final_obs", where(seat))
            if role.global_state_space is not None and seat in global_state:
                _contains(role.global_state_space, global_state[seat], "final global_state", where(seat))


def _exercise_env(config: ColosseumConfig, spec: GameSpec, layouts: Sequence[str]) -> dict[str, tuple]:
    """Reset every enabled layout and take up to ``VALIDATE_STEPS`` random legal steps under the
    contract checks (``EpisodeTracker``) and full ``space.contains`` checks.

    Returns one ``(obs, mask, global_state)`` sample (numpy, the env's values) per role that acted.
    """
    from colosseum.envs.contract import EpisodeTracker

    rng = np.random.default_rng(0)
    samples: dict[str, tuple] = {}
    env = make_env(config)
    try:
        for layout in layouts:
            tracker = EpisodeTracker(spec, max_idle_steps=config.env.max_idle_steps, context="validate")
            try:
                result = env.reset(seed=0, layout=layout)
            except EnvContractError:
                raise
            except Exception as e:
                raise ConfigError(f"env.reset(seed=0, layout={layout!r}) of {config.env.env_class!r} failed: "
                                  f"{type(e).__name__}: {e}") from e
            masks = tracker.on_reset(layout, result)
            for step in range(VALIDATE_STEPS + 1):
                _check_spaces(spec, layout, result, step)
                for seat in sorted(result.acting):
                    gs = (result.global_state or {}).get(seat)
                    samples.setdefault(spec.role_of(layout, seat), (result.obs[seat], masks[seat], gs))
                if result.episode_over or step == VALIDATE_STEPS:
                    break
                actions = {seat: random_legal_action(spec.roles[spec.role_of(layout, seat)], masks[seat], rng)
                           for seat in sorted(result.acting)}
                try:
                    result = env.step(actions)
                except EnvContractError:
                    raise
                except Exception as e:
                    raise ConfigError(f"env.step of {config.env.env_class!r} failed in layout {layout!r} at "
                                      f"episode step {step + 1}: {type(e).__name__}: {e}") from e
                masks = tracker.on_step(actions, result)
    finally:
        env.close()
    return samples


def _check_state_batch_dim(state_a: Any, state_b: Any, batches: tuple[int, int], where: str) -> None:
    """The same state built for two batch sizes must differ only in dim 0 of every leaf (SP1).

    Comparing two batch sizes catches layer-first layouts (``[L, B, H]``) that a single batch
    size cannot, e.g. ``num_layers == B``.
    """
    from colosseum.networks.state import tree_leaves as state_leaves

    leaves_a, leaves_b = state_leaves(state_a), state_leaves(state_b)
    if len(leaves_a) != len(leaves_b):
        raise ConfigError(f"{where}: the state structure depends on the batch size "
                          f"({len(leaves_a)} tensors for B={batches[0]}, {len(leaves_b)} for B={batches[1]})")
    for i, (a, b) in enumerate(zip(leaves_a, leaves_b, strict=True)):
        if a.dim() == 0 or b.dim() == 0 or a.shape[0] != batches[0] or b.shape[0] != batches[1] \
                or a.shape[1:] != b.shape[1:]:
            raise ConfigError(
                f"{where}: every state tensor must have the batch dimension first; state tensor #{i} "
                f"has shape {tuple(a.shape)} for B={batches[0]} and {tuple(b.shape)} for B={batches[1]} "
                f"(expected ({batches[0]}, ...) and ({batches[1]}, ...) with identical other dims). "
                f"Store RNN states as [B, num_layers, H], not nn.LSTM/nn.GRU's native [num_layers, B, H]."
            )


def _check_state_payload(state: Any, where: str) -> None:
    """The model state must survive the inter-process payload round trip (SP1): chunks carry
    ``initial_state`` to the learner as numpy over ``mp.Queue`` (pickle) or gRPC (``pack_payload``)."""
    from colosseum.networks.state import slice_batch, state_from_numpy, state_to_numpy
    from colosseum.transport.serialization import pack_payload, unpack_payload

    if state is None:
        return
    try:
        state_from_numpy(unpack_payload(*pack_payload(state_to_numpy(slice_batch(state, 0)))))
    except (TypeError, ValueError) as e:
        raise ConfigError(f"{where}: initial_state cannot be sent between processes: {e}") from e


def _as_spec(spec: ObsSpec, value: Any) -> Any:
    """``value`` with the space's leaf dtypes (as the worker's buffers store it)."""
    return tree_map(lambda zero, leaf: np.asarray(leaf, dtype=zero.dtype), spec.allocate(()), value)


def _repeat(tree: Any, n: int) -> Any:
    """Torch tree with ``n`` copies of a numpy tree along a new leading dim."""
    return None if tree is None else tree_to_torch(tree_stack([tree] * n))


def _check_model(model: Any, role: RoleSpec, sample: tuple | None, where: str) -> None:
    """``step`` on observations of the role and ``unroll`` on a synthetic chunk with BOOT/PAD
    slots, resets and ``global_state`` (spec block 9), plus SP1's state checks."""
    obs_spec = ObsSpec.from_space(role.observation_space)
    action_spec = ActionSpec.from_space(role.action_space)
    gs_spec = None if role.global_state_space is None else ObsSpec.from_space(role.global_state_space)
    obs1 = _as_spec(obs_spec, sample[0]) if sample is not None else obs_spec.allocate(())
    mask1 = None
    if action_spec.has_masks:
        mask1 = sample[1] if sample is not None and sample[1] is not None else action_spec.full_mask(())
    gs1 = None
    if gs_spec is not None:
        gs1 = _as_spec(gs_spec, sample[2]) if sample is not None and sample[2] is not None else gs_spec.allocate(())
    b, b_alt = 2, 3
    s = len(_CHUNK_KINDS)
    model.eval()
    with torch.no_grad():
        try:
            state0, state_alt = model.initial_state(b), model.initial_state(b_alt)
        except Exception as e:
            raise ConfigError(f"{where}: model.initial_state(B) failed: {type(e).__name__}: {e}") from e
        _check_state_batch_dim(state0, state_alt, (b, b_alt), f"{where}: initial_state")
        _check_state_payload(state0, where)
        try:
            out = model.step(_repeat(obs1, b), state0, _repeat(mask1, b))
            out_alt = model.step(_repeat(obs1, b_alt), state_alt, _repeat(mask1, b_alt))
        except Exception as e:
            raise ConfigError(f"{where}: model.step failed on a batch of the role's observations: "
                              f"{type(e).__name__}: {e}") from e
        if not isinstance(out.dist, Distribution):
            raise ConfigError(f"{where}: the policy must return a colosseum Distribution "
                              f"(colosseum.networks.dist), got {type(out.dist).__name__}")
        _check_state_batch_dim(out.state, out_alt.state, (b, b_alt), f"{where}: step() state")
        try:
            actions = out.dist.sample()
            log_probs = out.dist.log_prob(actions)
        except Exception as e:
            raise ConfigError(f"{where}: sampling from the policy distribution {type(out.dist).__name__} "
                              f"failed: {type(e).__name__}: {e}") from e
        expected = action_spec.allocate_actions((b,))
        if not tree_same_structure(actions, expected) or any(
                tuple(x.shape) != tuple(y.shape)
                for x, y in zip(tree_leaves(actions), tree_leaves(expected), strict=True)):
            raise ConfigError(f"{where}: the policy samples actions that do not match the action space "
                              f"{role.action_space!r}; check the policy head / distribution")
        if tuple(log_probs.shape) != (b,) or not torch.isfinite(log_probs).all():
            raise ConfigError(f"{where}: log_prob of sampled actions must be finite with shape [B]=({b},), "
                              f"got shape {tuple(log_probs.shape)}")

        kinds = torch.tensor(_CHUNK_KINDS)
        act_slot = kinds == SLOT_ACT
        reset_after = torch.tensor(_CHUNK_RESET_AFTER)
        one_action = tree_to_numpy(tree_index(actions, 0))
        zero_action = action_spec.allocate_actions(())
        boot_mask = action_spec.boot_mask()

        def seq(act_value: Any, other_value: Any) -> Any:
            if act_value is None:
                return None
            rows = [tree_stack([act_value if act_slot[t, j] else other_value for j in range(b)]) for t in range(s)]
            return tree_to_torch(tree_stack(rows))

        obs_seq = seq(obs1, obs1)
        mask_seq = seq(mask1, boot_mask)
        gs_seq = seq(gs1, gs1)
        flat_actions = tree_map(lambda x: x.reshape(s * b, *x.shape[2:]), seq(one_action, zero_action))
        try:
            unrolled = model.unroll(obs_seq, model.initial_state(b), reset_after, mask_seq,
                                    global_state=gs_seq, with_value=True)
            unroll_lp = unrolled.dist.log_prob(flat_actions)
            policy_only = model.unroll(obs_seq, model.initial_state(b), reset_after, mask_seq, with_value=False)
        except Exception as e:
            raise ConfigError(f"{where}: model.unroll failed on a synthetic [S={s}, B={b}] chunk with BOOT/PAD "
                              f"slots: {type(e).__name__}: {e}") from e
        if unrolled.value is None or tuple(unrolled.value.shape) != (s * b,):
            got = None if unrolled.value is None else tuple(unrolled.value.shape)
            raise ConfigError(f"{where}: unroll(with_value=True) must return time-major values [S*B]=({s * b},), "
                              f"got {got}. Squeeze the last dim in the value head.")
        if tuple(unroll_lp.shape) != (s * b,) or not torch.isfinite(unroll_lp[act_slot.reshape(-1)]).all():
            raise ConfigError(f"{where}: unroll gives non-finite or mis-shaped log-probs (shape "
                              f"{tuple(unroll_lp.shape)}, expected ({s * b},)) on ACT slots for actions sampled "
                              f"from step with the same masks; step and unroll must agree on the policy")
        if not torch.isfinite(unrolled.value[(kinds != SLOT_PAD).reshape(-1)]).all():
            raise ConfigError(f"{where}: unroll gives non-finite values on ACT or BOOT slots")
        if policy_only.value is not None:
            raise ConfigError(f"{where}: unroll(with_value=False) must return value=None (BC and the kickstart "
                              f"teacher run the policy path only)")


def _check_kickstart_teachers(config: ColosseumConfig, spec: GameSpec, agent_configs: dict[str, ColosseumConfig],
                              role_specs: dict[str, RoleSpec], samples: dict[str, tuple | None]) -> None:
    """Every trainable agent's kickstart teacher resolves (roles included); a neural one builds with its
    weights, can be unrolled on the student's chunks (state-layout rule), and the KL between teacher and
    student is finite on a synthetic batch of the role's observations."""
    from colosseum.learner.factory import build_teacher_model, check_teacher_compat, resolve_teacher

    for aid, acfg in agent_configs.items():
        teacher = resolve_teacher(config, aid, spec)
        if teacher is None or teacher.kind != "neural":
            continue
        where = f"agent {aid!r}: kickstart teacher {teacher.source}"
        teacher_model = build_teacher_model(acfg, teacher, spec)
        student = build_model(acfg, role_specs[aid])
        check_teacher_compat(student, teacher_model, where)
        _check_teacher_kl(student, teacher_model, role_specs[aid], samples.get(aid), where, teacher.kl)


def _check_teacher_kl(student: Any, teacher: Any, role: RoleSpec, sample: tuple | None, where: str,
                      direction: str) -> None:
    """The kickstart KL of ``teacher`` vs ``student`` on a synthetic [S=2, B=2] batch of the role's observations."""
    from colosseum.bc.kickstart import KickstartLoss

    obs_spec = ObsSpec.from_space(role.observation_space)
    action_spec = ActionSpec.from_space(role.action_space)
    obs1 = _as_spec(obs_spec, sample[0]) if sample is not None else obs_spec.allocate(())
    mask1 = None
    if action_spec.has_masks:
        mask1 = sample[1] if sample is not None and sample[1] is not None else action_spec.full_mask(())
    s, b = 2, 2

    def seq(value: Any) -> Any:
        return None if value is None else tree_to_torch(tree_stack([tree_stack([value] * b)] * s))

    obs_seq, mask_seq = seq(obs1), seq(mask1)
    reset_after = torch.zeros(s, b, dtype=torch.bool)
    student.eval()
    try:
        with torch.no_grad():
            state0 = student.initial_state(b)
            student_dist = student.unroll(obs_seq, state0, reset_after, mask_seq, with_value=False).dist
            loss = KickstartLoss(teacher, initial_lambda=1.0, direction=direction).compute(
                student_dist=student_dist, obs=obs_seq, reset_after=reset_after, state0=state0, action_mask=mask_seq,
                actions=student_dist.sample(), is_act=torch.ones(s, b, dtype=torch.bool), reduction="sum")
    except Exception as e:  # noqa: BLE001 - any failure means an incompatible teacher
        raise ConfigError(f"{where}: the kickstart KL between teacher and student failed ({type(e).__name__}: {e}); "
                          f"the teacher's action distribution must match the student's") from e
    if not torch.isfinite(loss):
        raise ConfigError(f"{where}: the kickstart KL between teacher and student is not finite on a synthetic batch")


def _check_frozen_players(config: ColosseumConfig, spec: GameSpec, fixed: Any, samples: dict[str, tuple],
                          report: ValidationReport) -> None:
    """Every frozen agent: its model is built and its weights loaded (``build_frozen_model``); each distinct
    architecture gets the model check of trainable agents once."""
    from colosseum.core.roles import agent_role_spec
    from colosseum.players.registry import build_frozen_model

    checked: set[str] = set()
    for aid, frozen in fixed.frozen.items():
        model = build_frozen_model(config, frozen, spec)
        key = json.dumps([frozen.networks, list(frozen.roles)], sort_keys=True)
        if key not in checked:
            sample = next((samples[r] for r in frozen.roles if r in samples), None)
            try:
                _check_model(model, agent_role_spec(spec, list(frozen.roles)), sample, f"agent {aid!r}")
            except ConfigError as e:
                raise ConfigError(f"{frozen.networks_source}: {e}") from e
            checked.add(key)
        report.lines.append(f"agent '{aid}' (frozen): {frozen.source}, roles {list(frozen.roles)}")


def _check_scripted_players(config: ColosseumConfig, spec: GameSpec, fixed: Any, layouts: Sequence[str],
                            report: ValidationReport) -> None:
    """Play every scripted agent in each enabled layout where it has a seat: one bot per such seat, other
    seats random legal, up to ``VALIDATE_STEPS`` steps under the contract checks; every bot action goes
    through ``check_bot_action``. A bot failure is a ConfigError naming the agent (SP2 context format)."""
    from colosseum.envs.contract import EpisodeTracker
    from colosseum.players.registry import make_bot
    from colosseum.players.scripted import bot_rng, check_bot_action

    for aid, bot_spec in fixed.bots.items():
        roles = set(fixed.roles[aid])
        played = [name for name in layouts if any(s.role in roles for s in spec.layouts[name])]
        decisions = 0
        rng = np.random.default_rng(0)
        env = make_env(config)
        try:
            for layout in played:
                tracker = EpisodeTracker(spec, max_idle_steps=config.env.max_idle_steps, context="validate")
                try:
                    result = env.reset(seed=0, layout=layout)
                except EnvContractError:
                    raise
                except Exception as e:
                    raise ConfigError(f"env.reset(seed=0, layout={layout!r}) of {config.env.env_class!r} failed: "
                                      f"{type(e).__name__}: {e}") from e
                masks = tracker.on_reset(layout, result)
                bots = {}
                for seat, seat_spec in enumerate(spec.layouts[layout]):
                    if seat_spec.role not in roles:
                        continue
                    bots[seat] = make_bot(bot_spec, spec)
                    try:
                        bots[seat].reset(role=seat_spec.role, seat=seat, layout=layout, rng=bot_rng(0, seat, aid))
                    except Exception as e:
                        raise ConfigError(f"validate, seat {seat}, episode step 0, layout {layout}: agent {aid!r}: "
                                          f"reset raised {type(e).__name__}: {e}") from e
                for step in range(VALIDATE_STEPS):
                    if result.episode_over:
                        break
                    actions = {}
                    for seat in sorted(result.acting):
                        role = spec.roles[spec.role_of(layout, seat)]
                        if seat not in bots:
                            actions[seat] = random_legal_action(role, masks[seat], rng)
                            continue
                        where = f"validate, seat {seat}, episode step {step}, layout {layout}: agent {aid!r}"
                        obs = tree_map(np.copy, _as_spec(ObsSpec.from_space(role.observation_space), result.obs[seat]))
                        mask = None if masks[seat] is None else tree_map(np.copy, masks[seat])  # as MatchRunner
                        try:
                            raw = bots[seat].act(obs, mask, (result.infos or {}).get(seat))
                        except Exception as e:
                            raise ConfigError(f"{where}: act raised {type(e).__name__}: {e}") from e
                        try:
                            actions[seat] = check_bot_action(role, raw, masks[seat], where)
                        except PlayerError as e:
                            raise ConfigError(str(e)) from e
                        decisions += 1
                    try:
                        result = env.step(actions)
                    except EnvContractError:
                        raise
                    except Exception as e:
                        raise ConfigError(f"env.step of {config.env.env_class!r} failed in layout {layout!r} at "
                                          f"episode step {step + 1}: {type(e).__name__}: {e}") from e
                    masks = tracker.on_step(actions, result)
        finally:
            env.close()
        what = (f"played {decisions} decisions in layouts {played}" if played
                else f"has no seat in the enabled layouts {list(layouts)}")
        report.lines.append(f"agent '{aid}' (scripted {bot_spec.class_path}): roles {sorted(roles)}; {what}")


def _game_spec(config: ColosseumConfig) -> GameSpec:
    """``env_spec``; a role space colosseum does not support (``ObsSpec``/``ActionSpec.from_space``
    raise TypeError/ValueError, which ``GameSpec.validate`` wraps) becomes a ConfigError."""
    try:
        return env_spec(config)
    except EnvContractError as e:
        if isinstance(e.__cause__, (TypeError, ValueError)):
            raise ConfigError(f"env {config.env.env_class!r}: {e}") from e
        raise


def _all_enabled_layouts(config: ColosseumConfig, spec: GameSpec) -> list[str]:
    """Layouts any trainable agent may play (per-agent ``matchmaking.layouts`` included), first-seen order."""
    from colosseum.league.lineups import enabled_layouts

    names: dict[str, None] = {}
    for aid in config.get_trainable_agent_ids():
        names.update(dict.fromkeys(enabled_layouts(spec, config.get_agent_config(aid).matchmaking)))
    return list(names)


def _check_matchmaker_class(config: ColosseumConfig, spec: GameSpec, player_roles: dict[str, list[str]]) -> None:
    """A custom ``matchmaking.matchmaker_class``: imports, constructs, and its lineups pass ``check_lineup``."""
    import random

    from colosseum.league.base import MatchmakerContext, load_matchmaker_class
    from colosseum.league.lineups import check_lineup

    path = config.matchmaking.matchmaker_class
    if path is None:
        return
    cls = load_matchmaker_class(path)
    context = MatchmakerContext.from_config(config, spec, player_roles, rng=random.Random(0))
    try:
        matchmaker = cls(context)
    except Exception as e:  # noqa: BLE001 - user code
        raise ConfigError(f"matchmaking.matchmaker_class {path!r}: constructing it failed: "
                          f"{type(e).__name__}: {e}") from e
    for owner in context.trainable:
        for _ in range(VALIDATE_LINEUPS):
            try:
                lineup = matchmaker.lineup_for(owner)
            except Exception as e:  # noqa: BLE001 - user code
                raise ConfigError(f"matchmaking.matchmaker_class {path!r}: lineup_for({owner!r}) raised "
                                  f"{type(e).__name__}: {e}") from e
            try:
                check_lineup(context, lineup, cls.__name__)
            except ValueError as e:
                raise ConfigError(f"matchmaking.matchmaker_class {path!r}: {e}") from e


def _check_critic_warmup(where: str, agent_config: ColosseumConfig, model: Any) -> None:
    """``init.critic_warmup_steps > 0`` needs a value loss, ``PolicyModel.value_parameters()`` and an algorithm
    class that takes ``critic_warmup_steps``."""
    import inspect

    from colosseum.core.registry import import_class

    steps = agent_config.init.critic_warmup_steps
    if steps <= 0:
        return
    if not agent_config.algorithm.value_loss_coeff > 0:
        raise ConfigError(f"{where}: init.critic_warmup_steps={steps} trains only the value loss; "
                          f"algorithm.value_loss_coeff must be > 0")
    try:
        params = list(model.value_parameters())
    except NotImplementedError as e:
        raise ConfigError(f"{where}: init.critic_warmup_steps={steps}: {e}") from e
    if not params:
        raise ConfigError(f"{where}: init.critic_warmup_steps={steps}: value_parameters() returned no parameters")
    path = agent_config.algorithm.algorithm_class
    if "critic_warmup_steps" not in inspect.signature(import_class(path)).parameters:
        raise ConfigError(f"{where}: init.critic_warmup_steps={steps}, but algorithm.algorithm_class {path!r} takes "
                          f"no critic_warmup_steps argument; use colosseum.algorithms.appo.APPO (or a subclass that "
                          f"accepts it) or set critic_warmup_steps: 0")


def validate_config(config: ColosseumConfig) -> ValidationReport:
    """Check a whole config before any process starts (spec block 9; SP3 spec block 1).

    - the env's ``GameSpec`` (``env_spec``; unsupported role spaces are a ConfigError), the
      agents' roles (``resolve_agent_roles``), the matchmaking checks per agent
      (``colosseum.league.mixture.validate_matchmaking``), a custom matchmaker's lineups, and the
      effective opponent mix of every agent (``ValidationReport.lines``);
    - ``networks.critic_encoder_class`` only for agents whose roles declare a ``global_state_space``;
    - ``reset`` of every enabled layout and up to ``VALIDATE_STEPS`` random legal steps under the
      contract checks, with full ``space.contains`` checks of observations, global states and
      final observations;
    - every agent's model: ``step`` on its role's observations and ``unroll`` on a synthetic chunk
      with BOOT/PAD slots, resets and ``global_state``;
    - the critic warm-up's requirements (``init.critic_warmup_steps``);
    - every agent's kickstart teacher (``learner.factory.resolve_teacher``): its roles, its weights, the
      state-layout rule (``learner.factory.check_teacher_compat``) and a finite teacher/student KL on a
      synthetic batch;
    - every agent's ``init`` source (strict / partial report in ``ValidationReport.lines``), except for
      agents ``training.resume_from`` restores (``learner.factory.resume_source_of``), which ignore it;
    - every frozen agent: weights, role signature and its architecture (once per architecture); every
      scripted agent: imported, constructed and played in each enabled layout where it has a seat, every
      action through the legality gate.

    Returns a ``ValidationReport``. Raises ConfigError (or EnvContractError for an env that breaks the
    contract).
    """
    from colosseum.core.roles import agent_role_spec, resolve_agent_roles
    from colosseum.league.mixture import describe_mix, validate_matchmaking
    from colosseum.learner.factory import resolve_init, resume_source_of
    from colosseum.players.registry import load_fixed_players, resolve_player_roles

    report = ValidationReport()
    spec = _game_spec(config)
    agent_roles = resolve_agent_roles(config, spec)
    player_roles = resolve_player_roles(config, spec)
    validate_matchmaking(spec, player_roles, config)
    _check_matchmaker_class(config, spec, player_roles)
    if config.matchmaking.matchmaker_class is not None:
        report.lines.append(f"matchmaker: {config.matchmaking.matchmaker_class} (custom; the mix below is the "
                            f"configured matchmaking section, which it may ignore)")
    report.lines.extend(describe_mix(config, spec))
    fixed = load_fixed_players(config, spec)   # roles, classes, paths and role signatures of fixed agents
    agent_configs = {aid: config.get_agent_config(aid) for aid in agent_roles}
    role_specs = {aid: agent_role_spec(spec, roles) for aid, roles in agent_roles.items()}
    for aid, acfg in agent_configs.items():
        if acfg.networks.critic_encoder_class and role_specs[aid].global_state_space is None:
            raise ConfigError(
                f"agent {aid!r}: networks.critic_encoder_class is set, but its roles {agent_roles[aid]} declare "
                f"no global_state_space; remove critic_encoder_class or give the roles a global_state_space"
            )
    layouts = _all_enabled_layouts(config, spec)
    samples = _exercise_env(config, spec, layouts)
    agent_samples: dict[str, tuple | None] = {}
    for aid, acfg in agent_configs.items():
        where = f"agent {aid!r}"
        try:
            model = build_model(acfg, role_specs[aid])
        except ConfigError:
            raise
        except Exception as e:
            raise ConfigError(f"{where}: failed to build the model from networks: {type(e).__name__}: {e}") from e
        sample = next((samples[r] for r in agent_roles[aid] if r in samples), None)
        agent_samples[aid] = sample
        _check_model(model, role_specs[aid], sample, where)
        _check_critic_warmup(where, acfg, model)
    _check_kickstart_teachers(config, spec, agent_configs, role_specs, agent_samples)
    for aid, acfg in agent_configs.items():
        resumed = resume_source_of(config, aid)
        if resumed is not None:   # spec block 6: an agent restored by resume_from ignores init
            if acfg.init.from_ is not None:
                report.lines.append(f"agent {aid!r}: resumes from {resumed}; init.from ignored")
            continue
        init = resolve_init(config, aid, spec)
        if init is not None:
            report.lines.extend(init.report)
    if config.training.resume_from:
        report.lines.append("training.resume_from is set: agents restored from it ignore init")
    _check_frozen_players(config, spec, fixed, samples, report)
    _check_scripted_players(config, spec, fixed, layouts, report)
    return report

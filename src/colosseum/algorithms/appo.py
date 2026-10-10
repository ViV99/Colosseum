"""APPO on chunk v2: async PPO with V-trace over act / boot / pad slots (spec block 6).

- The learner's network computes every value (bootstrap included) with ``PolicyModel.unroll``
  over all slots of the batch; BOOT slots exist only for that.
- Every loss and diagnostic reduces over ACT slots only (``torch.where``, never a mask product).
- ``K`` = ``ActionSpec.num_deciders``. Three switches (``AlgorithmConfig``):
  - ``unit_trace`` (``auto | joint | geo_mean | none``): the scalar log-ratio fed to V-trace:
    sum of the per-decider log-ratios, their mean over valid deciders, or 0 with
    rho = c = 1 (TD(lambda)). ``auto`` = ``joint`` (with ``Units`` too: the units-experiment
    ruling of T8.4, docs/benchmarks.md).
  - ``ratio_mode`` (``auto | joint | per_unit``): ``joint`` = PPO-clip on the joint ratio with
    the advantage multiplied by the clipped scalar rho (SP1); ``per_unit`` = PPO-clip per
    decider on the shared advantage WITHOUT the rho factor, mean over valid deciders, then
    mean over ACT slots. ``auto`` = ``per_unit`` with ``Units``, else ``joint``.
  - ``entropy_reduction`` (``auto | mean_valid | sum``): how entropy and kickstart KL reduce
    over deciders; ``auto`` = ``sum`` for ``joint``, ``mean_valid`` for ``per_unit``.
  With ``K == 1`` (one decider, where per-decider and joint quantities coincide) the modes
  collapse to the joint path, which keeps SP1's rho-weighted advantage: ``ratio_mode`` is
  ``joint`` (an explicit ``per_unit`` is logged once at INFO, since it would differ from
  ``joint`` only by the rho factor), ``entropy_reduction`` is ``sum`` and ``unit_trace`` is
  ``joint`` (``geo_mean`` is identical at ``K == 1``). An explicit ``unit_trace: none`` is
  honoured: rho = c = 1 is defined independently of ``K``.
- Observation normalizers update once per train step from ACT slots and BOOT slots with
  ``reset_after`` (a chunk-end BOOT repeats the next chunk's first ACT).
- Critic warm-up (``critic_warmup_steps``, spec block 6 of SP3): the first N train steps optimize only
  ``value_loss_coeff * value_loss`` over ``PolicyModel.value_parameters()``; every other parameter has
  ``requires_grad`` off for the step (no gradient, no Adam update or state), the statistics of the
  observation normalizers are frozen (the policy stays bit-identical) while the value path's global-state
  normalizers keep learning (the critic's inputs do not jump at the end of the warm-up), kickstart is off and
  its decay waits; policy loss and entropy are still reported.
- Kickstart from a scripted teacher (``KickstartLoss(None)``, DAgger): ``lambda * mean(-log pi(a_teacher))`` over
  the labeled ACT slots, joint for one decider and the mean over valid deciders with ``Units``; metric
  ``kickstart_label_frac``. ``teacher_active`` (lambda > 0) goes to the workers with the weights.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

import torch
from torch import Tensor

from colosseum.algorithms.base import BaseAlgorithm, deep_cpu_copy
from colosseum.algorithms.vtrace import compute_vtrace_slots
from colosseum.bc.kickstart import KickstartLoss, check_teacher_state_layout
from colosseum.core.config import AlgorithmConfig, LRSchedule
from colosseum.core.specs import ActionSpec
from colosseum.core.tree import tree_map, tree_stack
from colosseum.core.types import SLOT_ACT, SLOT_BOOT, SLOT_PAD, TrajectoryChunk
from colosseum.networks.model import PolicyModel, UnrollOutput
from colosseum.networks.state import cat_batch, slice_batch, state_to

logger = logging.getLogger(__name__)

RatioMode = Literal["joint", "per_unit"]
UnitTrace = Literal["joint", "geo_mean", "none"]
EntropyReduction = Literal["mean_valid", "sum"]


# Set once the K == 1 collapse of an explicit ratio_mode "per_unit" has been logged (once per process).
_per_unit_collapse_logged = False


def resolve_modes(config: AlgorithmConfig, action_spec: ActionSpec) -> tuple[RatioMode, UnitTrace, EntropyReduction]:
    """Resolve ``auto`` values and the ``K == 1`` collapse into ``(ratio_mode, unit_trace, entropy_reduction)``.

    ``K > 1`` (only with ``Units``): explicit values apply; ``auto`` gives ``per_unit`` /
    ``joint`` / (``mean_valid`` for ``per_unit``, ``sum`` for ``joint``). ``unit_trace: auto`` is
    ``joint`` for every action (T8.4 units-experiment ruling, docs/benchmarks.md).

    ``K == 1``: ``("joint", "joint", "sum")`` for every configuration except an explicit
    ``unit_trace: none``, which gives ``("joint", "none", "sum")``. An explicit
    ``ratio_mode: per_unit`` is logged once (INFO); explicit ``geo_mean`` / ``mean_valid``
    collapse silently (identical to ``joint`` / ``sum`` with one decider).
    """
    global _per_unit_collapse_logged
    if action_spec.num_deciders == 1:
        if config.ratio_mode == "per_unit" and not _per_unit_collapse_logged:
            _per_unit_collapse_logged = True
            logger.info("one decider: per_unit equals joint except for the rho factor; using joint")
        return "joint", ("none" if config.unit_trace == "none" else "joint"), "sum"
    ratio: RatioMode = config.ratio_mode if config.ratio_mode != "auto" else (
        "per_unit" if action_spec.has_units else "joint")
    trace: UnitTrace = config.unit_trace if config.unit_trace != "auto" else "joint"
    reduction: EntropyReduction = config.entropy_reduction if config.entropy_reduction != "auto" else (
        "sum" if ratio == "joint" else "mean_valid")
    return ratio, trace, reduction


# Layout of every key ``APPO._prepare_batch`` produces (``_select_chunks`` rejects others).
_TIME_MAJOR_KEYS = frozenset({
    "obs", "global_state", "actions", "action_masks", "kind", "reward", "terminal", "reset_after",
    "behavior_logp", "behavior_unit_logp", "teacher_action", "has_teacher",
})                                   # trees / tensors with leaves [S, B, ...]
_STATE_KEY = "initial_state"         # State pytree, leaves [B, ...]


def _select_chunks(batch: dict, idx: Tensor) -> dict:
    """Rows ``idx`` of the chunk dimension of a prepared batch (``[S, B, ...]`` -> ``[S, b, ...]``)."""
    idx = idx.to(batch["kind"].device)
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if key == _STATE_KEY:
            out[key] = slice_batch(value, idx)
        elif key in _TIME_MAJOR_KEYS:
            out[key] = None if value is None else tree_map(lambda t: t[:, idx], value)
        else:
            raise KeyError(f"_select_chunks: batch key {key!r} has no declared layout")
    return out


def _act_mean(x: Tensor, is_act: Tensor, n_act: Tensor) -> Tensor:
    """Mean of ``x`` over ACT slots (``where``-selected: values elsewhere may be NaN)."""
    return torch.where(is_act, x, torch.zeros_like(x)).sum() / n_act


def _explained_variance(predicted: Tensor, target: Tensor, is_act: Tensor, n_act: Tensor) -> Tensor:
    """1 - Var(target - predicted) / Var(target) over ACT slots; 0 for a (near) constant target."""
    def var(x: Tensor) -> Tensor:
        mean = _act_mean(x, is_act, n_act)
        return _act_mean((x - mean) ** 2, is_act, n_act)

    var_target = var(target.float())
    ev = 1.0 - var(target.float() - predicted.float()) / var_target.clamp(min=1e-12)
    return torch.where(var_target < 1e-8, torch.zeros_like(ev), ev)   # no host sync


class APPO(BaseAlgorithm):
    """Async PPO with V-trace over chunk v2 slots (module docstring)."""

    def __init__(
        self,
        model: PolicyModel,
        config: AlgorithmConfig,
        action_spec: ActionSpec,
        device: str | torch.device = "cpu",
        pin_memory: bool = False,
        kickstart: KickstartLoss | None = None,
        critic_warmup_steps: int = 0,
    ):
        self._model = model.to(device)
        if kickstart is not None:
            if kickstart.teacher is not None:
                check_teacher_state_layout(self._model, kickstart.teacher)
            kickstart.to(device)
        self._config = config
        self._action_spec = action_spec
        self._num_deciders = action_spec.num_deciders
        self._ratio_mode, self._unit_trace, self._entropy_reduction = resolve_modes(config, action_spec)
        self._device = device
        self._policy_version = 0
        self._consumed_samples = 0
        self._kickstart = kickstart
        self._pin_memory = pin_memory

        # AMP (automatic mixed precision)
        self._use_amp = config.use_amp and str(device).startswith("cuda")
        self._amp_dtype = getattr(torch, config.amp_dtype, torch.float16)
        self._scaler = torch.amp.GradScaler("cuda") if self._use_amp else None

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=config.learning_rate)

        # Critic warm-up (SP3 spec block 6): the first N train steps update only the value path.
        if critic_warmup_steps < 0:
            raise ValueError(f"critic_warmup_steps must be >= 0, got {critic_warmup_steps}")
        self._critic_warmup_steps = int(critic_warmup_steps)
        self._critic_warmup_done = 0
        self._frozen_in_warmup: list[torch.nn.Parameter] = []
        if self._critic_warmup_steps > 0:
            if not config.value_loss_coeff > 0:
                raise ValueError("critic_warmup_steps > 0 needs algorithm.value_loss_coeff > 0 "
                                 "(the warm-up trains only the value loss)")
            try:
                value_params = list(self._model.value_parameters())
            except NotImplementedError as e:
                raise ValueError(f"critic_warmup_steps={critic_warmup_steps} needs "
                                 f"PolicyModel.value_parameters(): {e}") from e
            if not value_params:
                raise ValueError(f"critic_warmup_steps={critic_warmup_steps}: "
                                 f"{type(self._model).__name__}.value_parameters() returned no parameters")
            value_ids = {id(p) for p in value_params}
            self._frozen_in_warmup = [p for p in self._model.parameters() if id(p) not in value_ids]
        self._zero_loss = torch.tensor(0.0, device=device)
        self._compute_vtrace = (
            torch.compile(compute_vtrace_slots) if config.use_torch_compile else compute_vtrace_slots
        )

        # The LR is a function of training progress (share of the global env-step budget),
        # set by the learner through set_progress() before every train step.
        self._progress = 0.0
        self.set_progress(0.0)

    @property
    def model(self) -> PolicyModel:
        return self._model

    @property
    def policy_version(self) -> int:
        return self._policy_version

    @property
    def consumed_samples(self) -> int:
        """Total ACT slots passed to train_step."""
        return self._consumed_samples

    @property
    def modes(self) -> tuple[str, str, str]:
        """Resolved ``(ratio_mode, unit_trace, entropy_reduction)``."""
        return self._ratio_mode, self._unit_trace, self._entropy_reduction

    @property
    def teacher_active(self) -> bool:
        """A scripted kickstart teacher with lambda > 0: the workers keep labelling decisions."""
        k = self._kickstart
        return k is not None and k.teacher is None and k.current_lambda > 0

    @property
    def critic_warming_up(self) -> bool:
        """True during the first ``critic_warmup_steps`` train steps (value path only)."""
        return self._critic_warmup_done < self._critic_warmup_steps

    def set_progress(self, progress: float) -> None:
        """Set the share (0..1) of the global env-step budget consumed so far (drives the LR)."""
        self._progress = min(1.0, max(0.0, float(progress)))
        lr = self._lr_at(self._progress)
        for group in self._optimizer.param_groups:
            group["lr"] = lr

    def _lr_at(self, progress: float) -> float:
        base = self._config.learning_rate
        if self._config.lr_schedule == LRSchedule.LINEAR:
            return base * (1.0 - progress)
        if self._config.lr_schedule == LRSchedule.COSINE:
            return base * 0.5 * (1.0 + math.cos(math.pi * progress))
        return base

    # ------------------------------------------------------------------
    # Batches
    # ------------------------------------------------------------------

    def _prepare_batch(self, chunks: list[TrajectoryChunk]) -> dict[str, Any]:
        """Stack chunks into ``[S, B, ...]`` trees and move them to the device once."""
        device = self._device
        use_pinning = self._pin_memory and str(device).startswith("cuda")

        def stack(get) -> Any:
            first = get(chunks[0])
            if first is None:
                return None
            return tree_stack([get(c) for c in chunks], axis=1)

        teacher_action, has_teacher = self._stack_labels(chunks)
        batch_cpu = {
            "obs": stack(lambda c: c.obs),
            "global_state": stack(lambda c: c.global_state),
            "actions": stack(lambda c: c.actions),
            "action_masks": stack(lambda c: c.action_masks),
            "kind": stack(lambda c: c.kind),
            "reward": stack(lambda c: c.reward),
            "terminal": stack(lambda c: c.terminal),
            "reset_after": stack(lambda c: c.reset_after),
            "behavior_logp": stack(lambda c: c.behavior_logp),
            "behavior_unit_logp": stack(lambda c: c.behavior_unit_logp),
            "teacher_action": teacher_action,
            "has_teacher": has_teacher,
        }

        def move(t: Tensor) -> Tensor:
            if use_pinning:
                return t.pin_memory().to(device, non_blocking=True)
            return t.to(device)

        batch: dict[str, Any] = {k: None if v is None else tree_map(move, v) for k, v in batch_cpu.items()}
        batch["initial_state"] = state_to(cat_batch([c.initial_state for c in chunks]), device)
        return batch

    @staticmethod
    def _stack_labels(chunks: list[TrajectoryChunk]) -> tuple[Any, Tensor | None]:
        """Teacher labels ``[S, B, ...]`` and ``has_teacher [S, B]`` (None if no chunk has labels). Chunks
        without labels (parked buffers, the delay of the teacher flag) get zeros and False."""
        template = next((c for c in chunks if c.has_teacher is not None), None)
        if template is None:
            return None, None
        actions = [c.teacher_action if c.has_teacher is not None
                   else tree_map(torch.zeros_like, template.teacher_action) for c in chunks]
        flags = [c.has_teacher if c.has_teacher is not None else torch.zeros_like(template.has_teacher)
                 for c in chunks]
        return tree_stack(actions, axis=1), torch.stack(flags, dim=1)

    def _label_actions(self, batch: dict) -> Any:
        """Teacher actions ``[S*B, ...]`` where a slot is labeled, else the recorded action (always legal under
        the slot's mask, so every log-prob is finite)."""
        S, B = batch["kind"].shape
        labeled = batch["has_teacher"].bool().reshape(S * B)
        teacher = tree_map(lambda t: t.reshape(S * B, *t.shape[2:]), batch["teacher_action"])
        return tree_map(lambda t, a: torch.where(labeled.reshape(-1, *([1] * (t.dim() - 1))), t, a),
                        teacher, self._flat_actions(batch))

    def _autocast(self) -> torch.autocast:
        """AMP autocast context (a no-op unless AMP is enabled on CUDA)."""
        return torch.autocast(device_type="cuda", dtype=self._amp_dtype, enabled=self._use_amp)

    def _flat_actions(self, batch: dict) -> Any:
        S, B = batch["kind"].shape
        return tree_map(lambda t: t.reshape(S * B, *t.shape[2:]), batch["actions"])

    def _evaluate(self, batch: dict) -> tuple[UnrollOutput, dict[str, Tensor]]:
        """The one evaluation path (training and ``evaluate_chunks``): unroll the model over
        every slot under AMP autocast, from the chunks' initial states with the chunks'
        ``reset_after`` flags, masks and ``global_state``.

        Returns the ``UnrollOutput`` and float32 tensors: ``log_probs`` / ``values`` ``[S, B]``,
        ``unit_log_probs`` / ``unit_entropy`` ``[S, B, K]`` and bool ``unit_valid`` ``[S, B, K]``.
        """
        S, B = batch["kind"].shape
        K = self._num_deciders
        actions = self._flat_actions(batch)
        with self._autocast():
            out = self._model.unroll(
                batch["obs"], batch["initial_state"], batch["reset_after"].bool(), batch["action_masks"],
                global_state=batch["global_state"], with_value=True,
            )
            dist = out.dist
            evals = {
                "log_probs": dist.log_prob(actions).float().reshape(S, B),
                "values": out.value.float().reshape(S, B),
                "unit_log_probs": dist.unit_log_prob(actions).float().reshape(S, B, K),
                "unit_entropy": dist.unit_entropy(actions).float().reshape(S, B, K),
                "unit_valid": dist.unit_valid(actions).bool().reshape(S, B, K),
            }
        return out, evals

    @torch.no_grad()
    def evaluate_chunks(self, chunks: list[TrajectoryChunk]) -> tuple[Tensor, Tensor, Tensor | None]:
        """Current model's log-probs of the recorded actions, its values, and (``K > 1``)
        per-decider log-probs: ``[S*B]``, ``[S*B]``, ``[S*B, K]`` in time-major order
        (index ``s*B + b`` is slot ``s`` of ``chunks[b]``), computed exactly as in training.
        """
        _, ev = self._evaluate(self._prepare_batch(chunks))
        unit = ev["unit_log_probs"].reshape(-1, self._num_deciders) if self._num_deciders > 1 else None
        return ev["log_probs"].reshape(-1), ev["values"].reshape(-1), unit

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, chunks: list[TrajectoryChunk]) -> dict[str, Tensor]:
        """APPO loss and diagnostics for one minibatch of chunks (see ``_loss_from_batch``)."""
        return self._loss_from_batch(self._prepare_batch(chunks), warmup=self.critic_warming_up)

    def _loss_from_batch(self, batch: dict, warmup: bool = False) -> dict[str, Tensor]:
        """APPO loss for one prepared minibatch (``_prepare_batch`` output or a slice of it).

        1. Unroll the model over every slot (``_evaluate``).
        2. Per-decider and joint log-ratios on ACT slots; the scalar V-trace log-ratio from
           ``unit_trace``.
        3. V-trace(lambda) over the slots (values of the learner's network everywhere).
        4. Policy loss per ``ratio_mode``; value MSE; entropy per ``entropy_reduction``.
        5. Optional kickstart KL with the same reduction.
        6. Diagnostics over ACT slots.

        ``warmup``: the critic warm-up's loss (value only, no kickstart).
        """
        cfg = self._config
        K = self._num_deciders
        out, ev = self._evaluate(batch)
        kind = batch["kind"]
        is_act = kind == SLOT_ACT                                   # [S, B]
        n_act = is_act.sum().clamp(min=1).float()
        valid = ev["unit_valid"] & is_act.unsqueeze(-1)             # [S, B, K]
        n_valid = valid.sum(-1)                                     # [S, B]
        n_valid_f = n_valid.clamp(min=1).float()
        zeros_sb = torch.zeros_like(ev["values"])
        zeros_sbk = torch.zeros_like(ev["unit_log_probs"])

        behavior_unit = batch["behavior_unit_logp"] if K > 1 else batch["behavior_logp"].unsqueeze(-1)
        unit_log_ratio = torch.where(
            valid, torch.clamp(ev["unit_log_probs"] - behavior_unit, -20.0, 20.0), zeros_sbk)
        joint_log_ratio = torch.where(
            is_act, torch.clamp(ev["log_probs"] - batch["behavior_logp"], -20.0, 20.0), zeros_sb)

        # Scalar log-ratio for V-trace (unit_trace).
        with torch.no_grad():
            rho_bar, c_bar = cfg.vtrace_rho_bar, cfg.vtrace_c_bar
            if self._unit_trace == "joint":
                trace_log_rho = joint_log_ratio.detach()
            elif self._unit_trace == "geo_mean":
                trace_log_rho = unit_log_ratio.detach().sum(-1) / n_valid_f
            else:                                                   # "none": rho = c = 1
                trace_log_rho, rho_bar, c_bar = zeros_sb, 1.0, 1.0
            vt = self._compute_vtrace(
                log_rhos=trace_log_rho, rewards=batch["reward"], values=ev["values"].detach(), is_act=is_act,
                terminal=batch["terminal"].bool(), gamma=cfg.gamma, rho_bar=rho_bar, c_bar=c_bar,
                lam=cfg.vtrace_lambda,
            )

        # Advantages: joint mode weights them with the clipped scalar rho, per_unit does not.
        adv = vt.td * vt.clipped_rho if self._ratio_mode == "joint" else vt.td
        if cfg.normalize_advantages:
            mean = _act_mean(adv, is_act, n_act)
            var = _act_mean((adv - mean) ** 2, is_act, n_act) * n_act / (n_act - 1).clamp(min=1)
            normalized = (adv - mean) / (var.sqrt() + 1e-8)
            adv = torch.where(n_act > 1, normalized, adv)
        adv = torch.where(is_act, adv, zeros_sb).detach()

        eps = cfg.eps_clip
        if self._ratio_mode == "joint":
            ratio = torch.exp(joint_log_ratio)
            surr = torch.min(ratio * adv, torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv)
            policy_loss = -_act_mean(surr, is_act, n_act)
        else:
            unit_ratio = torch.exp(unit_log_ratio)
            unit_adv = adv.unsqueeze(-1)
            unit_surr = torch.min(unit_ratio * unit_adv, torch.clamp(unit_ratio, 1.0 - eps, 1.0 + eps) * unit_adv)
            step_surr = torch.where(valid, unit_surr, zeros_sbk).sum(-1) / n_valid_f
            policy_loss = -_act_mean(step_surr, is_act, n_act)

        value_loss = _act_mean((ev["values"] - vt.vs.detach()) ** 2, is_act, n_act)
        unit_entropy = torch.where(valid, ev["unit_entropy"], zeros_sbk).sum(-1)
        if self._entropy_reduction == "mean_valid":
            unit_entropy = unit_entropy / n_valid_f
        entropy = _act_mean(unit_entropy, is_act, n_act)
        if warmup:                                   # critic warm-up: the value path only
            total_loss = cfg.value_loss_coeff * value_loss
        else:
            total_loss = policy_loss + cfg.value_loss_coeff * value_loss - cfg.entropy_coeff * entropy

        kickstart_loss = self._zero_loss
        label_frac = self._zero_loss
        kick = self._kickstart
        if kick is not None and kick.current_lambda > 0 and not warmup:
            if kick.teacher is None:                         # scripted teacher (DAgger): labels in the chunks
                if batch["has_teacher"] is not None:
                    label_frac = (batch["has_teacher"].bool() & is_act).sum().float() / n_act
                    with self._autocast():
                        kickstart_loss = kick.compute_labels(
                            student_dist=out.dist, teacher_actions=self._label_actions(batch),
                            has_teacher=batch["has_teacher"], is_act=is_act,
                            reduction="sum" if K == 1 else "mean_valid",
                        )
            else:
                with self._autocast():
                    kickstart_loss = kick.compute(
                        student_dist=out.dist, obs=batch["obs"], reset_after=batch["reset_after"],
                        state0=batch["initial_state"], action_mask=batch["action_masks"],
                        actions=self._flat_actions(batch), is_act=is_act, reduction=self._entropy_reduction,
                    )
            total_loss = total_loss + kickstart_loss

        result = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            **self._diagnostics(batch, ev, is_act, n_act, valid, n_valid, unit_log_ratio, joint_log_ratio,
                                trace_log_rho, vt.vs),
        }
        if kick is not None:
            result["kickstart_loss"] = kickstart_loss.detach()
            result["kickstart_lambda"] = torch.tensor(kick.current_lambda, device=total_loss.device)
            if kick.teacher is None:                         # every minibatch, so the step mean is right
                result["kickstart_label_frac"] = label_frac.detach()
        return result

    @torch.no_grad()
    def _diagnostics(self, batch, ev, is_act, n_act, valid, n_valid, unit_log_ratio, joint_log_ratio,
                     trace_log_rho, vs) -> dict[str, Tensor]:
        cfg = self._config
        eps = cfg.eps_clip
        n_dec = valid.sum().clamp(min=1).float()
        kind = batch["kind"]
        unit_ratio = torch.exp(unit_log_ratio)
        joint_ratio = torch.exp(joint_log_ratio)
        rho = torch.where(is_act, torch.exp(torch.clamp(trace_log_rho, -20.0, 20.0)), torch.zeros_like(trace_log_rho))
        abs_unit = unit_log_ratio.abs()
        nan = torch.full_like(abs_unit, float("nan"))
        sum_rho = rho.sum()
        sum_rho2 = (rho * rho).sum()

        def frac_dec(x: Tensor) -> Tensor:
            return torch.where(valid, x.float(), torch.zeros_like(abs_unit)).sum() / n_dec

        return {
            "approx_kl": frac_dec((unit_ratio - 1) - unit_log_ratio),
            "clip_fraction": frac_dec((unit_ratio - 1.0).abs() > eps),
            "clip_fraction_joint": _act_mean(((joint_ratio - 1.0).abs() > eps).float(), is_act, n_act),
            "rho_mean": _act_mean(rho, is_act, n_act),
            "rho_clip_frac": _act_mean((rho > cfg.vtrace_rho_bar).float(), is_act, n_act),
            "c_clip_frac": _act_mean((rho > cfg.vtrace_c_bar).float(), is_act, n_act),
            "log_rho_abs_mean": frac_dec(abs_unit),
            "log_rho_abs_p95": torch.nanquantile(torch.where(valid, abs_unit, nan).flatten(), 0.95),
            "log_rho_joint_abs_mean": _act_mean(joint_log_ratio.abs(), is_act, n_act),
            "log_rho_joint_abs_p95": torch.nanquantile(
                torch.where(is_act, joint_log_ratio.abs(), torch.full_like(joint_log_ratio, float("nan"))).flatten(),
                0.95),
            "ess": sum_rho * sum_rho / (n_act * sum_rho2).clamp(min=1e-12),
            "deciders_valid_mean": _act_mean(n_valid.float(), is_act, n_act),
            "deciders_valid_max": torch.where(is_act, n_valid, torch.zeros_like(n_valid)).max().float(),
            "boot_frac": (kind == SLOT_BOOT).float().mean(),
            "pad_frac": (kind == SLOT_PAD).float().mean(),
            "explained_variance": _explained_variance(ev["values"], vs, is_act, n_act),
        }

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _clip_gradients(self) -> Tensor:
        """Clip to ``max_grad_norm``; return the total gradient norm BEFORE clipping."""
        return torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.max_grad_norm)

    def _update_normalizers(self, batch: dict, value_path_only: bool = False) -> None:
        """Once per train step: observations of ACT slots and of BOOT slots with ``reset_after``.

        ``value_path_only`` (the critic warm-up): only the global-state normalizers, which feed the value path
        alone; the observation normalizers (the policy's inputs) stay frozen."""
        kind = batch["kind"]
        sel = (kind == SLOT_ACT) | ((kind == SLOT_BOOT) & batch["reset_after"].bool())
        gs = None if batch["global_state"] is None else tree_map(lambda t: t[sel], batch["global_state"])
        if value_path_only:
            if gs is not None:
                self._model.update_normalizers(None, gs)
            return
        obs = tree_map(lambda t: t[sel], batch["obs"])
        self._model.update_normalizers(obs, gs)

    def train_step(self, chunks: list[TrajectoryChunk]) -> dict[str, float]:
        """One training step: normalizer update, then num_epochs x minibatches of updates.

        Minibatches are slices of the step's batch over the chunk dimension B, so each
        chunk's slot sequence stays intact. Metrics are minibatch means read back with one
        host sync; ``grad_norm`` averages the finite pre-clipping norms; ``skipped_updates``
        counts the minibatches whose step the GradScaler skipped.
        """
        cfg = self._config
        warmup = self.critic_warming_up
        sums: dict[str, Tensor] = {}
        num_updates = 0

        full_batch = self._prepare_batch(chunks)
        grad_norm_sum = torch.zeros((), dtype=torch.float64, device=full_batch["kind"].device)
        finite_updates = torch.zeros_like(grad_norm_sum)
        # During the warm-up only the value path's (global-state) normalizers learn: the workers' policy stays
        # bit-identical, and the critic's inputs do not jump when the warm-up ends.
        self._update_normalizers(full_batch, value_path_only=warmup)
        frozen = [p for p in self._frozen_in_warmup if p.requires_grad] if warmup else []
        for p in frozen:                 # no gradient: Adam updates neither them nor their state
            p.requires_grad_(False)

        mb_size = cfg.minibatch_chunks if cfg.minibatch_chunks > 0 else len(chunks)
        try:
            for _epoch in range(cfg.num_epochs):
                indices = torch.randperm(len(chunks))
                for start in range(0, len(chunks), mb_size):
                    losses = self._loss_from_batch(_select_chunks(full_batch, indices[start:start + mb_size]),
                                                   warmup=warmup)
                    total_loss = losses["total_loss"]

                    self._optimizer.zero_grad()
                    if self._scaler is not None:
                        self._scaler.scale(total_loss).backward()
                        self._scaler.unscale_(self._optimizer)
                        grad_norm = self._clip_gradients()
                        self._scaler.step(self._optimizer)      # skipped if the gradients are not finite
                        self._scaler.update()
                    else:
                        total_loss.backward()
                        grad_norm = self._clip_gradients()
                        self._optimizer.step()

                    finite = torch.isfinite(grad_norm)
                    grad_norm_sum += torch.where(finite, grad_norm.detach().double(), 0.0)
                    finite_updates += finite
                    for key, value in losses.items():
                        value = value.detach().double()
                        sums[key] = sums[key] + value if key in sums else value
                    num_updates += 1
        finally:
            for p in frozen:
                p.requires_grad_(True)

        if self._kickstart is not None and not warmup:
            self._kickstart.step()           # the decay starts after the warm-up
        if warmup:
            self._critic_warmup_done += 1
        self._policy_version += 1
        self._consumed_samples += sum(c.num_acts for c in chunks)

        keys = list(sums)
        *totals, grad_norm_total, num_finite = torch.stack(
            [sums[k] for k in keys] + [grad_norm_sum, finite_updates]
        ).tolist()
        metrics = {k: v / max(1, num_updates) for k, v in zip(keys, totals)}
        metrics["grad_norm"] = grad_norm_total / num_finite if num_finite > 0 else float("nan")
        metrics["skipped_updates"] = float(num_updates - num_finite) if self._scaler is not None else 0.0
        metrics["policy_version"] = float(self._policy_version)
        metrics["lr"] = float(self._optimizer.param_groups[0]["lr"])
        metrics["critic_warmup"] = 1.0 if warmup else 0.0
        return metrics

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Deep CPU copy of the training state (model weights excluded).

        Keys: ``optimizer``, ``progress``, ``scaler`` (None without AMP), ``kickstart`` (None
        without kickstart), ``policy_version``, ``consumed_samples``, ``critic_warmup_done``.
        """
        return deep_cpu_copy({
            "optimizer": self._optimizer.state_dict(),
            "progress": float(self._progress),
            "scaler": self._scaler.state_dict() if self._scaler is not None else None,
            "kickstart": self._kickstart.state_dict() if self._kickstart is not None else None,
            "policy_version": int(self._policy_version),
            "consumed_samples": int(self._consumed_samples),
            "critic_warmup_done": int(self._critic_warmup_done),
        })

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore :meth:`state_dict` output (SP1 semantics, warnings on scaler/kickstart mismatch)."""
        state = deep_cpu_copy(state)
        self._optimizer.load_state_dict(state["optimizer"])
        self._load_optional_state("GradScaler", self._scaler, state["scaler"])
        self._load_optional_state("kickstart", self._kickstart, state["kickstart"])
        self._policy_version = int(state["policy_version"])
        self._consumed_samples = int(state["consumed_samples"])
        # absent in trainer states written before SP3: the warm-up restarts with the fresh optimizer
        self._critic_warmup_done = int(state.get("critic_warmup_done", 0))
        self.set_progress(float(state["progress"]))

    @staticmethod
    def _load_optional_state(name: str, component: Any, saved: dict | None) -> None:
        """Restore an optional component (GradScaler, kickstart) and warn on a mismatch."""
        if component is not None and saved is not None:
            component.load_state_dict(saved)
        elif saved is not None:
            logger.warning("Resume: the saved %s state is ignored because this run has no %s.", name, name)
        elif component is not None:
            logger.warning("Resume: no %s state was saved; this run's %s starts fresh.", name, name)

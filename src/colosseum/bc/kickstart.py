"""Online behavioral cloning via kickstarting (Schmitt et al., 2018), on chunk v2.

Adds ``lambda * KL`` between a frozen teacher policy and the student to the RL loss.
Lambda decays linearly from ``initial_lambda`` to 0 over ``decay_steps`` train steps.

Direction (``kickstart.kl``):
- ``"forward"`` (default): KL(teacher || student), mode-covering (Kickstarting, AlphaStar, VPT);
- ``"reverse"``: KL(student || teacher), mode-seeking.

The teacher is unrolled with ``with_value=False`` over the same ``[S, B]`` slots as the
student, with the chunks' action masks, ``reset_after`` flags and initial states. The KL is
computed per decider (``Distribution.unit_kl``, 0 where a decider is invalid), reduced over
deciders per ``reduction`` (``"sum"`` or ``"mean_valid"``) and averaged over ACT slots only.

Teachers are per agent (SP3) and may have any architecture. A stateless teacher is unrolled with
``state0=None``; a recurrent teacher must share the student's state layout
(``check_teacher_state_layout``) and gets the chunk's ``initial_state`` (recorded by the student's
behavior policy): exact when teacher == student, an approximation after. "Stateless" means
``not model.is_stateful`` here and in ``check_teacher_state_layout`` (one predicate).

A scripted teacher (``teacher=None``, DAgger) has no model: the worker writes its actions into the chunks
(``teacher_action``, ``has_teacher``) and ``compute_labels`` gives ``lambda * mean(-log pi(a_teacher))`` over
the labeled ACT slots.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from colosseum.core.tree import Tree
from colosseum.networks.dist import Distribution
from colosseum.networks.model import PolicyModel
from colosseum.networks.state import State, tree_leaves

KLDirection = Literal["forward", "reverse"]


def check_teacher_state_layout(student: PolicyModel, teacher: PolicyModel) -> None:
    """ValueError unless ``teacher`` can be unrolled on ``student``'s chunks.

    A stateless teacher (``not teacher.is_stateful``, the predicate ``KickstartLoss`` uses) works with any
    student (it is unrolled with ``state0=None``). A stateful teacher reuses the chunks' initial states, which
    the student's behavior policy recorded, so it needs a stateful student with its exact state layout
    (exact while teacher == student, an approximation afterwards).
    """
    if not teacher.is_stateful:
        return
    teacher_shapes = [tuple(t.shape) for t in tree_leaves(teacher.initial_state(1))]
    student_shapes = ([tuple(t.shape) for t in tree_leaves(student.initial_state(1))] if student.is_stateful
                      else "none (stateless)")
    if student_shapes != teacher_shapes:
        raise ValueError(
            "a recurrent kickstart teacher must share the student's state layout (student state leaves "
            f"{student_shapes}, teacher {teacher_shapes}); use a stateless teacher or one with the student's core"
        )


class KickstartLoss:
    """Decaying KL penalty between a frozen teacher and the student policy; with ``teacher=None`` (a scripted
    teacher, DAgger) the decaying label loss ``compute_labels`` instead."""

    def __init__(
        self,
        teacher: PolicyModel | None,
        initial_lambda: float = 1.0,
        decay_steps: int = 50_000,
        direction: KLDirection = "forward",
    ) -> None:
        if direction not in ("forward", "reverse"):
            raise ValueError(f"kickstart direction must be 'forward' or 'reverse', got {direction!r}")
        self._teacher = teacher
        self._teacher_stateful = False
        if teacher is not None:
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            self._teacher_stateful = bool(teacher.is_stateful)
        self._initial_lambda = float(initial_lambda)
        self._decay_steps = max(1, int(decay_steps))
        self._direction: KLDirection = direction
        self._current_step = 0

    @property
    def teacher(self) -> PolicyModel | None:
        """The neural teacher; None for a scripted teacher (labels in the chunks)."""
        return self._teacher

    @property
    def direction(self) -> KLDirection:
        return self._direction

    @property
    def step_count(self) -> int:
        return self._current_step

    @property
    def current_lambda(self) -> float:
        """Current (decayed) lambda."""
        progress = min(1.0, self._current_step / self._decay_steps)
        return self._initial_lambda * (1.0 - progress)

    def step(self) -> None:
        """Advance the decay by one train step."""
        self._current_step += 1

    def to(self, device: str | torch.device) -> KickstartLoss:
        if self._teacher is not None:
            self._teacher.to(device)
        return self

    def state_dict(self) -> dict[str, int]:
        return {"step": self._current_step}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._current_step = int(state["step"])

    def compute(
        self,
        *,
        student_dist: Distribution,
        obs: Tree,
        reset_after: Tensor,
        state0: State,
        action_mask: Tree | None,
        actions: Tree,
        is_act: Tensor,
        reduction: Literal["mean_valid", "sum"],
    ) -> Tensor:
        """Scaled kickstart loss ``lambda * mean over ACT slots of the reduced per-decider KL``.

        Args:
            student_dist: the student's masked distribution over ``S*B`` time-major rows,
                from the algorithm's main ``unroll``.
            obs: observation tree, leaves ``[S, B, ...]``.
            reset_after: ``[S, B]`` bool; the state is reset after slot s.
            state0: the student's chunk initial state (leaves ``[B, ...]``); passed to a recurrent
                teacher only.
            action_mask: mask tree, leaves ``[S, B, ...]``, or None.
            actions: the recorded actions, leaves ``[S*B, ...]`` (they gate ``only_if`` children).
            is_act: ``[S, B]`` bool; only ACT slots contribute.
            reduction: ``"sum"`` or ``"mean_valid"`` over the valid deciders of a slot.
        """
        if self._teacher is None:
            raise RuntimeError("a scripted kickstart teacher has no model: the loss comes from chunk labels "
                               "(compute_labels)")
        lam = self.current_lambda
        if lam <= 0:
            return torch.zeros((), device=reset_after.device)
        with torch.no_grad():
            teacher_state0 = state0 if self._teacher_stateful else None   # a stateless teacher needs no state
            teacher_dist = self._teacher.unroll(obs, teacher_state0, reset_after.bool(), action_mask,
                                                with_value=False).dist
        if self._direction == "forward":
            kl = teacher_dist.unit_kl(student_dist, actions)
        else:
            kl = student_dist.unit_kl(teacher_dist, actions)
        kl = kl.float()                                             # [S*B, K]
        valid = student_dist.unit_valid(actions)
        per_slot = torch.where(valid, kl, torch.zeros_like(kl)).sum(-1)
        if reduction == "mean_valid":
            per_slot = per_slot / valid.sum(-1).clamp(min=1).to(per_slot.dtype)
        act = is_act.reshape(-1)
        mean = torch.where(act, per_slot, torch.zeros_like(per_slot)).sum() / act.sum().clamp(min=1)
        return lam * mean

    def compute_labels(
        self,
        *,
        student_dist: Distribution,
        teacher_actions: Tree,
        has_teacher: Tensor,
        is_act: Tensor,
        reduction: Literal["mean_valid", "sum"],
    ) -> Tensor:
        """DAgger loss ``lambda * mean(-log pi(a_teacher))`` over the ACT slots that carry a teacher label.

        Args:
            student_dist: the student's masked distribution over ``S*B`` time-major rows.
            teacher_actions: action tree, leaves ``[S*B, ...]``; unlabeled rows hold any legal action.
            has_teacher: ``[S, B]`` bool, True on labeled slots.
            is_act: ``[S, B]`` bool.
            reduction: ``"sum"`` = the joint log-prob (one decider); ``"mean_valid"`` = the mean of
                ``unit_log_prob`` over the valid deciders (``Units``, as in BC).
        Unlabeled slots add nothing and are not in the denominator; no labeled slot gives 0.
        """
        lam = self.current_lambda
        if lam <= 0:
            return torch.zeros((), device=is_act.device)
        labeled = (has_teacher.bool() & is_act).reshape(-1)
        if reduction == "sum":
            nll = -student_dist.log_prob(teacher_actions).float()
        elif reduction == "mean_valid":
            unit_lp = student_dist.unit_log_prob(teacher_actions).float()
            valid = student_dist.unit_valid(teacher_actions)
            total = torch.where(valid, unit_lp, torch.zeros_like(unit_lp)).sum(-1)
            nll = -total / valid.sum(-1).clamp(min=1).to(total.dtype)
        else:
            raise ValueError(f"reduction must be 'sum' or 'mean_valid', got {reduction!r}")
        mean = torch.where(labeled, nll, torch.zeros_like(nll)).sum() / labeled.sum().clamp(min=1)
        return lam * mean

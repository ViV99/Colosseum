"""Online behavioral cloning via kickstarting (Schmitt et al., 2018), on chunk v2.

Adds ``lambda * KL`` between a frozen teacher policy and the student to the RL loss.
Lambda decays linearly from ``initial_lambda`` to 0 over ``decay_steps`` train steps.

Direction (``training.kickstart_kl``):
- ``"forward"`` (default): KL(teacher || student), mode-covering (Kickstarting, AlphaStar, VPT);
- ``"reverse"``: KL(student || teacher), mode-seeking.

The teacher is unrolled with ``with_value=False`` over the same ``[S, B]`` slots as the
student, with the chunks' action masks, ``reset_after`` flags and initial states. The KL is
computed per decider (``Distribution.unit_kl``, 0 where a decider is invalid), reduced over
deciders per ``reduction`` (``"sum"`` or ``"mean_valid"``) and averaged over ACT slots only.

In SP2 the teacher stays global and is built from the student's networks config, so both
share one state layout and the chunk's ``initial_state`` (recorded by the student's behavior
policy) is the teacher's ``state0``: exact when teacher == student, an approximation after.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from colosseum.core.tree import Tree
from colosseum.networks.dist import Distribution
from colosseum.networks.model import PolicyModel
from colosseum.networks.state import State

KLDirection = Literal["forward", "reverse"]


class KickstartLoss:
    """Decaying KL penalty between a frozen teacher and the student policy."""

    def __init__(
        self,
        teacher: PolicyModel,
        initial_lambda: float = 1.0,
        decay_steps: int = 50_000,
        direction: KLDirection = "forward",
    ) -> None:
        if direction not in ("forward", "reverse"):
            raise ValueError(f"kickstart direction must be 'forward' or 'reverse', got {direction!r}")
        self._teacher = teacher
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)
        self._initial_lambda = float(initial_lambda)
        self._decay_steps = max(1, int(decay_steps))
        self._direction: KLDirection = direction
        self._current_step = 0

    @property
    def teacher(self) -> PolicyModel:
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
            state0: the teacher's initial state (leaves ``[B, ...]``).
            action_mask: mask tree, leaves ``[S, B, ...]``, or None.
            actions: the recorded actions, leaves ``[S*B, ...]`` (they gate ``only_if`` children).
            is_act: ``[S, B]`` bool; only ACT slots contribute.
            reduction: ``"sum"`` or ``"mean_valid"`` over the valid deciders of a slot.
        """
        lam = self.current_lambda
        if lam <= 0:
            return torch.zeros((), device=reset_after.device)
        with torch.no_grad():
            teacher_dist = self._teacher.unroll(obs, state0, reset_after.bool(), action_mask, with_value=False).dist
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

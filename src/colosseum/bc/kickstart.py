"""Online behavioral cloning via kickstarting (Schmitt et al., 2018).

Adds ``lambda * KL`` between a frozen teacher policy and the student to the RL
loss. Lambda decays linearly from ``initial_lambda`` to 0 over ``decay_steps``
train steps.

Direction (``training.kickstart_kl``):
- ``"forward"`` (default): KL(teacher || student), i.e. the teacher-to-student
  cross-entropy minus a constant, as in Kickstarting, AlphaStar and VPT. It is
  mode-covering, so the student keeps the teacher's diversity.
- ``"reverse"``: KL(student || teacher), mode-seeking.

The teacher is a ``PolicyModel`` unrolled over the same ``[T, B]`` sequences as
the student, with the chunks' action masks, episode resets (``dones``) and initial
states. The student distribution is passed in from the algorithm's main forward,
so there is no second student pass. Both distributions are masked, and the masked
KL is computed over legal actions only.

In SP1 the teacher is built from the student's config, so both share one state
layout and the chunk's ``initial_state``, recorded by the student's behavior
policy, is used as the teacher's ``state0``. That is exact when teacher == student
(the usual BC -> RL start) and an approximation afterwards. A separate teacher
config with its own state is SP3.
"""

from __future__ import annotations

from typing import Literal

import torch

from colosseum.networks.distributions import Distribution
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
        student_dist: Distribution,
        observations: torch.Tensor,
        dones: torch.Tensor,
        state0: State,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scaled kickstart loss ``lambda * mean(KL)``.

        Args:
            student_dist: the student's (masked) distribution over ``T*B`` time-major
                rows, from the algorithm's main ``unroll``.
            observations: ``[T, B, *obs_shape]``.
            dones: ``[T, B]`` bool; the state is reset after a done step.
            state0: initial state for the teacher's unroll (leaves ``[B, ...]``).
            action_mask: ``[T, B, mask_size]`` bool or None.
        """
        lam = self.current_lambda
        if lam <= 0:
            return torch.zeros((), device=observations.device)
        with torch.no_grad():
            teacher_dist = self._teacher.unroll(observations, state0, dones.bool(), action_mask).dist
        if self._direction == "forward":
            kl = teacher_dist.kl_divergence(student_dist)
        else:
            kl = student_dist.kl_divergence(teacher_dist)
        return lam * kl.mean()

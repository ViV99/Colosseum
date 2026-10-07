"""Online Behavioral Cloning via Kickstarting.

Adds a KL divergence penalty to the RL loss that encourages the student
policy to stay close to a frozen teacher (BC) policy. The penalty decays
over training to allow the student to surpass the teacher.

Loss component: ``lambda * KL(student || teacher)``

The lambda coefficient decays linearly from ``initial_lambda`` to 0 over
``decay_steps`` training steps.
"""

from __future__ import annotations

import torch

from colosseum.networks.actor_critic import ActorCriticNetwork


class KickstartLoss:
    """KL divergence loss between student and teacher policies.

    Usage::

        teacher = load_bc_model(...)
        kickstart = KickstartLoss(teacher, initial_lambda=1.0, decay_steps=50000)

        # In training loop:
        kl_loss = kickstart.compute(student_obs)
        total_loss = rl_loss + kl_loss  # kl_loss already scaled by lambda
        kickstart.step()  # decay lambda
    """

    def __init__(
        self,
        teacher_network: ActorCriticNetwork,
        initial_lambda: float = 1.0,
        decay_steps: int = 50000,
    ) -> None:
        """
        Args:
            teacher_network: Frozen BC model (will be set to eval, no grad).
            initial_lambda: Starting weight for the KL loss.
            decay_steps: Number of training steps over which lambda decays to 0.
        """
        self._teacher = teacher_network
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)

        self._initial_lambda = initial_lambda
        self._decay_steps = max(1, decay_steps)
        self._current_step = 0

    @property
    def current_lambda(self) -> float:
        """Current (decayed) lambda value."""
        progress = min(1.0, self._current_step / self._decay_steps)
        return self._initial_lambda * (1.0 - progress)

    def step(self) -> None:
        """Advance the decay by one training step."""
        self._current_step += 1

    def compute(
        self,
        student_network: ActorCriticNetwork,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scaled KL(student || teacher) loss.

        Args:
            student_network: The student policy being trained.
            observations: Batch of observations [B, *obs_shape].

        Returns:
            Scalar loss: lambda * mean(KL(student || teacher))
        """
        lam = self.current_lambda
        if lam <= 0:
            return torch.tensor(0.0, device=observations.device)

        # Get student distribution
        student_latent = student_network.encoder(observations)
        student_dist = student_network.policy(student_latent)

        # Get teacher distribution (no grad)
        with torch.no_grad():
            teacher_latent = self._teacher.encoder(observations)
            teacher_dist = self._teacher.policy(teacher_latent)

        # Compute KL divergence via public API
        kl = student_dist.kl_divergence(teacher_dist)

        return lam * kl.mean()

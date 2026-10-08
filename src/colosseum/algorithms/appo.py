"""APPO: Async PPO with V-trace off-policy correction.

Combines:
- V-trace targets for value function learning (handles policy lag from async workers)
- PPO clipped surrogate for policy gradient (additional stability)
- Entropy bonus for exploration

This is the primary algorithm used by Sample Factory, PufferLib, and similar
to OpenAI Five's approach.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from colosseum.algorithms.base import BaseAlgorithm
from colosseum.algorithms.vtrace import compute_vtrace
from colosseum.bc.kickstart import KickstartLoss
from colosseum.core.config import AlgorithmConfig, LRSchedule
from colosseum.core.types import TrajectoryChunk
from colosseum.networks.model import PolicyModel
from colosseum.networks.state import cat_batch, state_to


class APPO(BaseAlgorithm):
    """Async PPO with V-trace off-policy correction."""

    def __init__(
        self,
        model: PolicyModel,
        config: AlgorithmConfig,
        device: str | torch.device = "cpu",
        pin_memory: bool = False,
        kickstart: KickstartLoss | None = None,
    ):
        self._model = model.to(device)
        self._config = config
        self._device = device
        self._policy_version = 0
        self._kickstart = kickstart
        self._pin_memory = pin_memory

        # AMP (automatic mixed precision)
        self._use_amp = config.use_amp and str(device).startswith("cuda")
        self._amp_dtype = getattr(torch, config.amp_dtype, torch.float16)
        self._scaler = torch.amp.GradScaler("cuda") if self._use_amp else None

        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=config.learning_rate)
        self._zero_loss = torch.tensor(0.0, device=device)

        # Optionally compile V-trace for faster execution
        if config.use_torch_compile:
            self._compute_vtrace = torch.compile(compute_vtrace)
        else:
            self._compute_vtrace = compute_vtrace

        # The LR is a function of training progress (share of the global env-step
        # budget), set by the learner through set_progress() before every train step.
        self._progress = 0.0
        self.set_progress(0.0)

    @property
    def model(self) -> PolicyModel:
        return self._model

    @property
    def policy_version(self) -> int:
        return self._policy_version

    def set_progress(self, progress: float) -> None:
        """Set the share (0..1) of the global env-step budget consumed so far.

        The optimizer LR follows ``config.lr_schedule``: constant, linear decay to
        0 at progress 1, or cosine decay to 0 at progress 1. (Kickstart decay
        stays in train steps.)
        """
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

    @property
    def optimizer_state_dict(self) -> dict:
        return self._optimizer.state_dict()

    def _prepare_batch(self, chunks: list[TrajectoryChunk]) -> dict[str, torch.Tensor]:
        """Stack trajectory chunks into batched tensors.

        Each chunk has shape [T, ...]. We stack them to get [T, B, ...] where B = len(chunks).
        If pin_memory is enabled and device is CUDA, tensors are pinned for
        async DMA transfer.
        """
        device = self._device
        use_pinning = self._pin_memory and str(device).startswith("cuda")

        batch_cpu = {
            "observations": torch.stack([c.observations for c in chunks], dim=1),
            "actions": torch.stack([c.actions for c in chunks], dim=1),
            "behavior_log_probs": torch.stack([c.action_log_probs for c in chunks], dim=1),
            "rewards": torch.stack([c.rewards for c in chunks], dim=1),
            "dones": torch.stack([c.dones for c in chunks], dim=1),
            "old_values": torch.stack([c.values for c in chunks], dim=1),
            "bootstrap_values": torch.stack([c.bootstrap_value for c in chunks]),
        }
        if chunks[0].action_masks is not None:
            batch_cpu["action_masks"] = torch.stack(
                [c.action_masks for c in chunks], dim=1,
            )

        if use_pinning:
            batch = {
                k: v.pin_memory().to(device, non_blocking=True)
                for k, v in batch_cpu.items()
            }
        else:
            batch = {k: v.to(device) for k, v in batch_cpu.items()}
        # Model state before each chunk's first transition: leaves [1, ...] -> [B, ...].
        batch["initial_state"] = state_to(cat_batch([c.initial_state for c in chunks]), device)
        return batch

    def _evaluate(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unroll the model over the batch: (log_probs, values, entropy), each ``[T, B]``."""
        T, B = batch["rewards"].shape
        out = self._model.unroll(
            batch["observations"],
            batch["initial_state"],
            batch["dones"].bool(),
            batch.get("action_masks"),
        )
        actions = batch["actions"].reshape(T * B, *batch["actions"].shape[2:])
        log_probs = out.dist.log_prob(actions).reshape(T, B)
        values = out.value.reshape(T, B)
        entropy = out.dist.entropy().reshape(T, B)
        return log_probs, values, entropy

    @torch.no_grad()
    def evaluate_chunks(self, chunks: list[TrajectoryChunk]) -> tuple[torch.Tensor, torch.Tensor]:
        """Current model's log-probs of the recorded actions and its values.

        Returns two ``[T*B]`` tensors in time-major order (index ``t*B + b`` is
        step ``t`` of ``chunks[b]``), computed exactly as in training:
        ``model.unroll`` from ``cat_batch(chunk.initial_state ...)``.
        """
        log_probs, values, _ = self._evaluate(self._prepare_batch(chunks))
        return log_probs.reshape(-1), values.reshape(-1)

    def compute_loss(self, chunks: list[TrajectoryChunk]) -> dict[str, torch.Tensor]:
        """Compute APPO loss from a batch of trajectory chunks.

        Steps:
        1. Stack chunks into [T, B, ...] tensors
        2. Forward pass: ``model.unroll`` from the chunks' initial states gives
           current log_probs, values, entropy (one path for every model)
        3. Compute V-trace targets + advantages
        4. PPO clipped surrogate with V-trace advantages
        5. Value loss: MSE(values, vtrace_targets)
        6. Entropy bonus
        """
        cfg = self._config
        batch = self._prepare_batch(chunks)

        T, B = batch["rewards"].shape
        obs_shape = batch["observations"].shape[2:]
        # Flattened observations [T*B, *obs_shape] for the kickstart KL term.
        flat_obs = batch["observations"].reshape(T * B, *obs_shape)

        amp_ctx = torch.autocast(
            device_type="cuda", dtype=self._amp_dtype, enabled=self._use_amp,
        )
        with amp_ctx:
            target_log_probs, new_values, entropy = self._evaluate(batch)

        # V-trace targets and advantages
        with torch.no_grad():
            vtrace_targets, vtrace_advantages = self._compute_vtrace(
                behavior_log_probs=batch["behavior_log_probs"],
                target_log_probs=target_log_probs.detach(),
                rewards=batch["rewards"],
                values=new_values.detach(),
                bootstrap_value=batch["bootstrap_values"],
                dones=batch["dones"],
                gamma=cfg.gamma,
                rho_bar=cfg.vtrace_rho_bar,
                c_bar=cfg.vtrace_c_bar,
                lam=cfg.vtrace_lambda,
            )

        # PPO clipped surrogate loss
        log_ratio = torch.clamp(
            target_log_probs - batch["behavior_log_probs"], -20.0, 20.0
        )
        ratio = torch.exp(log_ratio)
        adv = vtrace_advantages.detach()
        if cfg.normalize_advantages and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(new_values, vtrace_targets.detach())

        # Entropy loss (negative because we want to maximize entropy)
        entropy_loss = -entropy.mean()

        # Total loss
        total_loss = policy_loss + cfg.value_loss_coeff * value_loss + cfg.entropy_coeff * entropy_loss

        # Kickstart loss (KL to teacher policy)
        kickstart_loss = self._zero_loss
        if self._kickstart is not None and self._kickstart.current_lambda > 0:
            kickstart_loss = self._kickstart.compute(self._model, flat_obs)
            total_loss = total_loss + kickstart_loss

        # Metrics for logging
        with torch.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_fraction = ((ratio - 1.0).abs() > cfg.eps_clip).float().mean()

        result = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": -entropy_loss,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }
        if self._kickstart is not None:
            result["kickstart_loss"] = kickstart_loss.detach()
            result["kickstart_lambda"] = torch.tensor(self._kickstart.current_lambda)
        return result

    def train_step(self, chunks: list[TrajectoryChunk]) -> dict[str, float]:
        """Full training step with minibatch iterations.

        Performs num_epochs passes over the data, splitting into minibatches.
        """
        cfg = self._config
        metrics_accum: dict[str, float] = {}
        num_updates = 0

        # Refresh observation-normalization statistics once per train step, from
        # this step's fresh samples only (never once per epoch/minibatch forward).
        obs_all = torch.cat([torch.as_tensor(c.observations) for c in chunks], dim=0)
        self._model.update_normalizers(obs_all.to(self._device))

        for _epoch in range(cfg.num_epochs):
            # For APPO, we typically do a single pass (num_epochs=1)
            # because the data is already off-policy. Multiple epochs
            # further increase the off-policyness.
            # However, we support multiple epochs for flexibility.

            # Shuffle chunks and create minibatches (minibatching is over the
            # batch dimension B = number of chunks; each chunk's [T] sequence
            # stays intact so recurrent training is unaffected).
            indices = torch.randperm(len(chunks))
            mb_size = cfg.minibatch_chunks if cfg.minibatch_chunks > 0 else len(chunks)

            for start in range(0, len(chunks), mb_size):
                end = min(start + mb_size, len(chunks))
                mb_indices = indices[start:end]
                mb_chunks = [chunks[i] for i in mb_indices]

                if not mb_chunks:
                    continue

                losses = self.compute_loss(mb_chunks)
                total_loss = losses["total_loss"]

                self._optimizer.zero_grad()

                if self._scaler is not None:
                    self._scaler.scale(total_loss).backward()
                    if cfg.max_grad_norm > 0:
                        self._scaler.unscale_(self._optimizer)
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), cfg.max_grad_norm)
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    total_loss.backward()
                    if cfg.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self._model.parameters(), cfg.max_grad_norm)
                    self._optimizer.step()

                # Accumulate metrics
                for key, value in losses.items():
                    if key not in metrics_accum:
                        metrics_accum[key] = 0.0
                    metrics_accum[key] += value.item()
                num_updates += 1

        if self._kickstart is not None:
            self._kickstart.step()

        self._policy_version += 1

        # Average metrics
        metrics = {k: v / max(1, num_updates) for k, v in metrics_accum.items()}
        metrics["policy_version"] = float(self._policy_version)
        metrics["learning_rate"] = self._optimizer.param_groups[0]["lr"]
        return metrics

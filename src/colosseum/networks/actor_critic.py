from __future__ import annotations

import torch
import torch.nn as nn

from colosseum.networks.base import BaseEncoder, BasePolicy, BaseValue


class ActorCriticNetwork(nn.Module):
    """Combines encoder + optional recurrent trunk + policy + value.

    When ``recurrent`` is provided, the network processes sequences:
    encoder output is fed through the recurrent module before being
    passed to the policy and value heads. Hidden state is carried
    across timesteps within an episode and reset at episode boundaries.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        policy: BasePolicy,
        value: BaseValue,
        recurrent: nn.Module | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.policy = policy
        self.value = value
        self.recurrent = recurrent

    @property
    def is_recurrent(self) -> bool:
        return self.recurrent is not None

    @property
    def recurrent_hidden_size(self) -> int:
        """Size of the recurrent hidden state (0 if not recurrent)."""
        if self.recurrent is None:
            return 0
        return self.recurrent.hidden_size

    @property
    def recurrent_num_layers(self) -> int:
        if self.recurrent is None:
            return 0
        return self.recurrent.num_layers

    def initial_hidden(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized hidden state for LSTM: (h, c).

        Each tensor has shape [num_layers, batch_size, hidden_size].
        For GRU, c is a zero tensor of the same shape (unused but keeps API uniform).
        """
        h = torch.zeros(
            self.recurrent_num_layers, batch_size, self.recurrent_hidden_size,
        )
        c = torch.zeros_like(h)
        return h, c

    def forward(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple:
        """Forward pass.

        Args:
            obs: Observations [B, *obs_shape].
            hidden: Optional (h, c) recurrent state.

        Returns:
            (distribution, values, new_hidden)
            new_hidden is None if not recurrent.
        """
        latent = self.encoder(obs)

        new_hidden = None
        if self.recurrent is not None and hidden is not None:
            # latent: [B, latent_dim] → [1, B, latent_dim] for RNN input
            latent_seq = latent.unsqueeze(0)
            if isinstance(self.recurrent, nn.LSTM):
                output, (h_new, c_new) = self.recurrent(latent_seq, (hidden[0], hidden[1]))
                new_hidden = (h_new, c_new)
            else:
                # GRU
                output, h_new = self.recurrent(latent_seq, hidden[0])
                new_hidden = (h_new, torch.zeros_like(h_new))
            latent = output.squeeze(0)  # [1, B, hidden_size] → [B, hidden_size]

        dist = self.policy(latent)
        val = self.value(latent)
        return dist, val, new_hidden

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple | None]:
        """Inference: sample (or, if ``deterministic``, take the mode) an action.

        Args:
            obs: Observations tensor [B, *obs_shape].
            action_mask: Optional bool tensor of valid actions [B, num_actions].
            hidden: Optional (h, c) recurrent state.
            deterministic: If True, use the distribution mode (argmax / mean)
                instead of sampling. Useful for evaluation.

        Returns:
            (actions, log_probs, values, new_hidden)
            new_hidden is None if not recurrent.
        """
        dist, values, new_hidden = self.forward(obs, hidden)
        if action_mask is not None:
            dist = dist.apply_mask(action_mask)
        actions = dist.mode() if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values, new_hidden

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training: compute log_prob, value, entropy for given (obs, action) pairs.

        Stateless path — processes all observations independently (no recurrence).
        Used by APPO for the non-recurrent training path.

        Returns: (action_log_probs, values, entropy)
        """
        dist, values, _ = self.forward(obs)
        if action_mask is not None:
            dist = dist.apply_mask(action_mask)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy

    def evaluate_actions_recurrent(
        self,
        obs_seq: torch.Tensor,
        actions_seq: torch.Tensor,
        hidden_init: tuple[torch.Tensor, torch.Tensor],
        action_mask_seq: torch.Tensor | None = None,
        dones_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training: process a sequence through encoder + RNN + heads.

        Args:
            obs_seq: Observations [T, B, *obs_shape].
            actions_seq: Actions [T, B, *act_shape].
            hidden_init: Initial (h, c) state [num_layers, B, hidden_size].
            action_mask_seq: Optional masks [T, B, num_actions].
            dones_seq: Optional episode-termination flags [T, B]. When provided,
                the recurrent hidden state is RESET to zero after every step
                where ``done == 1`` so that memory does not leak across episode
                boundaries that fall within a chunk — mirroring the hidden-state
                resets the worker performs during rollout. This is required for
                correct BPTT when episodes are shorter than ``chunk_length``
                (the common case for short-horizon games). When ``dones_seq`` is
                None, the whole sequence is processed in a single pass (no
                resets).

        Returns:
            (log_probs, values, entropy) each of shape [T, B].
        """
        T, B = obs_seq.shape[:2]

        # Encode all timesteps: [T*B, *obs_shape] → [T*B, latent_dim]
        obs_flat = obs_seq.reshape(T * B, *obs_seq.shape[2:])
        latent_flat = self.encoder(obs_flat)
        latent_seq = latent_flat.reshape(T, B, -1)  # [T, B, latent_dim]

        is_lstm = isinstance(self.recurrent, nn.LSTM)

        if dones_seq is None:
            # Fast path: no episode resets, single cuDNN-accelerated pass.
            if is_lstm:
                rnn_out, _ = self.recurrent(latent_seq, (hidden_init[0], hidden_init[1]))
            else:
                rnn_out, _ = self.recurrent(latent_seq, hidden_init[0])
        else:
            # Reset-aware unroll: step the RNN one timestep at a time and zero
            # the hidden state at episode boundaries (done[t] == 1 → fresh state
            # going into t+1). The loop is over T (sequential) but each step is
            # batched over B.
            outputs = []
            h = hidden_init[0]
            c = hidden_init[1]
            for t in range(T):
                step_in = latent_seq[t : t + 1]  # [1, B, latent_dim]
                if is_lstm:
                    out_t, (h, c) = self.recurrent(step_in, (h, c))
                else:
                    out_t, h = self.recurrent(step_in, h)
                outputs.append(out_t)
                # not_done broadcasts over [num_layers, B, hidden_size]
                not_done = (1.0 - dones_seq[t].to(h.dtype)).view(1, B, 1)
                h = h * not_done
                if is_lstm:
                    c = c * not_done
            rnn_out = torch.cat(outputs, dim=0)
        # rnn_out: [T, B, hidden_size]

        # Policy + value for each timestep
        rnn_flat = rnn_out.reshape(T * B, -1)
        dist = self.policy(rnn_flat)
        values = self.value(rnn_flat)  # [T*B]

        actions_flat = actions_seq.reshape(T * B, *actions_seq.shape[2:])
        if action_mask_seq is not None:
            mask_flat = action_mask_seq.reshape(T * B, -1)
            dist = dist.apply_mask(mask_flat)

        log_probs = dist.log_prob(actions_flat)  # [T*B]
        entropy = dist.entropy()  # [T*B]

        return (
            log_probs.reshape(T, B),
            values.reshape(T, B),
            entropy.reshape(T, B),
        )

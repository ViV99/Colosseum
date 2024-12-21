from typing import Optional

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(
        self,
        state_encoder: nn.Module,
        critic_head: nn.Module,
        discrete_actor_heads: Optional[nn.ModuleList] = None,
        continuous_actor_heads: Optional[nn.ModuleList] = None,
        continuous_action_std_init: Optional[list[float]] = None,
    ):
        super().__init__()
        discrete_actor_heads = discrete_actor_heads or nn.ModuleList([])
        continuous_actor_heads = continuous_actor_heads or nn.ModuleList([])
        continuous_action_std_init = continuous_action_std_init or []

        if len(discrete_actor_heads) == 0 and len(continuous_actor_heads) == 0:
            raise ValueError("At least one head must be specified")
        if len(continuous_actor_heads) != 0 and (continuous_action_std_init is None
                                                 or len(continuous_action_std_init) != len(continuous_actor_heads)):
            raise ValueError("Initial action std should be provided for each continuous actor head")

        self.state_encoder = state_encoder
        self.critic_head = critic_head
        self.discrete_n_heads = len(discrete_actor_heads)
        self.discrete_actor_heads = discrete_actor_heads
        self.continuous_n_heads = len(continuous_actor_heads)
        self.continuous_actor_heads = continuous_actor_heads
        self.action_vars = nn.ParameterList(
            torch.tensor(std ** 2, dtype=torch.float32) for std in continuous_action_std_init
        ) if self.continuous_n_heads else None

    def set_continuous_action_std(self, new_action_std: list[float]):
        if self.continuous_n_heads:
            self.action_vars = nn.ParameterList(
                torch.tensor(std ** 2, dtype=torch.float32, device=self.action_vars[0].device) for std in new_action_std
            )
        else:
            raise NotImplementedError("Trying to set action std for ActorCritic without continuous actor heads")

    def act(
        self, state_seq: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        """
        Estimate state values and choose actions for those states.

        :param state_seq: Batch of sequences of states ([batch_size, seq_len, <state_dims>])
        :param padding_mask: Mask for padded states (True if padded, False otherwise) ([batch_size, seq_len])
        :return: Tuple of
                 chosen actions for each head (n_heads * ([batch_size, action_dim] or [batch_size,])),
                 action logprobs for each head (n_heads * [batch_size,]),
                 state values ([batch_size,])
        """
        state_emb = self.state_encoder(state_seq, src_key_padding_mask=padding_mask)[:, -1, ...]

        actions, action_logprobs = [], []
        for i in range(self.continuous_n_heads):
            action_mean = self.continuous_actor_heads[i](state_emb)
            cov_mat = torch.diag_embed(self.action_vars[i].expand_as(action_mean))
            dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            actions.append(action.detach())
            action_logprobs.append(action_logprob.detach())

        for i in range(self.discrete_n_heads):
            action_probs = self.discrete_actor_heads[i](state_emb)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            actions.append(action.detach())
            action_logprobs.append(action_logprob.detach())

        state_values = self.critic_head(state_emb)

        return actions, action_logprobs, state_values.detach()

    def evaluate(
        self, state_seq: torch.Tensor, actions: list[torch.Tensor], padding_mask: Optional[torch.Tensor] = None
    ) -> tuple[list[torch.Tensor], torch.Tensor, list[torch.Tensor]]:
        """
        Estimate action probabilities, state values and distribution entropies.

        :param state_seq: Batch of sequences of states ([batch_size, seq_len, <state_dims>])
        :param actions: Actions for each head (n_heads * ([batch_size, action_dim] or [batch_size,]))
        :param padding_mask: Mask for padded states (True if padded, False otherwise) ([batch_size, seq_len])
        :return: Tuple of action logprobs for each head (n_heads * [batch_size,]), state values ([batch_size,]),
                 entropies of distributions for each head (n_heads * [batch_size,])
        """
        state_emb = self.state_encoder(state_seq, src_key_padding_mask=padding_mask)[:, -1, ...]

        action_logprobs, entropies = [], []
        for i in range(self.continuous_n_heads):
            action_mean = self.continuous_actor_heads[i](state_emb)
            cov_mat = torch.diag_embed(self.action_vars[i].expand_as(action_mean))
            dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
            action_logprob = dist.log_prob(actions[i])
            entropy = dist.entropy()
            action_logprobs.append(action_logprob)
            entropies.append(entropy)

            # # For Single Action Environments.
            # if self.action_dim == 1:
            #     action = action.reshape(-1, self.action_dim)

        for i in range(self.discrete_n_heads):
            action_probs = self.discrete_actor_heads[i](state_emb)
            dist = torch.distributions.Categorical(action_probs)
            action_logprob = dist.log_prob(actions[i + self.continuous_n_heads])
            entropy = dist.entropy()
            action_logprobs.append(action_logprob)
            entropies.append(entropy)

        state_values = self.critic_head(state_emb)

        return action_logprobs, state_values, entropies

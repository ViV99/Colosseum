import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .actor_critic import ActorCritic


class PPO:
    def __init__(
        self,
        policy: ActorCritic,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        max_seq_len: int = 1000,
        num_epochs: int = 40,
        device: str = "cpu"
    ):
        self.device = device
        self.n_heads = policy.discrete_n_heads + policy.continuous_n_heads

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.max_seq_len = max_seq_len
        self.num_epochs = num_epochs

        self.policy = policy
        self.policy_old = copy.deepcopy(policy)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.to(device)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.state_encoder.parameters(), 'lr': lr_actor},
            {'params': self.policy.discrete_actor_heads.parameters(), 'lr': lr_actor},
            {'params': self.policy.continuous_actor_heads.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic_head.parameters(), 'lr': lr_critic}
        ])

    def to(self, device: str):
        self.device = device
        self.policy.to(device)
        self.policy_old.to(device)

    def eval(self):
        self.policy.eval()

    def train(self):
        self.policy.train()

    @staticmethod
    def pad_sequence(
        sequence: torch.Tensor, out_len: int, pad_side: str = "right"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param sequence: [seq_len, <state_dims>]
        :param out_len: desired length
        :param pad_side: "right" or "left"
        :return: (padded_sequence, padding_mask) == ([out_len, <state_dims>], [out_len,])
        """
        mask = torch.zeros(out_len, device=sequence[0].device, dtype=bool)
        if pad_side == "right":
            if len(sequence) >= out_len:
                return sequence[:out_len], mask

            mask[len(sequence):] = True
            return torch.row_stack(
                [sequence, sequence[-1].repeat([out_len - len(sequence)] + [1] * len(sequence[-1].shape))]
            ), mask

        if pad_side == "left":
            if len(sequence) >= out_len:
                return sequence[-out_len:], mask

            mask[:-len(sequence)] = True
            return torch.row_stack(
                [sequence[0].repeat([out_len - len(sequence)] + [1] * len(sequence[0].shape)), sequence]
            ), mask

        raise ValueError(f"pad_side '{pad_side}' is not supported")

    @staticmethod
    def pad_sequences(
        sequences: list[torch.Tensor], pad_side: str = "right"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param sequences: batch_size * [seq_len, <state_dims>]
        :param pad_side: "right" or "left"
        :return: (padded_batch, padding_mask) == ([batch_size, max_seq_len, <state_dims>], [batch_size, max_seq_len])
        """
        out_len = max(seq.shape[0] for seq in sequences)
        padded_sequences, masks = [], []
        for seq in sequences:
            padded_seq, mask = PPO.pad_sequence(seq, out_len, pad_side)
            padded_sequences.append(padded_seq)
            masks.append(mask)
        return torch.stack(padded_sequences), torch.stack(masks)

    def select_action(self, states: list[torch.Tensor]):
        """
        Choose an action based on the game state.

        :param states: Sequence of states (N * [<state_dims>,])
        :return: Tuple of
                 chosen actions for each head (n_heads * ([1, action_dim] or [1,])),
                 action logprobs for each head (n_heads * [1,]),
                 state values ([1,])
        """
        with torch.no_grad():
            actions, action_logprobs, state_values = self.policy.act(
                torch.stack(states[-self.max_seq_len:]).unsqueeze(0).to(self.device)
            )
            return actions, action_logprobs, state_values

    def _create_batch(self, states: torch.Tensor, is_terminals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param states: [N, <state_dims>]
        :param is_terminals: [N,]
        :return: []
        """
        sequences = []
        traj_start_ind = 0
        for i in range(len(states)):
            sequences.append(states[max(traj_start_ind, i - self.max_seq_len + 1):i + 1])
            if is_terminals[i] == 1:
                traj_start_ind = i + 1
        states, padding_masks = PPO.pad_sequences(sequences, pad_side="left")
        return states.detach().to(self.device), padding_masks.detach().to(self.device)

    def update(
        self,
        states: torch.Tensor,
        actions: list[torch.Tensor],
        state_rewards: torch.Tensor,
        is_terminals: torch.Tensor,
        old_state_values: torch.Tensor,
        old_logprobs: list[torch.Tensor]
    ):
        """
        :param states: [N, <state_dims>]
        :param actions: n_heads * ([N, action_dim] or [N,])
        :param state_rewards: [N,]
        :param is_terminals: [N,]
        :param old_state_values: [N,]
        :param old_logprobs: n_heads * [N,]
        """
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(state_rewards[::-1], is_terminals[::-1]):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.append(discounted_reward)

        rewards = torch.tensor(list(reversed(rewards)), dtype=torch.float32, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        states, padding_masks = self._create_batch(states, is_terminals)

        old_state_values = old_state_values.detach().to(self.device)
        for i in range(self.n_heads):
            actions[i] = actions[i].detach().to(self.device)
            old_logprobs[i] = old_logprobs[i].detach().to(self.device)

        advantages = rewards.detach() - old_state_values

        for _ in range(self.num_epochs):
            logprobs, state_values, dist_entropies = self.policy.evaluate(states, actions, padding_masks)
            state_values = state_values.squeeze()

            loss = 0
            for i in range(self.n_heads):
                ratios = torch.exp(logprobs[i] - old_logprobs[i].detach())

                # Finding Surrogate Loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                loss += -torch.min(surr1, surr2) + 0.5 * F.mse_loss(state_values, rewards) - 0.01 * dist_entropies[i]

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def state_dict(self) -> dict[str, ...]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "policy": self.policy.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, ...]):
        self.policy.load_state_dict(state_dict["policy"])
        self.policy_old = copy.deepcopy(self.policy)
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def share_memory(self):
        self.policy.share_memory()

"""Cores: the (optionally stateful) trunk between the encoder and the heads.

``ComposedModel`` runs ``encoder -> core -> heads``. A core turns a latent
``[B, input_dim]`` plus its state into features ``[B, output_dim]`` and the
next state. State leaves are batch-first (see :mod:`colosseum.networks.state`).

In SP1 every core's ``unroll`` is the reference per-step loop; fast sequence
paths (cuDNN RNN, full-sequence attention masks) come later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from colosseum.networks.state import State, batch_size_of, tree_leaves, where_done


class Core(nn.Module, ABC):
    input_dim: int
    output_dim: int

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State:
        return None

    @abstractmethod
    def step(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        """``x`` [B, input_dim] -> (features [B, output_dim], next state)."""

    def unroll(self, x: Tensor, state0: State, dones: Tensor) -> Tensor:
        """``x`` [T, B, input_dim] -> [T, B, output_dim]; state reset after step t where ``dones[t]``."""
        state = state0
        outputs = []
        for t in range(x.shape[0]):
            y, state = self.step(x[t], state)
            outputs.append(y)
            state = self.reset_state(state, dones[t])
        return torch.stack(outputs, dim=0)

    def reset_state(self, state: State, done: Tensor) -> State:
        """Rows where ``done`` ([B]) is True are replaced by initial-state rows.

        A state without tensor leaves (``None``, empty containers) is returned unchanged.
        """
        batch = batch_size_of(state)
        if batch is None:
            return state
        device = tree_leaves(state)[0].device
        return where_done(done, self.initial_state(batch, device), state)


class NoCore(Core):
    """Identity core for stateless models."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

    def step(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        return x, None

    def unroll(self, x: Tensor, state0: State, dones: Tensor) -> Tensor:
        return x


class LSTMCore(Core):
    """LSTM trunk. State ``{"h": [B, L, H], "c": [B, L, H]}``."""

    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_dim, hidden_size, num_layers)

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State:
        zeros = torch.zeros(batch_size, self.num_layers, self.hidden_size, device=device)
        return {"h": zeros, "c": zeros.clone()}

    def step(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        h = state["h"].transpose(0, 1).contiguous()
        c = state["c"].transpose(0, 1).contiguous()
        y, (h, c) = self.rnn(x.unsqueeze(0), (h, c))
        return y.squeeze(0), {"h": h.transpose(0, 1).contiguous(), "c": c.transpose(0, 1).contiguous()}


class GRUCore(Core):
    """GRU trunk. State ``{"h": [B, L, H]}``."""

    def __init__(self, input_dim: int, hidden_size: int = 128, num_layers: int = 1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_dim, hidden_size, num_layers)

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State:
        return {"h": torch.zeros(batch_size, self.num_layers, self.hidden_size, device=device)}

    def step(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        h = state["h"].transpose(0, 1).contiguous()
        y, h = self.rnn(x.unsqueeze(0), h)
        return y.squeeze(0), {"h": h.transpose(0, 1).contiguous()}


class _AttentionBlock(nn.Module):
    """Pre-LayerNorm transformer block (no gating)."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(), nn.Linear(4 * d_model, d_model))

    def forward(self, h: Tensor, blocked: Tensor) -> Tensor:
        a = self.ln1(h)
        h = h + self.attn(a, a, a, attn_mask=blocked, need_weights=False)[0]
        return h + self.ff(self.ln2(h))


class WindowAttentionCore(Core):
    """Causal attention of the current latent over the last ``window`` latents of the episode.

    State ``{"mem": [B, window, d_model] float, "len": [B] long}``: ``mem`` holds
    the projected latents of the previous steps, right-aligned (the newest is
    ``mem[:, -1]``); only the last ``len`` slots are valid. Each step re-runs
    the ``num_layers`` blocks over ``[valid memory; current]`` with a causal
    mask and returns the output at the current position (GTrXL-lite without
    gating). Resetting the state sets ``len`` to 0 and zeroes ``mem``.
    """

    def __init__(self, input_dim: int, d_model: int = 64, window: int = 16,
                 num_heads: int = 4, num_layers: int = 1) -> None:
        super().__init__()
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.input_dim = input_dim
        self.output_dim = d_model
        self.d_model = d_model
        self.window = window
        self.num_heads = num_heads
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(window + 1, d_model) * 0.02)
        self.blocks = nn.ModuleList(_AttentionBlock(d_model, num_heads) for _ in range(num_layers))
        self.out_norm = nn.LayerNorm(d_model)

    def initial_state(self, batch_size: int, device: str | torch.device = "cpu") -> State:
        return {
            "mem": torch.zeros(batch_size, self.window, self.d_model, device=device),
            "len": torch.zeros(batch_size, dtype=torch.long, device=device),
        }

    def step(self, x: Tensor, state: State) -> tuple[Tensor, State]:
        z = self.in_proj(x)  # [B, D]
        mem, length = state["mem"], state["len"]
        W = self.window
        seq = torch.cat([mem.to(z.dtype), z.unsqueeze(1)], dim=1) + self.pos  # [B, W+1, D]

        pos = torch.arange(W + 1, device=z.device)
        valid = pos.unsqueeze(0) >= (W - length).unsqueeze(1)  # [B, W+1]; current slot always valid
        causal = pos.unsqueeze(0) <= pos.unsqueeze(1)  # [W+1 (query), W+1 (key)]
        eye = torch.eye(W + 1, dtype=torch.bool, device=z.device)
        allowed = causal.unsqueeze(0) & (valid.unsqueeze(1) | eye.unsqueeze(0))  # every row keeps itself
        blocked = (~allowed).repeat_interleave(self.num_heads, dim=0)  # [B*heads, W+1, W+1]

        h = seq
        for block in self.blocks:
            h = block(h, blocked)
        out = self.out_norm(h[:, -1])

        new_mem = torch.cat([mem[:, 1:], z.unsqueeze(1).to(mem.dtype)], dim=1)
        new_len = torch.clamp(length + 1, max=W)
        return out, {"mem": new_mem, "len": new_len}

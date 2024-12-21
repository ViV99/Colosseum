from typing import Optional

import torch
import torch.nn as nn


class Seq3dEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.dim_change = nn.Sequential(
            nn.Conv3d(input_dim, hidden_dim * 2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv3d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=4*hidden_dim,
            dim_feedforward=4*hidden_dim,
            nhead=1,
            dropout=0.,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, data: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None):
        """
        :param data: [batch_size, seq_len, in_channels, x, y, z]
        :param src_key_padding_mask: [batch_size, seq_len]
        """
        batch_size, seq_len, in_channels, x, y, z = data.shape
        data = self.dim_change(data.view(batch_size * seq_len, in_channels, x, y, z)).view(batch_size, seq_len, -1)
        return self.encoder(data, src_key_padding_mask=src_key_padding_mask)

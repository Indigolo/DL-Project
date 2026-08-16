"""PatchTST forecasting model used by the submission template."""

from __future__ import annotations

import torch
from torch import nn


class _PatchEmbedding(nn.Module):
    """Splits a per-channel sequence into overlapping patches and embeds them."""

    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        # Replicate the last `stride` steps so the final patch always reaches
        # the end of the look-back window, matching the official PatchTST padding.
        self.pad = nn.ReplicationPad1d((0, stride))
        self.projection = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def num_patches(self, seq_len: int) -> int:
        return (seq_len + self.stride - self.patch_len) // self.stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch * n_vars, seq_len) -> (batch * n_vars, num_patches, d_model)."""
        x = self.pad(x.unsqueeze(1)).squeeze(1)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return self.dropout(self.projection(patches))


class ForecastModel(nn.Module):
    """Channel-independent PatchTST encoder for one-step-ahead forecasting.

    Each input channel (the normalized target plus every numeric covariate) is
    patched and encoded independently by a shared transformer encoder, then the
    per-channel representations are concatenated with a series embedding and
    mapped to the next normalized target value.
    """

    def __init__(
        self,
        input_size: int,
        context_length: int = 168,
        patch_len: int = 24,
        stride: int = 12,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        series_count: int = 1,
        series_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.input_size = input_size

        self.patch_embedding = _PatchEmbedding(patch_len, stride, d_model, dropout)
        num_patches = self.patch_embedding.num_patches(context_length)
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.series_embedding = nn.Embedding(series_count, series_embedding_dim)
        flattened_size = input_size * num_patches * d_model
        self.head = nn.Sequential(
            nn.Linear(flattened_size + series_embedding_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, series_idx: torch.Tensor) -> torch.Tensor:
        """Predict the next normalized target from a context window.

        x: (batch, context_length, input_size)
        """
        batch, seq_len, n_vars = x.shape
        channels = x.transpose(1, 2).reshape(batch * n_vars, seq_len)
        embedded = self.patch_embedding(channels) + self.positional_embedding
        encoded = self.encoder(embedded)
        per_channel = encoded.reshape(batch, n_vars, -1)
        flattened = per_channel.reshape(batch, -1)

        series_embedded = self.series_embedding(series_idx)
        combined = torch.cat([flattened, series_embedded], dim=-1)
        return self.head(combined).squeeze(-1)

"""LSTM forecasting model used by the submission template."""

from __future__ import annotations

import torch


class ForecastModel(torch.nn.Module):
    """Simple autoregressive LSTM baseline with optional series embeddings."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        series_count: int = 1,
        series_embedding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.series_embedding = torch.nn.Embedding(series_count, series_embedding_dim)
        self.lstm = torch.nn.LSTM(
            input_size=input_size + series_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, series_idx: torch.Tensor) -> torch.Tensor:
        """Predict the next normalized target from a context window."""
        series_embedding = self.series_embedding(series_idx)
        repeated_embedding = series_embedding.unsqueeze(1).expand(-1, x.size(1), -1)
        lstm_input = torch.cat([x, repeated_embedding], dim=-1)
        outputs, _ = self.lstm(lstm_input)
        return self.head(outputs[:, -1]).squeeze(-1)

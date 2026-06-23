"""LSTM forecasting model used by the submission template."""

from __future__ import annotations

import torch
from transformers import PatchTSTConfig, PatchTSTForPrediction

class ForecastModel(torch.nn.Module):
    """Simple autoregressive LSTM baseline with optional series embeddings."""

    def __init__(
        self,
        input_size: int,
        context_length: int = 96,
        prediction_length: int = 1,
        patch_length: int = 16,
        patch_stride: int = 8,
        d_model: int = 128,
        num_hidden_layers: int = 3,
        num_attention_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        config = PatchTSTConfig(
            num_input_channels=input_size,
            context_length=context_length,
            prediction_length=prediction_length,
            patch_length=patch_length,
            patch_stride=patch_stride,
            d_model=d_model,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            ffn_dim=ffn_dim,
            attention_dropout=dropout,
            positional_dropout=dropout,
            ff_dropout=dropout,
            head_dropout=dropout,
            loss="mse",
        )
        self.model = PatchTSTForPrediction(config)

    def forward(self, x: torch.Tensor, future_values: torch.Tensor | None = None, past_obs_mask: torch.Tensor | None = None,) -> torch.Tensor:
        """Predict the next normalized target from a context window."""
        outputs = self.model(
            past_values=x,
            past_observed_mask=past_obs_mask,
            future_values=future_values,
        )

        if future_values is not None:
            return outputs.loss

        preds = outputs.prediction_outputs
        return preds

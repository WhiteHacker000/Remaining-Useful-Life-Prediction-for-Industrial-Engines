"""Configurable LSTM regressor for fixed-length engine sensor windows."""

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """Regress capped RUL from a batch of normalized sensor sequences."""

    def __init__(
        self,
        input_size=14,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be positive")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.recurrent = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(hidden_size // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_size // 2, 1), 1),
        )

    def forward(self, sequences):
        if sequences.ndim != 3:
            raise ValueError("sequences must have shape (batch, time, features)")
        if sequences.shape[1] < 1:
            raise ValueError("sequences must contain at least one timestep")
        if sequences.shape[2] != self.input_size:
            raise ValueError(
                f"expected {self.input_size} input features, got {sequences.shape[2]}"
            )
        recurrent_output, _ = self.recurrent(sequences)
        return self.regression_head(recurrent_output[:, -1, :]).squeeze(-1)

    def get_config(self):
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout_rate,
        }

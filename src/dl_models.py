"""Sequence models for RUL prediction (Phase 2): LSTM, temporal CNN and Transformer encoder.

Every model takes a batch of sensor windows x with shape (batch, time, channels)
and, optionally, engineered features f with shape (batch, n_features), and returns
the predicted RUL divided by RUL_CAP (so targets are roughly in [0, 1]).

Each sequence model is split into an *encoder* (window -> embedding vector) and a
small regression *head*, so the same encoder can be reused inside the hybrid
(src/hybrid.py).
"""

import math

import torch
from torch import nn


def head(in_dim, dropout):
    return nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))


# ----------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------

class LSTMEncoder(nn.Module):
    """Stacked LSTM; the hidden state after the last time step summarises the window."""

    def __init__(self, n_in, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out_dim = hidden

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1]


class CausalConvBlock(nn.Module):
    """Two dilated causal 1-D convolutions with a residual connection (Bai et al., 2018)."""

    def __init__(self, c_in, c_out, kernel, dilation, dropout):
        super().__init__()
        self.pad = (kernel - 1) * dilation          # left padding only -> no look-ahead
        self.conv1 = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):                            # x: (batch, channels, time)
        h = self.drop(self.act(self.conv1(nn.functional.pad(x, (self.pad, 0)))))
        h = self.drop(self.act(self.conv2(nn.functional.pad(h, (self.pad, 0)))))
        return self.act(h + self.skip(x))


class TCNEncoder(nn.Module):
    """Temporal convolutional network: dilations 1, 2, 4, ... so the receptive field
    grows exponentially with depth. Levels are added until it covers the window."""

    def __init__(self, n_in, window, channels=64, kernel=3, dropout=0.2):
        super().__init__()
        blocks, c_in, dilation, field = [], n_in, 1, 1
        while field < window:
            blocks.append(CausalConvBlock(c_in, channels, kernel, dilation, dropout))
            field += 2 * (kernel - 1) * dilation
            c_in, dilation = channels, dilation * 2
        self.net = nn.Sequential(*blocks)
        self.receptive_field = field
        self.out_dim = channels

    def forward(self, x):
        return self.net(x.transpose(1, 2))[:, :, -1]


class TransformerEncoder(nn.Module):
    """Self-attention over the window. Padded steps are masked out of the attention;
    the output at the last (current) cycle is used as the summary."""

    def __init__(self, n_in, window, d_model=64, heads=4, layers=2, ff=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_in, d_model)
        pos = torch.arange(window).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(window, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out_dim = d_model

    def forward(self, x):
        pad = x[:, :, -1] < 0.5                      # mask channel: 0 = padding
        h = self.encoder(self.proj(x) + self.pe, src_key_padding_mask=pad)
        return self.norm(h[:, -1])


# ----------------------------------------------------------------------------
# Regressors
# ----------------------------------------------------------------------------

class SequenceRegressor(nn.Module):
    """Encoder + head. Ignores engineered features: learns from the sensor sequence only."""

    def __init__(self, encoder, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.head = head(encoder.out_dim, dropout)

    def forward(self, x, f=None):
        return self.head(self.encoder(x)).squeeze(-1)


def make_encoder(kind, n_in, window, dropout=0.2):
    if kind == "lstm":
        return LSTMEncoder(n_in, dropout=dropout)
    if kind == "tcn":
        return TCNEncoder(n_in, window, dropout=dropout)
    if kind == "transformer":
        return TransformerEncoder(n_in, window, dropout=min(dropout, 0.1))
    raise ValueError(kind)


def make_dl_model(name, n_in, window, n_feat=None, dropout=0.2):
    """name: 'lstm' | 'tcn' | 'transformer' | 'features' | 'hybrid-<encoder>'"""
    from .hybrid import FeatureRegressor, HybridRegressor      # hybrid builds on these encoders
    if name == "features":
        return FeatureRegressor(n_feat, dropout)
    if name.startswith("hybrid-"):
        return HybridRegressor(make_encoder(name.split("-", 1)[1], n_in, window, dropout), n_feat, dropout)
    return SequenceRegressor(make_encoder(name, n_in, window, dropout), dropout)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

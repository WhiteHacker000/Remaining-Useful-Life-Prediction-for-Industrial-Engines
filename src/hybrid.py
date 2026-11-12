"""Hybrid feature-fusion model (Phase 2).

Combines the two views of an engine that Phases 1 and 2 compare:

- a sequence encoder (LSTM, TCN or Transformer from src/dl_models.py) that reads the raw
  sensor window and can learn trajectory shapes, and
- a small MLP over the 88 engineered degradation features of Phase 1, which carry
  information a fixed window cannot see (e.g. distance from the engine's own healthy
  baseline, measured on its first cycles).

The two embeddings are concatenated and mapped to RUL by a shared head, and the whole
network is trained end to end with one loss.
"""

import torch
from torch import nn

from .dl_models import head


class FeatureMLP(nn.Module):
    """Small network on the 88 Phase 1 features at the current cycle."""

    def __init__(self, n_feat, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, 64), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(64, 32), nn.ReLU())
        self.out_dim = 32

    def forward(self, f):
        return self.net(f)


class FeatureRegressor(nn.Module):
    """Engineered features only (no sequence): the 'feature branch' alone."""

    def __init__(self, n_feat, dropout=0.2):
        super().__init__()
        self.features = FeatureMLP(n_feat, dropout)
        self.head = head(self.features.out_dim, dropout)

    def forward(self, x, f):
        return self.head(self.features(f)).squeeze(-1)


class HybridRegressor(nn.Module):
    """Feature fusion: sequence encoder (raw history) + MLP (engineered degradation
    features), concatenated and passed to a shared head. Trained end to end."""

    def __init__(self, encoder, n_feat, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.features = FeatureMLP(n_feat, dropout)
        self.fuse = nn.Sequential(nn.Linear(encoder.out_dim + self.features.out_dim, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x, f):
        return self.fuse(torch.cat([self.encoder(x), self.features(f)], dim=1)).squeeze(-1)

"""Toric encoder with explicit expected net long/short payoff outputs."""

import torch
from torch import nn

from .trading_model_v3 import ToricEncoder


class ToricPayoffModel(ToricEncoder):
    def __init__(self, num_features, max_len=32, dim_angles=32, num_states=16, dropout=0.1, num_outputs=2):
        if not 0 <= dropout < 1:
            raise ValueError("invalid dropout")
        if num_outputs not in (1, 2):
            raise ValueError("payoff model supports long-only or long/short outputs")
        super().__init__(num_features=num_features, max_len=max_len, dim_angles=dim_angles,
                         num_states=num_states, num_layers=1)
        self.config = dict(num_features=num_features, max_len=max_len, dim_angles=dim_angles,
                           num_states=num_states, dropout=dropout, num_outputs=num_outputs)
        self.num_outputs = num_outputs
        self.payoff_head = nn.Sequential(nn.Linear(2 * dim_angles, dim_angles), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(dim_angles, num_outputs))
        self.register_buffer("feature_mean", torch.zeros(num_features))
        self.register_buffer("feature_std", torch.ones(num_features))
        self.register_buffer("target_mean", torch.zeros(num_outputs))
        self.register_buffer("target_std", torch.ones(num_outputs))
        self.register_buffer("fitted_statistics", torch.tensor(False))

    def fit_statistics(self, train_features, train_targets):
        features = torch.as_tensor(train_features)
        targets = torch.as_tensor(train_targets)
        if (features.ndim != 2 or features.shape[1] != self.num_features or
                targets.ndim != 2 or targets.shape[1] != self.num_outputs or not len(features) or not len(targets) or
                not torch.isfinite(features).all() or not torch.isfinite(targets).all()):
            raise ValueError("finite training features and aligned payoff columns required")
        self.feature_mean.copy_(features.mean(0))
        self.feature_std.copy_(features.std(0, unbiased=False).clamp_min(1e-5))
        self.target_mean.copy_(targets.mean(0))
        self.target_std.copy_(targets.std(0, unbiased=False).clamp_min(1e-5))
        self.fitted_statistics.fill_(True)

    def forward(self, features):
        if not self.fitted_statistics.item():
            raise RuntimeError("fit train-only statistics before using the payoff model")
        return self.payoff_head(self.encode((features - self.feature_mean) / self.feature_std))

    @torch.inference_mode()
    def predict_payoffs(self, features):
        if self.training:
            raise RuntimeError("call eval before payoff inference")
        return self(features) * self.target_std + self.target_mean

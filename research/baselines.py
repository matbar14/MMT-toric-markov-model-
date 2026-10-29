"""Generic sequence-model baselines and naive predictors.

All forecasters map a window ``x`` of shape (B, T, F) to a forecast of shape
(B, H, C), trained with plain MSE on standardized data — the exact same
contract as ``research.toric_core.ToricForecaster``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RecurrentForecaster(nn.Module):
    """GRU or LSTM encoder with a last-hidden-state linear head."""

    def __init__(
        self,
        num_features: int,
        horizon: int,
        num_channels: int = 1,
        kind: str = "gru",
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_diff_input: bool = False,
    ) -> None:
        super().__init__()
        if kind not in ("gru", "lstm"):
            raise ValueError("kind must be 'gru' or 'lstm'")
        rnn_cls = nn.GRU if kind == "gru" else nn.LSTM
        self.use_diff_input = use_diff_input
        self.horizon = horizon
        self.num_channels = num_channels
        input_size = num_features * 2 if use_diff_input else num_features
        self.rnn = rnn_cls(
            input_size,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, horizon * num_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.use_diff_input:
            features = torch.cat(
                [features, features.diff(dim=1, prepend=features[:, :1, :])], dim=-1
            )
        out, _ = self.rnn(features)
        return self.head(out[:, -1]).view(
            features.shape[0], self.horizon, self.num_channels
        )


class TransformerForecaster(nn.Module):
    """Small non-causal Transformer encoder; the last token feeds the head."""

    def __init__(
        self,
        num_features: int,
        horizon: int,
        num_channels: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 512,
        use_diff_input: bool = False,
    ) -> None:
        super().__init__()
        self.use_diff_input = use_diff_input
        self.horizon = horizon
        self.num_channels = num_channels
        input_size = num_features * 2 if use_diff_input else num_features
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos = nn.Parameter(torch.randn(max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head = nn.Linear(d_model, horizon * num_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        seq_len = features.shape[1]
        if self.use_diff_input:
            features = torch.cat(
                [features, features.diff(dim=1, prepend=features[:, :1, :])], dim=-1
            )
        hidden = self.input_proj(features) + self.pos[:seq_len].unsqueeze(0)
        hidden = self.encoder(hidden)
        return self.head(hidden[:, -1]).view(
            features.shape[0], self.horizon, self.num_channels
        )


class MLPForecaster(nn.Module):
    """Flattened-window MLP: the plainest generic nonlinear learner.

    Control model for the question "is the failure specific to recurrent /
    phase machinery, or does ANY nonlinear model fit the data?".  A
    two-hidden-layer perceptron over the flattened window has full expressive
    freedom over the (T x C) -> (H x C) map and no temporal weight sharing, so
    when it fails on small data the bottleneck is gradient difficulty, not the
    architecture family.
    """

    def __init__(
        self,
        seq_len: int,
        horizon: int,
        num_channels: int = 1,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.num_channels = num_channels
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * num_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * num_channels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).view(
            features.shape[0], self.horizon, self.num_channels
        )


class MoELinearForecaster(nn.Module):
    """Mixture of linear experts with a small nonlinear gate.

    The minimal model with an explicit soft regime mechanism: each expert is a
    (channel-shared) linear map from the flattened window to the forecast, and
    a two-layer MLP gate computes regime posteriors from the same window.  If
    the right inductive bias for regime-conditional data is "identify the
    regime, then apply a linear predictor", this model should capture it
    without any phase machinery — it is the reference the toric core has to
    beat to justify its complexity.  Note the gate is deliberately nonlinear:
    regime evidence such as signal envelope is a quadratic function of the
    window, which a purely linear gate could not represent.
    """

    def __init__(
        self,
        seq_len: int,
        horizon: int,
        num_channels: int = 1,
        num_experts: int = 3,
        gate_hidden: int = 64,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        self.horizon = horizon
        self.num_channels = num_channels
        self.num_experts = num_experts
        flat_dim = seq_len * num_channels
        self.gate = nn.Sequential(
            nn.Linear(flat_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, num_experts),
        )
        self.experts = nn.ModuleList(
            [nn.Linear(flat_dim, horizon * num_channels) for _ in range(num_experts)]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        flat = features.reshape(features.shape[0], -1)
        weights = torch.softmax(self.gate(flat), dim=-1)  # (B, K)
        expert_out = torch.stack(
            [expert(flat) for expert in self.experts], dim=1
        )  # (B, K, H*C)
        forecast = (weights.unsqueeze(-1) * expert_out).sum(dim=1)
        return forecast.view(
            features.shape[0], self.horizon, self.num_channels
        )


def moving_avg(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Edge-padded moving average along the time axis; x is (B, T, C)."""
    front = (kernel_size - 1) // 2
    back = kernel_size - 1 - front
    padded = torch.cat(
        [x[:, :1].expand(-1, front, -1), x, x[:, -1:].expand(-1, back, -1)], dim=1
    )
    return F.avg_pool1d(
        padded.transpose(1, 2), kernel_size=kernel_size, stride=1
    ).transpose(1, 2)


class DLinearForecaster(nn.Module):
    """Channel-independent DLinear (Zeng et al., 2023).

    A strong, embarrassingly simple baseline for seasonal forecasting: split
    the window into trend (moving average) and seasonal residual, and apply a
    shared linear map from T inputs to H outputs per component.
    """

    def __init__(
        self,
        seq_len: int,
        horizon: int,
        num_channels: int = 1,
        kernel_size: int = 25,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.horizon = horizon
        self.num_channels = num_channels
        self.trend_head = nn.Linear(seq_len, horizon)
        self.seasonal_head = nn.Linear(seq_len, horizon)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        trend = moving_avg(features, self.kernel_size)
        seasonal = features - trend
        forecast = self.trend_head(trend.transpose(1, 2)) + self.seasonal_head(
            seasonal.transpose(1, 2)
        )
        return forecast.transpose(1, 2)


def persistence_forecast(x: torch.Tensor, horizon: int) -> torch.Tensor:
    """Repeat the last observed value across the horizon."""
    return x[:, -1:, :].expand(-1, horizon, -1).clone()


def drift_forecast(x: torch.Tensor, horizon: int) -> torch.Tensor:
    """Extrapolate the last per-channel difference linearly."""
    last = x[:, -1:, :]
    step = (x[:, -1:, :] - x[:, -2:-1, :]).clamp(-5.0, 5.0)
    offsets = torch.arange(1, horizon + 1, device=x.device, dtype=x.dtype)
    return last + offsets.view(1, -1, 1) * step


def seasonal_naive_forecast(x: torch.Tensor, horizon: int, period: int) -> torch.Tensor:
    """Tile the last ``period`` observations over the horizon."""
    tail = x[:, -period:, :]
    repeats = math.ceil(horizon / period)
    return tail.repeat(1, repeats, 1)[:, :horizon, :].clone()

"""Positional embeddings for toric models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .quantizer import angles_to_complex, quantize_angle


class QuantizedPositionalShifts(nn.Module):
    """Positional shifts represented by quantized angles."""

    def __init__(self, max_len: int, embedding_dim: int, n_bits: int = 8) -> None:
        super().__init__()
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.n_bits = n_bits
        self.raw_shifts = nn.Parameter(torch.randn(max_len, embedding_dim) * math.pi)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        shifts = self.raw_shifts[positions]
        shifts = torch.tanh(shifts) * math.pi
        shifts_q = quantize_angle(shifts, self.n_bits)
        return angles_to_complex(shifts_q)

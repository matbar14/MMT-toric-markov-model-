"""Utilities for angle quantization and complex normalization."""

from __future__ import annotations

import math

import torch


class AngleQuantizer(torch.autograd.Function):
    """Straight-through quantizer for angles in [-pi, pi]."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
        levels = 2**n_bits
        scale = (2 * math.pi) / levels

        x_norm = (x + math.pi) / (2 * math.pi)
        idx = torch.floor(x_norm * levels).long().clamp(0, levels - 1)
        x_quant = idx.float() * scale - math.pi

        ctx.save_for_backward(x)
        return x_quant

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Straight-through estimator.
        return grad_output, None


def quantize_angle(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    return AngleQuantizer.apply(x, n_bits)


def angles_to_complex(angles: torch.Tensor) -> torch.Tensor:
    """Convert angles to complex points on unit circles."""
    return torch.complex(torch.cos(angles), torch.sin(angles))


def normalize_complex(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize complex tensor elementwise to unit modulus."""
    return z / (z.abs() + eps)


def complex_to_features(z: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor (..., d) to real features (..., 2d)."""
    return torch.cat([z.real, z.imag], dim=-1)

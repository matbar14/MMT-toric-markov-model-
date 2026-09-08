"""Fractal toric recurrent cell."""

from __future__ import annotations

import torch
import torch.nn as nn

from .quantizer import normalize_complex


class FractalToricCell(nn.Module):
    """Multi-level toric recurrence with optional lightweight attention."""

    def __init__(
        self,
        dim_angles: int,
        num_levels: int,
        num_states: int,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        if dim_angles % num_levels != 0:
            raise ValueError("dim_angles must be divisible by num_levels")

        self.dim_angles = dim_angles
        self.num_levels = num_levels
        self.dim_per_level = dim_angles // num_levels
        self.num_states = num_states
        self.use_attention = use_attention

        # FIX: Add bidirectional rotations (forward and backward)
        self.state_rotation_fwd = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(num_states, self.dim_per_level, dtype=torch.cfloat) * 0.1
                )
                for _ in range(num_levels)
            ]
        )
        
        self.state_rotation_bwd = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(num_states, self.dim_per_level, dtype=torch.cfloat) * 0.1
                )
                for _ in range(num_levels)
            ]
        )
        
        if use_attention:
            self.context = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(self.dim_per_level, dtype=torch.cfloat))
                    for _ in range(num_levels)
                ]
            )
            self.attn_scale = nn.Parameter(torch.tensor(1.0))

        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        h_levels: list[torch.Tensor],
        token_levels: list[torch.Tensor],
        pos_levels: list[torch.Tensor],
        state_probs: torch.Tensor,
    ) -> list[torch.Tensor]:
        updated_levels: list[torch.Tensor] = []

        for level_idx in range(self.num_levels):
            # FIX: Bidirectional rotations
            rotation_fwd = torch.einsum(
                "bs,sd->bd",
                state_probs.to(torch.cfloat),
                self.state_rotation_fwd[level_idx],
            )
            rotation_bwd = torch.einsum(
                "bs,sd->bd",
                state_probs.to(torch.cfloat),
                self.state_rotation_bwd[level_idx],
            )
            
            # Apply both forward and backward rotations
            phase_fwd = token_levels[level_idx] * pos_levels[level_idx] * rotation_fwd
            phase_bwd = token_levels[level_idx] * pos_levels[level_idx] * rotation_bwd.conj()
            
            h_fwd = normalize_complex(h_levels[level_idx] * phase_fwd)
            h_bwd = normalize_complex(h_levels[level_idx] * phase_bwd)
            h_new = normalize_complex((h_fwd + h_bwd) / 2.0)

            if self.use_attention:
                score_complex = (h_new * self.context[level_idx].conj()).sum(dim=-1)
                score = score_complex.abs()
                weight = torch.sigmoid(score * self.attn_scale).unsqueeze(-1)
                h_new = normalize_complex(h_new + weight * self.context[level_idx])
            updated_levels.append(h_new)

        g = torch.sigmoid(self.gate)
        mixed_levels = [
            normalize_complex(g * new + (1 - g) * old)
            for new, old in zip(updated_levels, h_levels)
        ]
        return mixed_levels

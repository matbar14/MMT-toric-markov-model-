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
        
        # DYNAMIC magnitude control - learnable parameters
        # Model learns optimal clamp bounds for each level
        self.mag_min = nn.ParameterList(
            [nn.Parameter(torch.tensor(0.3)) for _ in range(num_levels)]
        )
        self.mag_max = nn.ParameterList(
            [nn.Parameter(torch.tensor(1.7)) for _ in range(num_levels)]
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
            
            # Preserve magnitude information before normalization
            h_fwd_raw = h_levels[level_idx] * phase_fwd
            h_bwd_raw = h_levels[level_idx] * phase_bwd
            
            # Store magnitudes (information about signal strength)
            mag_fwd = h_fwd_raw.abs()
            mag_bwd = h_bwd_raw.abs()
            
            # Normalize for direction
            h_fwd = normalize_complex(h_fwd_raw)
            h_bwd = normalize_complex(h_bwd_raw)
            
            # Combine both directions
            h_combined = (h_fwd + h_bwd) / 2.0
            
            # Average magnitude
            avg_mag = (mag_fwd + mag_bwd) / 2.0
            
            # DYNAMIC clamping - model learns optimal bounds for each level
            # Use sigmoid to ensure mag_min < mag_max and both are positive
            min_bound = torch.sigmoid(self.mag_min[level_idx]) * 0.5  # Range: 0 to 0.5
            max_bound = 1.0 + torch.sigmoid(self.mag_max[level_idx]) * 1.5  # Range: 1.0 to 2.5
            
            # Re-apply magnitude with learned dynamic bounds
            h_new = normalize_complex(h_combined) * avg_mag.clamp(min=min_bound, max=max_bound)

            if self.use_attention:
                # Use BOTH real and imaginary parts for attention score
                score_complex = (h_new * self.context[level_idx].conj()).sum(dim=-1)
                # Use magnitude for score
                score = torch.sqrt(score_complex.real**2 + score_complex.imag**2)
                weight = torch.sigmoid(score * self.attn_scale).unsqueeze(-1)
                h_new = h_new * (1 + weight)
                # Normalize after attention to prevent further explosion
                h_new = normalize_complex(h_new)
            updated_levels.append(h_new)

        g = torch.sigmoid(self.gate)
        mixed_levels = [
            normalize_complex(g * new + (1 - g) * old)
            for new, old in zip(updated_levels, h_levels)
        ]
        return mixed_levels

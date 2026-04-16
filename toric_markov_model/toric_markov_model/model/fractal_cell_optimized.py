"""Optimized fractal toric cell with parallel processing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import normalize_complex


class OptimizedFractalToricCell(nn.Module):
    """Multi-level toric recurrence with parallel scan optimization."""

    def __init__(
        self,
        dim_angles: int,
        num_levels: int,
        num_states: int,
        use_attention: bool = True,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        if dim_angles % num_levels != 0:
            raise ValueError("dim_angles must be divisible by num_levels")

        self.dim_angles = dim_angles
        self.num_levels = num_levels
        self.dim_per_level = dim_angles // num_levels
        self.num_states = num_states
        self.use_attention = use_attention
        self.use_flash = use_flash

        self.state_rotation = nn.ParameterList(
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
        """Single step forward (for sequential processing)."""
        updated_levels: list[torch.Tensor] = []

        for level_idx in range(self.num_levels):
            rotation = torch.einsum(
                "bs,sd->bd",
                state_probs.to(torch.cfloat),
                self.state_rotation[level_idx],
            )
            phase = token_levels[level_idx] * pos_levels[level_idx] * rotation

            h_new = normalize_complex(h_levels[level_idx] * phase)

            if self.use_attention:
                score = (h_new * self.context[level_idx].conj()).sum(dim=-1).real
                weight = torch.sigmoid(score * self.attn_scale).unsqueeze(-1)
                h_new = normalize_complex(h_new * (1 + weight))

            updated_levels.append(h_new)

        g = torch.sigmoid(self.gate)
        mixed_levels = [
            normalize_complex(g * new + (1 - g) * old)
            for new, old in zip(updated_levels, h_levels)
        ]
        return mixed_levels

    def forward_parallel(
        self,
        h_levels_init: list[torch.Tensor],
        token_levels_seq: list[torch.Tensor],
        pos_levels_seq: list[torch.Tensor],
        state_probs_seq: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Parallel processing of a sequence chunk.
        
        Args:
            h_levels_init: Initial hidden states [batch, dim_per_level]
            token_levels_seq: Token embeddings [batch, seq_len, dim_per_level] per level
            pos_levels_seq: Position embeddings [batch, seq_len, dim_per_level] per level
            state_probs_seq: State probabilities [batch, seq_len, num_states]
        
        Returns:
            Final h_levels after processing the sequence
        """
        batch_size, seq_len = token_levels_seq[0].shape[:2]
        device = token_levels_seq[0].device
        
        # Process each level independently (can be parallelized)
        final_levels = []
        
        for level_idx in range(self.num_levels):
            # Compute rotations for all timesteps at once
            # [batch, seq_len, num_states] @ [num_states, dim] -> [batch, seq_len, dim]
            rotations = torch.einsum(
                "bts,sd->btd",
                state_probs_seq.to(torch.cfloat),
                self.state_rotation[level_idx],
            )
            
            # Compute phases for all timesteps
            phases = token_levels_seq[level_idx] * pos_levels_seq[level_idx] * rotations
            
            # Apply recurrence with parallel scan approximation
            # For now, use cumulative product (can be optimized with associative scan)
            h = h_levels_init[level_idx].unsqueeze(1)  # [batch, 1, dim]
            
            # Iterative application (TODO: replace with parallel scan)
            h_seq = [h.squeeze(1)]
            for t in range(seq_len):
                h_new = normalize_complex(h_seq[-1] * phases[:, t, :])
                h_seq.append(h_new)
            
            h_final = h_seq[-1]
            
            if self.use_attention:
                score = (h_final * self.context[level_idx].conj()).sum(dim=-1).real
                weight = torch.sigmoid(score * self.attn_scale).unsqueeze(-1)
                h_final = normalize_complex(h_final * (1 + weight))
            
            final_levels.append(h_final)
        
        # Apply gating
        g = torch.sigmoid(self.gate)
        mixed_levels = [
            normalize_complex(g * new + (1 - g) * old)
            for new, old in zip(final_levels, h_levels_init)
        ]
        return mixed_levels

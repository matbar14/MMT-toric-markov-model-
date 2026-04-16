"""KV-cache for efficient inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ToricCache:
    """Cache for toric model inference."""
    
    h_levels: list[torch.Tensor] | None = None
    state_probs: torch.Tensor | None = None
    seq_len: int = 0
    
    def update(
        self,
        h_levels: list[torch.Tensor],
        state_probs: torch.Tensor,
    ) -> None:
        """Update cache with new hidden states."""
        self.h_levels = h_levels
        self.state_probs = state_probs
        self.seq_len += 1
    
    def reset(self) -> None:
        """Reset cache."""
        self.h_levels = None
        self.state_probs = None
        self.seq_len = 0
    
    def is_empty(self) -> bool:
        """Check if cache is empty."""
        return self.h_levels is None

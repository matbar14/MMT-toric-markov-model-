"""Discrete Markov chain modules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteMarkovChain(nn.Module):
    """Differentiable Markov chain with Gumbel-Softmax transitions."""

    def __init__(self, hidden_dim: int, num_states: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_states = num_states
        self.temperature = temperature

        self.token_proj = nn.Linear(hidden_dim, num_states, bias=False)
        self.state_proj = nn.Parameter(torch.randn(num_states, num_states) * 0.01)
        self.state_bias = nn.Parameter(torch.zeros(num_states))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        token_features: torch.Tensor,
        prev_state_probs: torch.Tensor,
        hard: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        token_features = self.norm(token_features)

        logits_token = self.token_proj(token_features)
        logits_prev = prev_state_probs @ self.state_proj
        logits = logits_token + logits_prev + self.state_bias

        if hard:
            state_idx = torch.argmax(logits, dim=-1)
            state_probs = F.one_hot(state_idx, self.num_states).float()
            return state_probs, state_idx

        state_probs = F.gumbel_softmax(
            logits,
            tau=self.temperature,
            hard=False,
            dim=-1,
        )
        return state_probs, None

    def set_temperature(self, temperature: float) -> None:
        self.temperature = temperature

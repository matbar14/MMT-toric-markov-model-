"""Main Toric-Markov model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .embeddings import QuantizedAngleEmbedding, QuantizedPositionalShifts
from .fractal_cell import FractalToricCell
from .markov import DiscreteMarkovChain
from .quantizer import angles_to_complex, complex_to_features, quantize_angle


class ToricMarkovModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim_angles: int,
        max_len: int,
        num_states: int = 256,
        num_levels: int = 4,
        num_layers: int = 2,
        n_bits: int = 8,
        use_attention: bool = True,
        use_continuous: bool = True,
        chunk_size: int = 0,
    ) -> None:
        super().__init__()
        if dim_angles % num_levels != 0:
            raise ValueError("dim_angles must be divisible by num_levels")

        self.vocab_size = vocab_size
        self.dim_angles = dim_angles
        self.max_len = max_len
        self.num_states = num_states
        self.num_levels = num_levels
        self.num_layers = num_layers
        self.n_bits = n_bits
        self.use_attention = use_attention
        self.use_continuous = use_continuous
        self.chunk_size = chunk_size if chunk_size > 0 else max_len

        self.token_emb = QuantizedAngleEmbedding(vocab_size, dim_angles, n_bits)
        self.pos_emb = QuantizedPositionalShifts(max_len, dim_angles, n_bits)

        self.markov_chain = DiscreteMarkovChain(hidden_dim=2 * dim_angles, num_states=num_states)
        self.toric_layers = nn.ModuleList(
            [
                FractalToricCell(
                    dim_angles=dim_angles,
                    num_levels=num_levels,
                    num_states=num_states,
                    use_attention=use_attention,
                )
                for _ in range(num_layers)
            ]
        )

        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.state_to_logits = nn.Linear(num_states, vocab_size, bias=False)

    def _all_token_complex(self) -> torch.Tensor:
        all_angles = torch.tanh(self.token_emb.raw_angles) * math.pi
        all_angles_q = quantize_angle(all_angles, self.n_bits)
        return angles_to_complex(all_angles_q)

    def _process_chunk_parallel(
        self,
        token_phase_chunk: torch.Tensor,
        token_levels_chunk: list[torch.Tensor],
        pos_levels_chunk: list[torch.Tensor],
        h_levels: list[torch.Tensor],
        state_probs: torch.Tensor,
        all_token_complex: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Process a chunk of tokens in parallel."""
        batch_size, chunk_len, _ = token_phase_chunk.shape
        device = token_phase_chunk.device
        
        chunk_logits = []
        
        # Process each position in chunk sequentially (but chunk is smaller)
        for t in range(chunk_len):
            token_t = token_phase_chunk[:, t, :]
            token_feat_t = complex_to_features(token_t)
            
            if self.training:
                state_probs, _ = self.markov_chain(token_feat_t, state_probs, hard=False)
            else:
                state_probs, _ = self.markov_chain(token_feat_t, state_probs, hard=True)
            
            if self.use_continuous:
                token_levels_t = [lvl[:, t, :] for lvl in token_levels_chunk]
                pos_levels_t = [lvl[:, t, :] for lvl in pos_levels_chunk]
                for layer in self.toric_layers:
                    h_levels = layer(h_levels, token_levels_t, pos_levels_t, state_probs)
                
                h_concat = torch.cat(h_levels, dim=-1)
                logits_t = (
                    torch.einsum("bd,vd->bv", h_concat.real, all_token_complex.real)
                    + torch.einsum("bd,vd->bv", h_concat.imag, all_token_complex.imag)
                )
                logits_t = logits_t * self.logit_scale
            else:
                logits_t = self.state_to_logits(state_probs)
            
            chunk_logits.append(logits_t.unsqueeze(0))
        
        logits = torch.cat(chunk_logits, dim=0)
        return logits, h_levels, state_probs

    def forward_chunked(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with chunked processing for better parallelization."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        token_phase = self.token_emb(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_phase = self.pos_emb(positions)
        
        token_levels = torch.chunk(token_phase, self.num_levels, dim=-1)
        pos_levels = torch.chunk(pos_phase, self.num_levels, dim=-1)
        
        level_dim = self.dim_angles // self.num_levels
        h_levels = [torch.ones(batch_size, level_dim, dtype=torch.cfloat, device=device) for _ in range(self.num_levels)]
        
        state_probs = torch.zeros(batch_size, self.num_states, device=device)
        state_probs[:, 0] = 1.0
        
        all_token_complex = self._all_token_complex()
        all_logits = []
        
        # Process in chunks
        num_chunks = (seq_len - 1 + self.chunk_size - 1) // self.chunk_size
        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, seq_len - 1)
            
            token_phase_chunk = token_phase[:, start:end, :]
            token_levels_chunk = [lvl[:, start:end, :] for lvl in token_levels]
            pos_levels_chunk = [lvl[:, start:end, :] for lvl in pos_levels]
            
            chunk_logits, h_levels, state_probs = self._process_chunk_parallel(
                token_phase_chunk,
                token_levels_chunk,
                pos_levels_chunk,
                h_levels,
                state_probs,
                all_token_complex,
            )
            all_logits.append(chunk_logits)
        
        logits = torch.cat(all_logits, dim=0)
        targets = input_ids[:, 1:].transpose(0, 1)
        return logits, targets

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.chunk_size > 0 and self.chunk_size < input_ids.shape[1]:
            return self.forward_chunked(input_ids)
        return self.forward_with_per_step_logits(input_ids)

    def forward_with_per_step_logits(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        token_phase = self.token_emb(input_ids)

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_phase = self.pos_emb(positions)

        token_levels = torch.chunk(token_phase, self.num_levels, dim=-1)
        pos_levels = torch.chunk(pos_phase, self.num_levels, dim=-1)

        level_dim = self.dim_angles // self.num_levels
        h_levels = [torch.ones(batch_size, level_dim, dtype=torch.cfloat, device=device) for _ in range(self.num_levels)]

        state_probs = torch.zeros(batch_size, self.num_states, device=device)
        state_probs[:, 0] = 1.0

        all_logits = []
        all_token_complex = self._all_token_complex()

        for t in range(seq_len - 1):
            token_t = token_phase[:, t, :]
            token_feat_t = complex_to_features(token_t)

            if self.training:
                state_probs, _ = self.markov_chain(token_feat_t, state_probs, hard=False)
            else:
                state_probs, _ = self.markov_chain(token_feat_t, state_probs, hard=True)

            if self.use_continuous:
                token_levels_t = [lvl[:, t, :] for lvl in token_levels]
                pos_levels_t = [lvl[:, t, :] for lvl in pos_levels]
                for layer in self.toric_layers:
                    h_levels = layer(h_levels, token_levels_t, pos_levels_t, state_probs)

                h_concat = torch.cat(h_levels, dim=-1)
                logits_t = (
                    torch.einsum("bd,vd->bv", h_concat.real, all_token_complex.real)
                    + torch.einsum("bd,vd->bv", h_concat.imag, all_token_complex.imag)
                )
                logits_t = logits_t * self.logit_scale
            else:
                logits_t = self.state_to_logits(state_probs)

            all_logits.append(logits_t.unsqueeze(0))

        logits = torch.cat(all_logits, dim=0)
        targets = input_ids[:, 1:].transpose(0, 1)
        return logits, targets

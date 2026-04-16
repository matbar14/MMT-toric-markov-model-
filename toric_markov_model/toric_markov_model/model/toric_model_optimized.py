"""Optimized Toric-Markov model with KV-cache and parallel processing."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .cache import ToricCache
from .embeddings import QuantizedAngleEmbedding, QuantizedPositionalShifts
from .fractal_cell_optimized import OptimizedFractalToricCell
from .markov import DiscreteMarkovChain
from .quantizer import angles_to_complex, complex_to_features, quantize_angle


class ToricMarkovModelOptimized(nn.Module):
    """Optimized version with KV-cache, parallel scan, and compilation support."""
    
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
        chunk_size: int = 32,
        use_flash: bool = True,
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
        self.chunk_size = chunk_size
        self.use_flash = use_flash

        self.token_emb = QuantizedAngleEmbedding(vocab_size, dim_angles, n_bits)
        self.pos_emb = QuantizedPositionalShifts(max_len, dim_angles, n_bits)

        self.markov_chain = DiscreteMarkovChain(hidden_dim=2 * dim_angles, num_states=num_states)
        self.toric_layers = nn.ModuleList(
            [
                OptimizedFractalToricCell(
                    dim_angles=dim_angles,
                    num_levels=num_levels,
                    num_states=num_states,
                    use_attention=use_attention,
                    use_flash=use_flash,
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

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: ToricCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, ToricCache | None]:
        """
        Forward pass with optional KV-cache.
        
        Args:
            input_ids: [batch, seq_len]
            cache: Optional cache from previous forward pass
            use_cache: Whether to return updated cache
        
        Returns:
            logits: [seq_len-1, batch, vocab_size]
            targets: [seq_len-1, batch]
            cache: Updated cache if use_cache=True
        """
        if use_cache and cache is not None and not cache.is_empty():
            return self._forward_with_cache(input_ids, cache)
        
        return self._forward_full(input_ids, use_cache)

    def _forward_full(
        self,
        input_ids: torch.Tensor,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, ToricCache | None]:
        """Full forward pass without cache."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        token_phase = self.token_emb(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_phase = self.pos_emb(positions)

        token_levels = torch.chunk(token_phase, self.num_levels, dim=-1)
        pos_levels = torch.chunk(pos_phase, self.num_levels, dim=-1)

        level_dim = self.dim_angles // self.num_levels
        h_levels = [
            torch.ones(batch_size, level_dim, dtype=torch.cfloat, device=device)
            for _ in range(self.num_levels)
        ]

        state_probs = torch.zeros(batch_size, self.num_states, device=device)
        state_probs[:, 0] = 1.0

        all_logits = []
        all_token_complex = self._all_token_complex()

        # Process in chunks for better parallelization
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

        cache_out = None
        if use_cache:
            cache_out = ToricCache(h_levels=h_levels, state_probs=state_probs, seq_len=seq_len - 1)

        return logits, targets, cache_out

    def _forward_with_cache(
        self,
        input_ids: torch.Tensor,
        cache: ToricCache,
    ) -> tuple[torch.Tensor, torch.Tensor, ToricCache]:
        """
        Forward pass with cache (for incremental generation).
        Only processes the last token.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Only process the last token
        last_token_id = input_ids[:, -1:]
        token_phase = self.token_emb(last_token_id)
        
        # Use cache.seq_len as position, but clamp to max_len-1
        position_idx = min(cache.seq_len, self.max_len - 1)
        position = torch.tensor([position_idx], device=device).unsqueeze(0).expand(batch_size, -1)
        pos_phase = self.pos_emb(position)

        token_levels = torch.chunk(token_phase, self.num_levels, dim=-1)
        pos_levels = torch.chunk(pos_phase, self.num_levels, dim=-1)

        # Use cached states
        h_levels = cache.h_levels
        state_probs = cache.state_probs

        token_t = token_phase[:, 0, :]
        token_feat_t = complex_to_features(token_t)

        state_probs, _ = self.markov_chain(token_feat_t, state_probs, hard=True)

        if self.use_continuous:
            token_levels_t = [lvl[:, 0, :] for lvl in token_levels]
            pos_levels_t = [lvl[:, 0, :] for lvl in pos_levels]
            for layer in self.toric_layers:
                h_levels = layer(h_levels, token_levels_t, pos_levels_t, state_probs)

            h_concat = torch.cat(h_levels, dim=-1)
            all_token_complex = self._all_token_complex()
            logits_t = (
                torch.einsum("bd,vd->bv", h_concat.real, all_token_complex.real)
                + torch.einsum("bd,vd->bv", h_concat.imag, all_token_complex.imag)
            )
            logits_t = logits_t * self.logit_scale
        else:
            logits_t = self.state_to_logits(state_probs)

        # Update cache
        cache.update(h_levels, state_probs)

        # Return single logit
        logits = logits_t.unsqueeze(0)
        targets = input_ids[:, -1:].transpose(0, 1)

        return logits, targets, cache

    @torch.inference_mode()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively with KV-cache.
        
        Args:
            prompt: [batch, prompt_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None = no filtering)
            use_cache: Use KV-cache for faster generation
        
        Returns:
            generated: [batch, prompt_len + max_new_tokens]
        """
        batch_size = prompt.shape[0]
        device = prompt.device
        
        generated = prompt
        cache = ToricCache() if use_cache else None
        
        for _ in range(max_new_tokens):
            # Pad to max_len if needed
            if generated.shape[1] < self.max_len:
                input_ids = torch.cat([
                    generated,
                    torch.zeros(
                        batch_size,
                        self.max_len - generated.shape[1],
                        dtype=torch.long,
                        device=device,
                    )
                ], dim=1)
            else:
                input_ids = generated[:, -self.max_len:]
            
            logits, _, cache = self.forward(input_ids, cache=cache, use_cache=use_cache)
            
            # Get logits for next token
            next_logits = logits[-1, :, :] / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = -float('inf')
            
            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated

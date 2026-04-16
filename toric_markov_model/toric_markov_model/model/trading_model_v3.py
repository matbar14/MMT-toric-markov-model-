"""Trading model v3 with PATTERN DETECTION instead of direction classification.

CRITICAL CHANGE: Instead of predicting direction on every candle, we detect PATTERNS:
- CVD patterns (divergences, reversals, exhaustion, spikes)
- Basis patterns (spread anomalies)
- OI patterns (accumulation/distribution)
- Volume Profile patterns (POC breakouts)

Model outputs which pattern is present (multi-label classification).
Trade ONLY when a pattern is detected with high confidence.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .embeddings import QuantizedPositionalShifts
from .fractal_cell import FractalToricCell
from .markov import DiscreteMarkovChain
from .quantizer import angles_to_complex, complex_to_features, quantize_angle


class ContinuousFeatureEmbedding(nn.Module):
    """Embed continuous features as quantized angles (like QuantizedAngleEmbedding but for continuous input)."""
    
    def __init__(self, num_features: int, embedding_dim: int, n_bits: int = 8):
        super().__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.n_bits = n_bits
        
        # Project features to angles
        self.feature_to_angles = nn.Linear(num_features, embedding_dim)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, seq_len, num_features]
        Returns:
            complex phases: [batch, seq_len, embedding_dim]
        """
        # Project to angle space
        angles = self.feature_to_angles(features)
        # Bound to [-pi, pi]
        angles = torch.tanh(angles) * math.pi
        # Quantize
        angles_q = quantize_angle(angles, self.n_bits)
        # Convert to complex
        return angles_to_complex(angles_q)


class ToricTradingModelV3(nn.Module):
    """COMPLETE fractal toric architecture for PATTERN DETECTION."""
    
    def __init__(
        self,
        num_features: int = 35,
        dim_angles: int = 128,
        max_len: int = 64,
        num_states: int = 256,
        num_levels: int = 4,
        num_layers: int = 3,
        n_bits: int = 8,
        use_attention: bool = True,
        num_patterns: int = 17,  # 16 patterns + hold
        predict_return: bool = True,
    ):
        super().__init__()
        if dim_angles % num_levels != 0:
            raise ValueError("dim_angles must be divisible by num_levels")
        
        self.num_features = num_features
        self.dim_angles = dim_angles
        self.max_len = max_len
        self.num_states = num_states
        self.num_levels = num_levels
        self.num_layers = num_layers
        self.n_bits = n_bits
        self.use_attention = use_attention
        self.num_patterns = num_patterns
        self.predict_return = predict_return
        
        # Feature embedding (replaces token_emb)
        self.feature_emb = ContinuousFeatureEmbedding(num_features, dim_angles, n_bits)
        
        # Positional shifts (SAME as original)
        self.pos_emb = QuantizedPositionalShifts(max_len, dim_angles, n_bits)
        
        # Markov chain (SAME as original)
        self.markov_chain = DiscreteMarkovChain(
            hidden_dim=2 * dim_angles,
            num_states=num_states,
        )
        
        # Fractal toric layers (SAME as original)
        self.toric_layers = nn.ModuleList([
            FractalToricCell(
                dim_angles=dim_angles,
                num_levels=num_levels,
                num_states=num_states,
                use_attention=use_attention,
            )
            for _ in range(num_layers)
        ])
        
        # FIX 2: Feature anchor projection for residual connection
        self.feature_anchor = nn.Linear(num_features, 2 * dim_angles)
        
        # PRIMARY TASK: Pattern detection (multi-label classification)
        # Each pattern is independent - can have multiple patterns active
        self.pattern_head = nn.Sequential(
            nn.Linear(2 * dim_angles, dim_angles),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_angles, num_patterns),
            # No sigmoid - use BCEWithLogitsLoss
        )
        
        # Pattern confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(2 * dim_angles, dim_angles),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_angles, num_patterns),
            nn.Sigmoid(),
        )
        
        # AUXILIARY TASKS for multi-task learning (now take 2*dim_angles input)
        if predict_return:
            # Task 1: Price return
            self.return_head = nn.Sequential(
                nn.Linear(2 * dim_angles, dim_angles),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim_angles, 1),
            )
            
            # Task 2: Volume change
            self.volume_head = nn.Sequential(
                nn.Linear(2 * dim_angles, dim_angles),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim_angles, 1),
            )
            
            # Task 3: CVD change
            self.cvd_head = nn.Sequential(
                nn.Linear(2 * dim_angles, dim_angles),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim_angles, 1),
            )
            
            # Task 4: POC movement
            self.poc_head = nn.Sequential(
                nn.Linear(2 * dim_angles, dim_angles),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim_angles, 1),
            )
            
            # Task 5: Market regime (trend/flat classification)
            self.regime_head = nn.Sequential(
                nn.Linear(2 * dim_angles, dim_angles),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim_angles, 3),  # 3 classes: flat, weak_trend, strong_trend
            )
            
            # Task 6: Reversal detection (binary)
            self.reversal_head = nn.Sequential(
                nn.Linear(2 * dim_angles, dim_angles),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim_angles, 2),  # 2 classes: no_reversal, reversal
            )
            
            # Task 7: Breakout detection
            self.breakout_head = nn.Sequential(
                nn.Linear(2 * dim_angles, dim_angles),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(dim_angles, 3),  # 3 classes: no_breakout, breakout_up, breakout_down
            )
    def forward(
        self,
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass - detect patterns instead of predicting direction.
        Returns dict with pattern predictions.
        
        Args:
            features: [batch, seq_len, num_features]
        
        Returns:
            Dictionary with pattern_logits, pattern_confidence, and auxiliary predictions
        """
        batch_size, seq_len = features.shape[0], features.shape[1]
        device = features.device
        
        # Feature embedding (replaces token_emb)
        token_phase = self.feature_emb(features)  # [batch, seq_len, dim_angles]
        
        # Positional shifts (SAME as original)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_phase = self.pos_emb(positions)  # [batch, seq_len, dim_angles]
        
        # Split into levels (SAME as original)
        token_levels = torch.chunk(token_phase, self.num_levels, dim=-1)
        pos_levels = torch.chunk(pos_phase, self.num_levels, dim=-1)
        
        # Initialize hidden states (SAME as original)
        level_dim = self.dim_angles // self.num_levels
        h_levels = [
            torch.ones(batch_size, level_dim, dtype=torch.cfloat, device=device)
            for _ in range(self.num_levels)
        ]
        
        # Initialize markov state (SAME as original)
        state_probs = torch.zeros(batch_size, self.num_states, device=device)
        state_probs[:, 0] = 1.0
        
        # Process sequence (SAME as original forward_with_per_step_logits)
        for t in range(seq_len):
            # Get current token phase
            token_t = token_phase[:, t, :]
            token_feat_t = complex_to_features(token_t)
            
            # Update markov state
            if self.training:
                state_probs, _ = self.markov_chain(token_feat_t, state_probs, hard=False)
            else:
                state_probs, _ = self.markov_chain(token_feat_t, state_probs, hard=True)
            
            # Update hidden states through fractal layers
            token_levels_t = [lvl[:, t, :] for lvl in token_levels]
            pos_levels_t = [lvl[:, t, :] for lvl in pos_levels]
            
            for layer in self.toric_layers:
                h_levels = layer(h_levels, token_levels_t, pos_levels_t, state_probs)
        
        # Concatenate final hidden states (SAME as original)
        h_concat = torch.cat(h_levels, dim=-1)  # [batch, dim_angles] complex
        
        # FIX 1: Use BOTH real and imaginary parts - don't lose information!
        h_out = torch.cat([h_concat.real, h_concat.imag], dim=-1)  # [batch, 2*dim_angles]
        
        # FIX 2: Add residual connection - anchor to original features
        # Get last timestep features as anchor
        last_features = features[:, -1, :]  # [batch, num_features]
        # Project to same dimension
        features_proj = self.feature_anchor(last_features)  # [batch, 2*dim_angles]
        # Combine with residual
        h_out = h_out + 0.1 * features_proj  # Small residual weight
        
        # All predictions
        outputs = {
            'pattern_logits': self.pattern_head(h_out),  # [batch, num_patterns]
            'pattern_confidence': self.confidence_head(h_out),  # [batch, num_patterns]
        }
        
        if self.predict_return:
            outputs['predicted_return'] = torch.tanh(self.return_head(h_out))
            outputs['predicted_volume_change'] = torch.tanh(self.volume_head(h_out))
            outputs['predicted_cvd_change'] = torch.tanh(self.cvd_head(h_out))
            outputs['predicted_poc_movement'] = torch.tanh(self.poc_head(h_out))
            outputs['regime_logits'] = self.regime_head(h_out)
            outputs['reversal_logits'] = self.reversal_head(h_out)
            outputs['breakout_logits'] = self.breakout_head(h_out)
        
        return outputs
    
    @torch.inference_mode()
    def detect_patterns(
        self,
        features: torch.Tensor,
        confidence_threshold: float = 0.7,
        pattern_prob_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """
        Detect patterns with confidence filtering.
        
        Args:
            features: [batch, seq_len, num_features]
            confidence_threshold: Minimum confidence to report pattern
            pattern_prob_threshold: Minimum pattern probability to report pattern
        
        Returns:
            Dictionary with detected patterns and their confidences
        """
        outputs = self.forward(features)
        
        # Apply sigmoid to get probabilities
        pattern_probs = torch.sigmoid(outputs['pattern_logits'])
        pattern_confidence = outputs['pattern_confidence']
        pattern_scores = pattern_probs * pattern_confidence

        # Last label is reserved for "hold"
        hold_idx = self.num_patterns - 1
        non_hold_probs = pattern_probs[..., :hold_idx]
        non_hold_confidence = pattern_confidence[..., :hold_idx]
        non_hold_scores = pattern_scores[..., :hold_idx]

        best_non_hold_pattern = torch.argmax(non_hold_scores, dim=-1)
        best_non_hold_prob = torch.gather(
            non_hold_probs, 1, best_non_hold_pattern.unsqueeze(1)
        ).squeeze(1)
        best_non_hold_confidence = torch.gather(
            non_hold_confidence, 1, best_non_hold_pattern.unsqueeze(1)
        ).squeeze(1)
        best_non_hold_score = torch.gather(
            non_hold_scores, 1, best_non_hold_pattern.unsqueeze(1)
        ).squeeze(1)

        hold_prob = pattern_probs[:, hold_idx]
        hold_confidence = pattern_confidence[:, hold_idx]
        hold_score = pattern_scores[:, hold_idx]

        # Multi-label gating: do not force argmax against "hold".
        has_pattern = (
            (best_non_hold_prob >= pattern_prob_threshold)
            & (best_non_hold_confidence >= confidence_threshold)
        )

        strongest_pattern = torch.where(
            has_pattern,
            best_non_hold_pattern,
            torch.full_like(best_non_hold_pattern, hold_idx),
        )
        strongest_confidence = torch.where(
            has_pattern,
            best_non_hold_confidence,
            hold_confidence,
        )

        active_patterns = (
            (pattern_probs >= pattern_prob_threshold)
            & (pattern_confidence >= confidence_threshold)
        )
        # "Hold" should only be active if no other pattern passed thresholds.
        active_patterns[:, hold_idx] = ~torch.any(active_patterns[:, :hold_idx], dim=1)
        
        result = {
            'pattern_probs': pattern_probs,
            'pattern_confidence': pattern_confidence,
            'pattern_scores': pattern_scores,
            'strongest_pattern': strongest_pattern,
            'strongest_confidence': strongest_confidence,
            'has_pattern': has_pattern,
            'best_non_hold_pattern': best_non_hold_pattern,
            'best_non_hold_prob': best_non_hold_prob,
            'best_non_hold_confidence': best_non_hold_confidence,
            'best_non_hold_score': best_non_hold_score,
            'hold_prob': hold_prob,
            'hold_confidence': hold_confidence,
            'hold_score': hold_score,
            'active_patterns': active_patterns,
        }
        
        if self.predict_return:
            result['predicted_return'] = outputs['predicted_return']
            result['predicted_volume_change'] = outputs['predicted_volume_change']
            result['predicted_cvd_change'] = outputs['predicted_cvd_change']
            result['predicted_poc_movement'] = outputs['predicted_poc_movement']
            result['regime'] = torch.argmax(outputs['regime_logits'], dim=-1)
            result['reversal'] = torch.argmax(outputs['reversal_logits'], dim=-1)
            result['breakout'] = torch.argmax(outputs['breakout_logits'], dim=-1)
        
        return result

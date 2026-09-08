"""Toric encoder with an event gate, conditional pattern labels and regression."""

from __future__ import annotations

import math

import torch
from torch import nn

from .embeddings import QuantizedPositionalShifts
from .fractal_cell import FractalToricCell
from .markov import DiscreteMarkovChain
from .quantizer import angles_to_complex, complex_to_features, quantize_angle


class ContinuousFeatureEmbedding(nn.Module):
    def __init__(self, num_features: int, embedding_dim: int, n_bits: int = 8):
        super().__init__()
        self.n_bits = n_bits
        self.feature_to_angles = nn.Linear(num_features, embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        angles = torch.tanh(self.feature_to_angles(features)) * math.pi
        return angles_to_complex(quantize_angle(angles, self.n_bits))


class ToricEncoder(nn.Module):
    def __init__(
        self, num_features: int = 28, dim_angles: int = 64, max_len: int = 64,
        num_states: int = 128, num_levels: int = 4, num_layers: int = 2,
        n_bits: int = 8, use_attention: bool = True,
    ):
        super().__init__()
        dimensions = (num_features, dim_angles, max_len, num_states, num_levels, num_layers)
        if any(value <= 0 for value in dimensions) or dim_angles % num_levels:
            raise ValueError("positive dimensions and dim_angles divisible by num_levels required")
        if not 1 <= n_bits <= 16:
            raise ValueError("invalid n_bits")
        self.num_features = num_features
        self.dim_angles = dim_angles
        self.max_len = max_len
        self.num_states = num_states
        self.num_levels = num_levels
        self.feature_emb = ContinuousFeatureEmbedding(num_features, dim_angles, n_bits)
        self.pos_emb = QuantizedPositionalShifts(max_len, dim_angles, n_bits)
        self.markov_chain = DiscreteMarkovChain(2 * dim_angles, num_states)
        self.toric_layers = nn.ModuleList([
            FractalToricCell(dim_angles, num_levels, num_states, use_attention)
            for _ in range(num_layers)
        ])
        self.complex_feature_fusion = nn.Sequential(
            nn.Linear(2 * dim_angles, 2 * dim_angles),
            nn.LayerNorm(2 * dim_angles), nn.GELU(),
        )
        self.feature_anchor = nn.Linear(num_features, 2 * dim_angles)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.num_features:
            raise ValueError("features must have shape (batch, sequence, num_features)")
        batch_size, seq_len, _ = features.shape
        if batch_size == 0 or not 1 <= seq_len <= self.max_len:
            raise ValueError("empty batch or sequence length outside model limits")
        if not torch.isfinite(features).all():
            raise ValueError("features contain NaN or infinity")
        token_phase = self.feature_emb(features)
        positions = torch.arange(seq_len, device=features.device)
        pos_phase = self.pos_emb(positions).unsqueeze(0).expand(batch_size, -1, -1)
        token_levels = token_phase.chunk(self.num_levels, dim=-1)
        pos_levels = pos_phase.chunk(self.num_levels, dim=-1)
        level_dim = self.dim_angles // self.num_levels
        states = [torch.ones(batch_size, level_dim, dtype=token_phase.dtype, device=features.device)
                  for _ in range(self.num_levels)]
        state_probs = features.new_zeros(batch_size, self.num_states)
        state_probs[:, 0] = 1
        for timestep in range(seq_len):
            state_probs, _ = self.markov_chain(
                complex_to_features(token_phase[:, timestep]), state_probs,
            )
            for layer in self.toric_layers:
                states = layer(states, [level[:, timestep] for level in token_levels],
                               [level[:, timestep] for level in pos_levels], state_probs)
        hidden = self.complex_feature_fusion(complex_to_features(torch.cat(states, dim=-1)))
        return hidden + 0.1 * self.feature_anchor(features[:, -1])


class ToricTradingModelV3(ToricEncoder):
    AUX_NAMES = (
        "predicted_return", "predicted_volume_change",
        "predicted_cvd_change", "predicted_poc_movement",
    )

    def __init__(
        self, num_features: int = 28, dim_angles: int = 64, max_len: int = 64,
        num_states: int = 128, num_levels: int = 4, num_layers: int = 2,
        n_bits: int = 8, use_attention: bool = True, num_patterns: int = 17,
        predict_return: bool = True, dropout: float = 0.2,
    ):
        if num_patterns < 2 or not 0 <= dropout < 1:
            raise ValueError("invalid num_patterns or dropout")
        super().__init__(num_features, dim_angles, max_len, num_states, num_levels,
                         num_layers, n_bits, use_attention)
        self.config = dict(
            num_features=num_features, dim_angles=dim_angles, max_len=max_len,
            num_states=num_states, num_levels=num_levels, num_layers=num_layers,
            n_bits=n_bits, use_attention=use_attention, num_patterns=num_patterns,
            predict_return=predict_return, dropout=dropout,
        )
        self.num_patterns = num_patterns
        self.predict_return = predict_return
        self.decision_thresholds = dict(confidence_threshold=0.0, pattern_prob_threshold=0.5,
                                        gate_threshold=0.5)
        self.pattern_head = self._head(dim_angles, num_patterns - 1, dropout)
        self.non_hold_gate_head = self._head(dim_angles, 1, dropout)
        if predict_return:
            self.aux_head = self._head(dim_angles, 4, dropout)
        self.register_buffer("aux_target_mean", torch.zeros(4))
        self.register_buffer("aux_target_std", torch.ones(4))
        self.register_buffer("has_aux_stats", torch.tensor(False))

    @staticmethod
    def _head(dim_angles: int, output_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(2 * dim_angles, dim_angles), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_angles, output_dim),
        )

    def set_aux_target_stats(self, stats: dict) -> None:
        mean = torch.as_tensor(stats["aux_target_mean"], device=self.aux_target_mean.device)
        std = torch.as_tensor(stats["aux_target_std"], device=self.aux_target_std.device)
        if mean.shape != (4,) or std.shape != (4,):
            raise ValueError("auxiliary statistics must have shape (4,)")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("invalid auxiliary statistics")
        self.aux_target_mean.copy_(mean)
        self.aux_target_std.copy_(std)
        self.has_aux_stats.fill_(True)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encode(features)
        gate_logit = self.non_hold_gate_head(hidden)
        outputs = {
            "pattern_logits": torch.cat((self.pattern_head(hidden), -gate_logit), dim=-1),
            "non_hold_logit": gate_logit,
        }
        if self.predict_return:
            auxiliary = self.aux_head(hidden)
            outputs.update({name: auxiliary[:, index:index + 1]
                            for index, name in enumerate(self.AUX_NAMES)})
        return outputs

    def decode_outputs(
        self, outputs: dict[str, torch.Tensor], confidence_threshold: float = 0.0,
        pattern_prob_threshold: float = 0.5, gate_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Confidence is a joint gate-times-conditional score, not calibrated certainty."""
        if any(not 0 <= value <= 1 for value in
               (confidence_threshold, pattern_prob_threshold, gate_threshold)):
            raise ValueError("decision thresholds must be in [0, 1]")
        conditional = outputs["pattern_logits"][:, :-1].sigmoid()
        gate = outputs["non_hold_logit"].sigmoid()
        scores = conditional * gate
        active = ((conditional >= pattern_prob_threshold) & (scores >= confidence_threshold)
                  & (gate >= gate_threshold))
        has_pattern = active.any(dim=-1)
        selected = scores.masked_fill(~active, -1).argmax(dim=-1)
        best = torch.where(has_pattern, selected, scores.argmax(dim=-1))
        best_prob = conditional.gather(1, best[:, None]).squeeze(1)
        best_score = scores.gather(1, best[:, None]).squeeze(1)
        hold_prob = 1 - gate.squeeze(1)
        joint = torch.cat((scores, hold_prob[:, None]), dim=-1)
        return {
            "pattern_probs": joint, "conditional_pattern_probs": conditional,
            "pattern_confidence": joint, "pattern_scores": joint,
            "strongest_pattern": torch.where(has_pattern, best, self.num_patterns - 1),
            "strongest_confidence": torch.where(has_pattern, best_score, hold_prob),
            "has_pattern": has_pattern, "best_non_hold_pattern": best,
            "best_non_hold_prob": best_prob, "best_non_hold_confidence": best_score,
            "best_non_hold_score": best_score, "non_hold_prob": gate.squeeze(1),
            "hold_prob": hold_prob, "hold_confidence": hold_prob, "hold_score": hold_prob,
            "active_patterns": torch.cat((active, ~has_pattern[:, None]), dim=-1),
        }

    def set_decision_thresholds(self, thresholds):
        if (set(thresholds) != set(self.decision_thresholds) or
                any(not 0 <= value <= 1 for value in thresholds.values())):
            raise ValueError("complete decision thresholds in [0, 1] required")
        self.decision_thresholds = dict(thresholds)

    @torch.inference_mode()
    def detect_patterns(
        self, features: torch.Tensor, confidence_threshold: float | None = None,
        pattern_prob_threshold: float | None = None, gate_threshold: float | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.training:
            raise RuntimeError("call model.eval() before inference")
        thresholds = dict(self.decision_thresholds)
        overrides = dict(confidence_threshold=confidence_threshold, pattern_prob_threshold=pattern_prob_threshold,
                         gate_threshold=gate_threshold)
        thresholds.update({name: value for name, value in overrides.items() if value is not None})
        outputs = self(features)
        result = self.decode_outputs(outputs, **thresholds)
        if self.predict_return:
            if not self.has_aux_stats.item():
                raise RuntimeError("load train auxiliary statistics before inference")
            for index, name in enumerate(self.AUX_NAMES):
                result[name] = outputs[name] * self.aux_target_std[index] + self.aux_target_mean[index]
        return result

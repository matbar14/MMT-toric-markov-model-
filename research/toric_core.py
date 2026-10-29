"""Clean toric core: a phase-only recurrent network living on the flat torus.

State space
-----------
The hidden state is a vector of phases h in T^d = (S^1)^d, represented as
unit-modulus complex numbers.  Every learned dynamical parameter is an
*angle* and every transition is an *addition of angles* (a rotation), so the
state provably never leaves the torus.  Renormalization is kept only as a
floating-point guard, not as part of the mathematics.

What is cleaned up relative to the legacy implementation
--------------------------------------------------------
Compared with ``toric_markov_model/model/fractal_cell.py`` + ``markov.py``:

1. State rotations are parameterized by angles ``theta[state, dim]`` and
   applied as ``exp(i * theta)`` — unit modulus by construction.  The legacy
   code used unconstrained complex parameters (``randn * 0.1``) whose
   magnitudes are discarded by renormalization: dead degrees of freedom whose
   only live part is the random initial phase.
2. ``kernel="stochastic"`` turns the discrete chain into a *proper* Markov
   kernel: ``p' proportional to (p @ P) * exp(token_logits + bias)`` with a
   row-stochastic P — a product of the transition prior and observation
   evidence, i.e. an HMM-style filter with soft evidence.  The legacy
   unconstrained ``prev @ A`` map is preserved as ``kernel="free"``.
3. The gate mixing uses the torus geodesic: a convex combination of unit
   vectors renormalized is exactly the spherical interpolation between the
   old and new phases, so ``normalize(g * h_new + (1 - g) * h_old)`` is kept —
   but now it is exact, not incidental, because both operands are unit modulus
   by construction.
4. Optional real-valued amplitude bypass (``use_amplitude=True``): the pure
   phase pipeline discards magnitudes entirely.  Cyclic data may not need
   amplitudes; amplitude-heavy data does.  The flag adds a parallel linear
   path from the raw features into the fusion layer.

The 8-bit straight-through angle quantizer is conceptually reused from
``model/quantizer.py`` and duplicated here so that this module is fully
self-contained and the trading package can be refactored later without
breaking these experiments.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .baselines import moving_avg


class _AngleSTE(torch.autograd.Function):
    """Straight-through quantization of angles in [-pi, pi] onto a uniform grid."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, n_bits: int) -> torch.Tensor:
        levels = 2**n_bits
        scale = 2 * math.pi / levels
        norm = (x + math.pi) / (2 * math.pi)
        idx = torch.floor(norm * levels).long().clamp(0, levels - 1)
        return idx.float() * scale - math.pi

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def quantize_angle(x: torch.Tensor, n_bits: int = 8) -> torch.Tensor:
    return _AngleSTE.apply(x, n_bits)


def unit(angles: torch.Tensor) -> torch.Tensor:
    """Angles -> unit-modulus complex numbers exp(i * angles)."""
    return torch.polar(torch.ones_like(angles), angles)


def normalize(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project back onto the unit circle (floating-point guard)."""
    return z / (z.abs() + eps)


class ToricMarkovChain(nn.Module):
    """Input-conditioned transition over K discrete latent states.

    ``kernel="stochastic"`` (default) treats the state posterior as an HMM
    filter: the transition prior ``(p @ P)`` is combined with the observation
    evidence ``exp(token_logits)`` in log space.  ``P`` is row-stochastic by
    construction (row-softmax of ``log_P``), so the learned transition matrix
    can be inspected directly.

    ``kernel="free"`` reproduces the legacy behaviour
    ``logits += prev_probs @ A`` with an unconstrained real matrix.
    """

    def __init__(
        self,
        input_dim: int,
        num_states: int,
        kernel: str = "stochastic",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if kernel not in ("stochastic", "free"):
            raise ValueError("kernel must be 'stochastic' or 'free'")
        if not 0 < temperature < float("inf"):
            raise ValueError("temperature must be finite and positive")
        self.num_states = num_states
        self.kernel = kernel
        self.temperature = temperature
        self.token_proj = nn.Linear(input_dim, num_states, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_states))
        if kernel == "stochastic":
            self.log_P = nn.Parameter(torch.zeros(num_states, num_states))
        else:
            self.A = nn.Parameter(torch.randn(num_states, num_states) * 0.01)

    def forward(self, features: torch.Tensor, prev_probs: torch.Tensor) -> torch.Tensor:
        logits = self.token_proj(features) + self.bias
        if self.kernel == "stochastic":
            transition = torch.softmax(self.log_P, dim=-1)
            logits = logits + torch.log(prev_probs @ transition + 1e-10)
        else:
            logits = logits + prev_probs @ self.A
        return torch.softmax(logits / self.temperature, dim=-1)

    def transition_matrix(self) -> torch.Tensor | None:
        """The learned row-stochastic transition matrix (stochastic mode only)."""
        if self.kernel != "stochastic":
            return None
        return torch.softmax(self.log_P.detach(), dim=-1)


class ToricPhaseLayer(nn.Module):
    """One toric transition of all levels.

    Each level is rotated by a state-selected phase kick (the expected angle
    under the current state posterior), multiplied by the token and positional
    phases, optionally pulled toward a learned context phase (attention), and
    geodesically mixed with the previous state under a learned gate.

    All operands are unit modulus by construction, so the state remains on the
    torus exactly; ``normalize`` calls are numerical guards.
    """

    def __init__(
        self,
        dim_angles: int,
        num_levels: int,
        num_states: int,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        if dim_angles % num_levels:
            raise ValueError("dim_angles must be divisible by num_levels")
        self.num_levels = num_levels
        self.dim_per_level = dim_angles // num_levels
        # Phase kick per (level, state, dimension), in radians.
        self.theta = nn.Parameter(
            torch.randn(num_levels, num_states, self.dim_per_level) * 0.5
        )
        # Context phase per level (attention pull target).
        self.ctx_angle = nn.Parameter(torch.randn(num_levels, self.dim_per_level))
        self.attn_scale = nn.Parameter(torch.tensor(1.0))
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.use_attention = use_attention

    def forward(
        self,
        h_levels: list[torch.Tensor],
        token_levels: list[torch.Tensor],
        pos_levels: list[torch.Tensor],
        state_probs: torch.Tensor,
    ) -> list[torch.Tensor]:
        updated: list[torch.Tensor] = []
        g = torch.sigmoid(self.gate)
        for level in range(self.num_levels):
            kick = state_probs @ self.theta[level]  # (B, Dl) expected angle
            h = h_levels[level] * unit(kick) * token_levels[level] * pos_levels[level]
            h = normalize(h)
            if self.use_attention:
                context = unit(self.ctx_angle[level])
                score = (h * context.conj()).sum(dim=-1).abs()  # (B,)
                weight = torch.sigmoid(score * self.attn_scale).unsqueeze(-1)
                h = normalize(h + weight * context)
            updated.append(normalize(g * h + (1 - g) * h_levels[level]))
        return updated


class ToricEncoder(nn.Module):
    """Phase-only encoder over a window of real-valued features.

    Features are embedded as quantized angles (phases), positions act as
    learned quantized phase shifts (a toric cousin of RoPE), a discrete
    Markov chain tracks a soft latent regime, and every level of the hidden
    state is rotated by regime-selected phase kicks at every step.
    """

    def __init__(
        self,
        num_features: int,
        dim_angles: int = 48,
        num_levels: int = 2,
        num_states: int = 24,
        num_layers: int = 1,
        n_bits: int = 8,
        kernel: str = "stochastic",
        use_attention: bool = True,
        use_amplitude: bool = False,
        use_diff_input: bool = False,
        quantize_positions: bool = True,
        max_len: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(dim_angles, num_levels, num_states, num_layers, max_len) <= 0:
            raise ValueError("all size parameters must be positive")
        if dim_angles % num_levels:
            raise ValueError("dim_angles must be divisible by num_levels")
        self.num_features = num_features
        self.dim_angles = dim_angles
        self.num_levels = num_levels
        self.max_len = max_len
        self.n_bits = n_bits
        self.quantize_positions = quantize_positions
        self.use_amplitude = use_amplitude
        self.use_diff_input = use_diff_input

        # First differences double the input width when enabled: they expose the
        # local phase increment directly, which is the regime identifier.
        effective_features = num_features * 2 if use_diff_input else num_features
        self.feature_to_angle = nn.Linear(effective_features, dim_angles)
        self.pos_angle = nn.Parameter(torch.randn(max_len, dim_angles) * 0.1)
        self.chain = ToricMarkovChain(2 * dim_angles, num_states, kernel)
        self.toric_layers = nn.ModuleList(
            [
                ToricPhaseLayer(dim_angles, num_levels, num_states, use_attention)
                for _ in range(num_layers)
            ]
        )
        hidden = 2 * dim_angles
        self.fusion = nn.Sequential(
            nn.Linear(2 * dim_angles, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden),
        )
        if use_amplitude:
            self.amp_proj = nn.Linear(num_features, hidden)
        self.hidden_dim = hidden

    def encode(
        self, features: torch.Tensor, return_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.num_features:
            raise ValueError("features must have shape (batch, sequence, num_features)")
        batch_size, seq_len, _ = features.shape
        if batch_size == 0 or not 1 <= seq_len <= self.max_len:
            raise ValueError("empty batch or sequence length outside model limits")
        if not torch.isfinite(features).all():
            raise ValueError("features contain NaN or infinity")

        model_input = features
        if self.use_diff_input:
            model_input = torch.cat(
                [features, features.diff(dim=1, prepend=features[:, :1, :])], dim=-1
            )

        token_angle = torch.tanh(self.feature_to_angle(model_input)) * math.pi
        token = unit(quantize_angle(token_angle, self.n_bits))
        pos_angles = self.pos_angle[:seq_len]
        if self.quantize_positions:
            pos_angles = quantize_angle(pos_angles, self.n_bits)
        pos = unit(pos_angles)  # (T, D)

        token_levels = token.chunk(self.num_levels, dim=-1)  # each (B, T, Dl)
        pos_levels = [p for p in pos.chunk(self.num_levels, dim=-1)]  # each (T, Dl)

        state_probs = features.new_zeros(batch_size, self.chain.num_states)
        state_probs[:, 0] = 1.0
        h_levels = [
            torch.ones(
                batch_size,
                self.dim_angles // self.num_levels,
                dtype=torch.cfloat,
                device=features.device,
            )
            for _ in range(self.num_levels)
        ]
        chain_features = torch.cat([token.real, token.imag], dim=-1)

        for timestep in range(seq_len):
            state_probs = self.chain(chain_features[:, timestep], state_probs)
            for layer in self.toric_layers:
                h_levels = layer(
                    h_levels,
                    [level[:, timestep] for level in token_levels],
                    [level[timestep] for level in pos_levels],
                    state_probs,
                )

        hidden = torch.cat(
            [h.real for h in h_levels] + [h.imag for h in h_levels], dim=-1
        )
        out = self.fusion(hidden)
        if self.use_amplitude:
            out = out + self.amp_proj(features[:, -1])
        if return_state:
            return out, state_probs
        return out

    def forward(
        self, features: torch.Tensor, return_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.encode(features, return_state=return_state)


class ToricForecaster(nn.Module):
    """Toric encoder with a linear head producing an H-step, C-channel forecast."""

    def __init__(
        self,
        num_features: int,
        horizon: int,
        num_channels: int = 1,
        **encoder_kwargs: object,
    ) -> None:
        super().__init__()
        self.encoder = ToricEncoder(num_features, **encoder_kwargs)
        self.horizon = horizon
        self.num_channels = num_channels
        self.head = nn.Linear(self.encoder.hidden_dim, horizon * num_channels)

    def forward(
        self, features: torch.Tensor, return_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if return_state:
            hidden, state_probs = self.encoder(features, return_state=True)
            forecast = self.head(hidden).view(
                features.shape[0], self.horizon, self.num_channels
            )
            return forecast, state_probs
        hidden = self.encoder(features)
        return self.head(hidden).view(
            features.shape[0], self.horizon, self.num_channels
        )


class HybridForecaster(nn.Module):
    """Linear autoregressive base + toric regime correction.

    Quasi-periodic series are dominated by structure a plain linear map from
    T lags to H outputs already captures (linear prediction theory: a fixed-
    frequency sinusoid satisfies a 2-term linear recurrence).  This forecaster
    keeps that linear base and lets the toric encoder add a *regime-conditional
    correction* on top.  The correction head is zero-initialized, so training
    starts exactly at the linear baseline: the toric core contributes only if
    it demonstrably reduces the loss.  This is the fairest test of whether the
    invented architecture adds value beyond linearity.
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        horizon: int,
        num_channels: int = 1,
        linear_kernel: int = 25,
        **encoder_kwargs: object,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.num_channels = num_channels
        self.linear_kernel = linear_kernel
        self.trend_head = nn.Linear(seq_len, horizon)
        self.seasonal_head = nn.Linear(seq_len, horizon)
        self.encoder = ToricEncoder(num_features, **encoder_kwargs)
        self.correction_head = nn.Linear(
            self.encoder.hidden_dim, horizon * num_channels
        )
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def linear_base(self, features: torch.Tensor) -> torch.Tensor:
        trend = moving_avg(features, self.linear_kernel)
        seasonal = features - trend
        base = self.trend_head(trend.transpose(1, 2)) + self.seasonal_head(
            seasonal.transpose(1, 2)
        )
        return base.transpose(1, 2)

    def forward(
        self, features: torch.Tensor, return_state: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.shape[0]
        base = self.linear_base(features)
        if return_state:
            hidden, state_probs = self.encoder(features, return_state=True)
        else:
            hidden, state_probs = self.encoder(features), None
        correction = self.correction_head(hidden).view(
            batch_size, self.horizon, self.num_channels
        )
        forecast = base + correction
        if return_state:
            return forecast, state_probs
        return forecast


def markov_agreement(
    true_states: torch.Tensor, state_probs: torch.Tensor, num_states: int
) -> float:
    """Best-permutation agreement between argmax inferred states and ground truth.

    The discrete chain is unsupervised: its state ids have no inherent meaning.
    This scores the best label permutation (feasible for small K) so the
    interpretability claim — "the chain discovers the regimes" — is checkable.
    """
    import itertools

    if true_states.ndim != 1 or state_probs.ndim != 2:
        raise ValueError("expected 1-D true states and (N, K) probabilities")
    if state_probs.shape[-1] != num_states:
        raise ValueError(
            "num_states must equal the number of chain states for a permutation check"
        )
    inferred = state_probs.argmax(dim=-1)
    best = 0.0
    for perm in itertools.permutations(range(num_states)):
        mapped = torch.tensor(
            [perm[int(i)] for i in inferred.tolist()], device=inferred.device
        )
        best = max(best, (mapped == true_states).float().mean().item())
    return best

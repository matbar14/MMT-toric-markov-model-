"""Round-4 pre-control 2: a fully differentiable phase-rotation recurrence.

Round 3 localized the toric model's failure to gradient descent through the
phase-recurrent parameterization.  The cheapest test of the Round-4 candidate
("make the rotations differentiable end-to-end, initialize the dynamics near
identity") is to strip the suspected obstacles and re-run the 2-minute overfit
probe: no straight-through quantizer, no discrete chain, no attention pull —
only a continuous unit-modulus state rotated by input-dependent learned kicks,

    h_{t+1} = normalize( h_t * exp(i * (theta_base + f(x_t))) ),   h_0 = 1,

with f either linear (``--kick linear``) or a two-layer MLP (``--kick mlp``),
followed by the SAME fusion + linear head as the toric encoder.  Everything
is differentiable; the state provably stays on the torus; the dynamics start
at the identity (theta_base init 0).

Reading the results (mirrors ``overfit_budget``):
  - probe (300 sequences, 100 epochs, no early stopping): MMT plateaus at
    min train MSE ~0.496-0.509 / test 0.513-0.530; the flat MLP overfits to
    0.029 train / 0.154 test.  If the differentiable rotation ALSO stays at
    the mean-predictor plateau, the barrier is the phase-recurrence form
    itself, not differentiability — Round 4 as conceived is refuted in its
    cheapest instance.  If it fits materially below ~0.45 train, the
    differentiable parameterization is confirmed as the Round-4 direction.
  - nonlinear task (2,500 sequences, 30 epochs): MMT test MSE 0.8963, the
    analytic readout 0.5311, MoE-linear 0.3957.  Where does the differentiable
    rotation land between them?

Usage (repository root)::

    # 2-minute overfit probe (same 300-sequence task as Round 1/3)
    python -m research.phase_rnn --task probe --kick mlp \
        --lrs 1e-3,3e-3 --seeds 0,1 --epochs 100 \
        --out research/results_phase_rnn_probe.json

    # the Round-3 nonlinear regime task, full protocol
    python -m research.phase_rnn --task nonlinear --kick mlp \
        --epochs 30 --patience 5 --diff-input --amplitude --seed 0 \
        --out research/results_phase_rnn_nonlinear.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .nonlinear_torus import (
    phase_readout_forecast,
    simulate_regime_switching_oscillator,
)
from .synthetic_torus import simulate_switching_rotation
from .train_utils import count_parameters, evaluate, evaluate_naive, train_model


class PhaseRotationEncoder(nn.Module):
    """Continuous differentiable phase recurrence + the toric fusion/head stack.

    The only dynamics are rotations: the state is a vector of unit-modulus
    complex numbers, advanced by input-conditioned angle kicks.  No chain, no
    quantization, no attention, no gating — every operation is differentiable
    and the state never leaves the torus.
    """

    def __init__(
        self,
        num_features: int,
        dim_angles: int = 48,
        kick: str = "mlp",
        kick_hidden: int = 64,
        use_amplitude: bool = False,
        use_diff_input: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if kick not in ("linear", "mlp"):
            raise ValueError("kick must be 'linear' or 'mlp'")
        self.num_features = num_features
        self.use_amplitude = use_amplitude
        self.use_diff_input = use_diff_input
        effective_features = num_features * 2 if use_diff_input else num_features
        # Input-conditioned angular kick (radians, unbounded, differentiable).
        if kick == "linear":
            self.kick_net: nn.Module = nn.Linear(effective_features, dim_angles)
        else:
            self.kick_net = nn.Sequential(
                nn.Linear(effective_features, kick_hidden),
                nn.GELU(),
                nn.Linear(kick_hidden, dim_angles),
            )
        # Base rotation initialized at identity (zero kick => h stays at 1).
        self.base_angle = nn.Parameter(torch.zeros(dim_angles))
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[-1] != self.num_features:
            raise ValueError("features must have shape (batch, sequence, num_features)")
        model_input = features
        if self.use_diff_input:
            model_input = torch.cat(
                [features, features.diff(dim=1, prepend=features[:, :1, :])], dim=-1
            )
        batch_size, seq_len, _ = model_input.shape
        h = torch.ones(batch_size, self.base_angle.shape[0], dtype=torch.cfloat,
                       device=features.device)
        for t in range(seq_len):
            kick = self.base_angle + self.kick_net(model_input[:, t])
            h = h * torch.polar(torch.ones_like(kick), kick)
            h = h / (h.abs() + 1e-8)  # floating-point guard, part of the math is exp(i R)
        hidden = torch.cat([h.real, h.imag], dim=-1)
        out = self.fusion(hidden)
        if self.use_amplitude:
            out = out + self.amp_proj(features[:, -1])
        return out


class PhaseRNNForecaster(nn.Module):
    """PhaseRotationEncoder with a linear head producing an H-step forecast."""

    def __init__(self, num_features: int, horizon: int, num_channels: int = 1, **kwargs):
        super().__init__()
        self.encoder = PhaseRotationEncoder(num_features, **kwargs)
        self.horizon = horizon
        self.num_channels = num_channels
        self.head = nn.Linear(self.encoder.hidden_dim, horizon * num_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(features)).view(
            features.shape[0], self.horizon, self.num_channels
        )


def parse_floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(v) for v in text.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("probe", "nonlinear"), default="probe")
    parser.add_argument("--kick", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--kick-hidden", type=int, default=64)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--n-train", type=int, default=0, help="override per task")
    parser.add_argument("--n-val", type=int, default=0)
    parser.add_argument("--n-test", type=int, default=0)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100,
                        help="probe task: budget per config; nonlinear task: max epochs")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lrs", type=str, default="", help="probe task: lr grid")
    parser.add_argument("--seeds", type=str, default="", help="probe task: seed list")
    parser.add_argument("--kernel", type=str, default="stochastic",
                        help="unused (kept for CLI symmetry with the other harnesses)")
    parser.add_argument("--states", type=int, default=24,
                        help="unused (kept for CLI symmetry with the other harnesses)")
    parser.add_argument("--no-quant", action="store_true",
                        help="unused: there is nothing to quantize in this parameterization")
    parser.add_argument("--amplitude", action="store_true")
    parser.add_argument("--diff-input", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="research/results_phase_rnn_probe.json")
    args = parser.parse_args()

    if args.task == "probe":
        n_train, n_val, n_test = args.n_train or 300, args.n_val or 100, args.n_test or 100
        # No early stopping: the budget is the diagnostic (mirrors overfit_budget).
        epochs, patience = args.epochs, args.epochs
    else:
        n_train, n_val, n_test = args.n_train or 2500, args.n_val or 600, args.n_test or 600
        epochs, patience = args.epochs, args.patience

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    length = args.seq + args.horizon

    if args.task == "probe":
        print(f"Overfit probe task (switching rotations, {n_train} sequences)")
        x_all, _ = simulate_switching_rotation(n_train + n_val + n_test, length, seed=7)
        regimes_last = None
    else:
        print(f"Nonlinear regime task (K=3, {n_train} sequences)")
        x_all, s_all = simulate_regime_switching_oscillator(
            n_train + n_val + n_test, length, num_regimes=3,
            noise_std=0.05, stickiness=0.97, seed=7,
        )
        regimes_last = s_all[n_train + n_val :, args.seq - 1].astype(np.int64)

    train_x = torch.from_numpy(x_all[:n_train, : args.seq])
    train_y = torch.from_numpy(x_all[:n_train, args.seq :])
    val_x = torch.from_numpy(x_all[n_train : n_train + n_val, : args.seq])
    val_y = torch.from_numpy(x_all[n_train : n_train + n_val, args.seq :])
    test_x = torch.from_numpy(x_all[n_train + n_val :, : args.seq])
    test_y = torch.from_numpy(x_all[n_train + n_val :, args.seq :])
    print(f"windows: train {tuple(train_x.shape)}, val {tuple(val_x.shape)}, test {tuple(test_x.shape)}")
    mean_val_mse = ((val_y - train_y.mean()) ** 2).mean().item()
    print(f"predict-train-mean reference: val mse {mean_val_mse:.4f}\n")

    def build() -> PhaseRNNForecaster:
        return PhaseRNNForecaster(
            train_x.shape[-1], args.horizon,
            num_channels=train_x.shape[-1],  # proper multi-channel head
            dim_angles=args.dim, kick=args.kick, kick_hidden=args.kick_hidden,
            use_amplitude=args.amplitude, use_diff_input=args.diff_input,
        )

    results: dict = {
        "config": {**{k: v for k, v in vars(args).items()},
                   "n_train": n_train, "n_val": n_val, "n_test": n_test},
        "mean_predictor_val_mse": round(mean_val_mse, 5),
    }

    if args.task == "probe":
        lrs = parse_floats(args.lrs) if args.lrs else [args.lr]
        seeds = parse_ints(args.seeds) if args.seeds else [args.seed]
        runs: dict = {}
        best = None
        for lr in lrs:
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = build()
                params = count_parameters(model)
                print(f"[phase_rnn({args.kick})] lr={lr:g} seed={seed} params={params:,}", flush=True)
                outcome = train_model(
                    model, train_x, train_y, val_x, val_y,
                    epochs=epochs, batch_size=args.batch, lr=lr,
                    patience=patience, seed=seed, log=False,
                )
                test_mse, test_mae = evaluate(model, test_x, test_y)
                train_mses = [h["train_mse"] for h in outcome["history"] if "train_mse" in h]
                entry = {
                    "lr": lr, "seed": seed, "params": params,
                    "epochs_run": len(train_mses),
                    "min_train_mse": round(min(train_mses), 5),
                    "final_train_mse": round(train_mses[-1], 5),
                    "best_val_mse": round(min(h["val_mse"] for h in outcome["history"] if "val_mse" in h), 5),
                    "test_mse": round(test_mse, 5),
                    "test_mae": round(test_mae, 5),
                    "train_seconds": outcome["history"][-1].get("train_seconds"),
                }
                key = f"lr{lr:g}_seed{seed}"
                runs[key] = entry
                print(
                    f"    min train {entry['min_train_mse']:.4f}  test {entry['test_mse']:.4f}"
                    f"  ({entry['epochs_run']} epochs, {entry['train_seconds']}s)", flush=True,
                )
                if best is None or entry["test_mse"] < best["test_mse"]:
                    best = {**entry, "config_key": key}
        runs["best"] = best
        results["runs"] = runs
        print(f"\nBEST: {best['config_key']}  min train {best['min_train_mse']:.4f}"
              f"  test {best['test_mse']:.4f}")
        print("reference (Round 3): MMT min train 0.496-0.509 / test 0.513-0.530;")
        print("MLP min train 0.029 / test 0.154; plateau = predict-the-mean ~0.50")
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = build()
        params = count_parameters(model)
        print(f"[phase_rnn({args.kick})] params={params:,}")
        outcome = train_model(
            model, train_x, train_y, val_x, val_y,
            epochs=epochs, batch_size=args.batch, lr=args.lr,
            patience=patience, seed=args.seed, log=True,
        )
        test_mse, test_mae = evaluate(model, test_x, test_y)
        phase_mse, _ = evaluate_naive(
            lambda xb: phase_readout_forecast(xb, args.horizon), test_x, test_y
        )
        results["phase_rnn"] = {
            "params": params,
            "best_val_mse": round(outcome["best_val_mse"], 5),
            "test_mse": round(test_mse, 5),
            "test_mae": round(test_mae, 5),
        }
        results["phase_readout_reference"] = {"test_mse": round(phase_mse, 5)}
        print(f"\n[phase_rnn({args.kick})] test mse={test_mse:.4f}")
        print("references (Round 3): MLP 0.3725 | MoE-linear 0.3957 | phase readout 0.5311"
              " | DLinear 0.6123 | GRU 0.8737 | MMT 0.8963")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()

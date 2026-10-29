"""Round 4: graft the toric regime machinery onto the differentiable rotation.

Round 3.5 established two facts: (a) the fully differentiable rotation
recurrence (``phase_rnn.py``) fits everything thrown at it (300-sequence
probe train MSE 0.033) but generalizes poorly on the nonlinear regime task
(0.693 vs MMT 0.388 / GRU 0.357); (b) MMT's machinery — discrete chain,
gating, attention — buys that generalization but rides on top of a
quantized/STE phase path that leaves a real optimization gap (probe train
0.22 vs 0.015 for a GRU).

Round 4 tests, as an ablation ladder, which piece of the machinery closes
the generalization gap when grafted onto the differentiable SO(2)
recurrence.  One shared base (reviewer constraint: rotation stays SO(2) —
the state lives on (S^1)^d and every transition is complex multiplication
``exp(i * angle)``; no quaternion parameterization anywhere):

    h_{t+1} = normalize( h_t * exp(i * (theta_base + f(x_t))) ),  h_0 = 1,

with f a two-layer MLP kick, then the same fusion + linear head + amplitude
bypass + diff inputs as the Round 3.5 reference.  The grafted pieces:

- ``chain``  : ToricMarkovChain (stochastic kernel, K states) driven by the
  raw (diff) features; its posterior adds per-state kick offsets
  ``state_theta[K, D]`` — the differentiable replacement of the toric
  state-selected kicks.
- ``gate``   : per-dimension input-conditioned geodesic slerp,
  ``h = normalize(g * h_rotated + (1 - g) * h_old)`` with
  ``g = sigmoid(Linear(x_t))`` — the toric gate made input-conditioned.
- ``attention``: per-dimension pull toward a learned context phase (the
  toric attention minus quantization).

Variants (--variant): ``rot`` (reference, no machinery), ``chain``,
``gate``, ``chain_gate``, ``full`` (= chain + gate + attention).  The plan
and stop-loss are pre-registered in ``research/README.md`` (Round 4
section) and were committed before the first run.

Usage (repository root)::

    # pre-flight: overfit probe (must reach min train MSE < 0.10)
    python -m research.phase_hybrid --task probe --variant full \
        --lrs 1e-3,3e-3,1e-2 --seeds 0 --epochs 100 \
        --out research/results_round4_probe_full.json

    # primary gate: nonlinear regime task, fixed protocol
    python -m research.phase_hybrid --task nonlinear --variant chain \
        --epochs 30 --patience 5 --diff-input --amplitude --seed 0 \
        --out research/results_round4_nonlinear_chain.json

    # secondary gate: home turf (plain features, as the reference runs)
    python -m research.phase_hybrid --task turf --variant full \
        --epochs 6 --patience 2 \
        --out research/results_round4_turf_easy_full.json
    python -m research.phase_hybrid --task turf --variant full \
        --speed-scale 2.0 --seq 96 --horizon 24 --epochs 25 --patience 5 \
        --out research/results_round4_turf_hard_full.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .nonlinear_torus import simulate_regime_switching_oscillator
from .synthetic_torus import simulate_switching_rotation
from .toric_core import ToricMarkovChain, normalize, unit
from .train_utils import count_parameters, evaluate, train_model

VARIANTS = {
    "rot": (False, False, False),
    "chain": (True, False, False),
    "gate": (False, True, False),
    "chain_gate": (True, True, False),
    "full": (True, True, True),
}


class PhaseHybridEncoder(nn.Module):
    """Differentiable SO(2) rotation recurrence + optionally grafted machinery.

    Base dynamics (always): unit-modulus complex state advanced by
    input-conditioned angle kicks — pure rotations, state on the torus.
    The chain / gate / attention pieces are switched on per variant; every
    added operation is differentiable end to end.
    """

    def __init__(
        self,
        num_features: int,
        dim_angles: int = 48,
        kick_hidden: int = 64,
        num_states: int = 8,
        use_chain: bool = False,
        use_gate: bool = False,
        use_attention: bool = False,
        use_amplitude: bool = False,
        use_diff_input: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.use_chain = use_chain
        self.use_gate = use_gate
        self.use_attention = use_attention
        self.use_amplitude = use_amplitude
        self.use_diff_input = use_diff_input
        effective_features = num_features * 2 if use_diff_input else num_features
        # Input-conditioned angular kick (radians, unbounded, differentiable).
        self.kick_net = nn.Sequential(
            nn.Linear(effective_features, kick_hidden),
            nn.GELU(),
            nn.Linear(kick_hidden, dim_angles),
        )
        # Base rotation initialized at identity (zero kick => h stays at 1).
        self.base_angle = nn.Parameter(torch.zeros(dim_angles))
        if use_chain:
            self.chain = ToricMarkovChain(effective_features, num_states, "stochastic")
            # Per-state kick offsets: the differentiable replacement of the
            # toric state-selected kicks (theta init N(0, 0.5^2) there).
            self.state_theta = nn.Parameter(
                torch.randn(num_states, dim_angles) * 0.5
            )
        if use_gate:
            # Per-dimension input-conditioned geodesic gate.
            self.gate_net = nn.Linear(effective_features, dim_angles)
        if use_attention:
            self.ctx_angle = nn.Parameter(torch.randn(dim_angles) * 0.1)
            self.attn_scale = nn.Parameter(torch.tensor(1.0))
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
        h = torch.ones(
            batch_size, self.base_angle.shape[0], dtype=torch.cfloat,
            device=features.device,
        )
        if self.use_chain:
            state_probs = features.new_zeros(batch_size, self.chain.num_states)
            state_probs[:, 0] = 1.0
        for t in range(seq_len):
            x_t = model_input[:, t]
            kick = self.base_angle + self.kick_net(x_t)
            if self.use_chain:
                state_probs = self.chain(x_t, state_probs)
                kick = kick + state_probs @ self.state_theta
            h_rotated = h * torch.polar(torch.ones_like(kick), kick)
            h_rotated = normalize(h_rotated)
            if self.use_gate:
                g = torch.sigmoid(self.gate_net(x_t))  # (B, D)
                h = normalize(g * h_rotated + (1.0 - g) * h)
            else:
                h = h_rotated
            if self.use_attention:
                context = unit(self.ctx_angle)  # (D,)
                score = (h * context.conj()).real  # (B, D) in [-1, 1]
                weight = torch.sigmoid(score * self.attn_scale)
                h = normalize(h + weight * context)
        hidden = torch.cat([h.real, h.imag], dim=-1)
        out = self.fusion(hidden)
        if self.use_amplitude:
            out = out + self.amp_proj(features[:, -1])
        return out


class PhaseHybridForecaster(nn.Module):
    """PhaseHybridEncoder with a linear head producing an H-step forecast."""

    def __init__(self, num_features: int, horizon: int, num_channels: int = 1, **kwargs):
        super().__init__()
        self.encoder = PhaseHybridEncoder(num_features, **kwargs)
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
    parser.add_argument("--task", choices=("probe", "nonlinear", "turf"), default="probe")
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="full")
    parser.add_argument("--kick-hidden", type=int, default=64)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--states", type=int, default=8, help="chain states K (when grafted)")
    parser.add_argument("--n-train", type=int, default=0, help="override per task")
    parser.add_argument("--n-val", type=int, default=0)
    parser.add_argument("--n-test", type=int, default=0)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=100,
                        help="probe task: budget per config; else max epochs")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lrs", type=str, default="", help="probe task: lr grid")
    parser.add_argument("--seeds", type=str, default="", help="probe task: seed list")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="turf task")
    parser.add_argument("--amplitude", action="store_true")
    parser.add_argument("--diff-input", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="research/results_phase_hybrid.json")
    args = parser.parse_args()

    use_chain, use_gate, use_attention = VARIANTS[args.variant]

    if args.task == "probe":
        n_train, n_val, n_test = args.n_train or 300, args.n_val or 100, args.n_test or 100
        epochs, patience = args.epochs, args.epochs  # no early stopping: budget IS the diagnostic
    else:
        n_train, n_val, n_test = args.n_train or 2500, args.n_val or 600, args.n_test or 600
        epochs, patience = args.epochs, args.patience

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    length = args.seq + args.horizon

    if args.task == "probe":
        print(f"Overfit probe task (switching rotations, {n_train} sequences)")
        x_all, _ = simulate_switching_rotation(n_train + n_val + n_test, length, seed=7)
    elif args.task == "turf":
        print(f"Home-turf task (4 regimes, speed x{args.speed_scale:g}, {n_train} sequences)")
        x_all, _ = simulate_switching_rotation(
            n_train + n_val + n_test, length, speed_scale=args.speed_scale, seed=7
        )
    else:
        print(f"Nonlinear regime task (K=3, {n_train} sequences)")
        x_all, _ = simulate_regime_switching_oscillator(
            n_train + n_val + n_test, length, num_regimes=3,
            noise_std=0.05, stickiness=0.97, seed=7,
        )

    train_x = torch.from_numpy(x_all[:n_train, : args.seq])
    train_y = torch.from_numpy(x_all[:n_train, args.seq :])
    val_x = torch.from_numpy(x_all[n_train : n_train + n_val, : args.seq])
    val_y = torch.from_numpy(x_all[n_train : n_train + n_val, args.seq :])
    test_x = torch.from_numpy(x_all[n_train + n_val :, : args.seq])
    test_y = torch.from_numpy(x_all[n_train + n_val :, args.seq :])
    print(f"windows: train {tuple(train_x.shape)}, val {tuple(val_x.shape)}, test {tuple(test_x.shape)}")
    mean_val_mse = ((val_y - train_y.mean()) ** 2).mean().item()
    print(f"predict-train-mean reference: val mse {mean_val_mse:.4f}\n")

    def build() -> PhaseHybridForecaster:
        return PhaseHybridForecaster(
            train_x.shape[-1], args.horizon,
            num_channels=train_x.shape[-1],  # proper multi-channel head
            dim_angles=args.dim, kick_hidden=args.kick_hidden,
            num_states=args.states, use_chain=use_chain, use_gate=use_gate,
            use_attention=use_attention, use_amplitude=args.amplitude,
            use_diff_input=args.diff_input,
        )

    results: dict = {
        "config": {**vars(args), "n_train": n_train, "n_val": n_val, "n_test": n_test,
                   "use_chain": use_chain, "use_gate": use_gate,
                   "use_attention": use_attention},
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
                print(f"[{args.variant}] lr={lr:g} seed={seed} params={params:,}", flush=True)
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
        print("pre-flight gate: min train MSE < 0.10 required to launch Round 4")
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        model = build()
        params = count_parameters(model)
        print(f"[{args.variant}] params={params:,}")
        outcome = train_model(
            model, train_x, train_y, val_x, val_y,
            epochs=epochs, batch_size=args.batch, lr=args.lr,
            patience=patience, seed=args.seed, log=True,
        )
        test_mse, test_mae = evaluate(model, test_x, test_y)
        results[args.variant] = {
            "params": params,
            "best_val_mse": round(outcome["best_val_mse"], 5),
            "test_mse": round(test_mse, 5),
            "test_mae": round(test_mae, 5),
        }
        if args.task == "nonlinear":
            print(f"\n[{args.variant}] test mse={test_mse:.4f}"
                  f"  (targets: GRU 0.3567 | old MMT 0.3878 | stop-loss > 0.388)")
        else:
            print(f"\n[{args.variant}] test mse={test_mse:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()

"""Roadmap item 1: a task where linear AR provably fails — regime-conditional
nonlinear structure with identifiable regimes.

Generative process
------------------
A hidden discrete regime s_t follows a sticky Markov chain.  Each regime k has
its own amplitude a_k and angular velocity w_k:

    theta_{t+1} = wrap(theta_t + w_{s_t}),
    x_t = a_{s_t} * (cos theta_t, sin theta_t) + eps,   eps ~ N(0, sigma^2 I_2).

Both observation channels are the full complex phase readout, deliberately
scaled by the regime amplitude — the regime is identifiable in principle from
the window (envelope gives a, rotation rate gives w).

Why a single linear map is mis-specified here
---------------------------------------------
For a FIXED angular velocity the forecast is a linear function of the window
(a fixed-frequency sinusoid satisfies a 2-term linear recurrence — the reason
DLinear swept every earlier round).  With regime-dependent (a_k, w_k) the
optimal predictor is the regime posterior mixture

    E[y | window] = sum_k P(s = k | window) * a_k * R_{k w_k h} z_T,

where R is the rotation operator and z_T the current (amp-scaled) phase point.
No single linear operator applied to the window reproduces this: the mixture
weights depend on the window's envelope and rotation-rate statistics, which
are quadratic / angular (nonlinear) functions of the samples, and alternating
regimes switch the effective recurrence coefficients over time.  So the task
separates models with an explicit discrete-state mechanism from pure linear
maps — the test the toric architecture has to win to justify itself.

Models compared
---------------
- dlinear      : per-channel linear AR (expected to plateau at the
                 regime-averaged predictor)
- mlp          : generic nonlinear learner over the flattened window (control:
                 is the failure architecture-specific or just gradient
                 difficulty?)
- moe_linear   : linear experts + nonlinear gate — the minimal "identify the
                 regime, then apply a linear predictor" reference
- gru          : generic recurrent baseline (with optional diff inputs)
- mmt          : the toric forecaster (the contender)
- persistence  : naive reference
- phase        : generator-matched analytic readout (no learning): estimates
                 phase, envelope and rotation rate from the last window steps
                 and propagates them — proves the information is in the window
                 and quantifies how far each learned model is from it.

Usage (repository root)::

    python -m research.nonlinear_torus --models dlinear,mlp,moe_linear,gru,mmt,naive,phase \
        --epochs 30 --patience 5 --diff-input --amplitude --seed 0 \
        --out research/results_nonlinear.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .baselines import (
    DLinearForecaster,
    MLPForecaster,
    MoELinearForecaster,
    RecurrentForecaster,
    persistence_forecast,
)
from .toric_core import ToricForecaster, markov_agreement
from .train_utils import evaluate, evaluate_naive, count_parameters, train_model

# Regime parameters: deliberately well-separated so regimes are identifiable
# in principle — the question is whether each model can EXTRACT them.
REGIME_AMPS = (0.6, 1.4, 2.4)  # envelope per regime
REGIME_OMEGAS = (0.05, 0.20, 0.45)  # rad/step per regime


def wrap(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def simulate_regime_switching_oscillator(
    n_sequences: int,
    length: int,
    num_regimes: int = 3,
    noise_std: float = 0.05,
    stickiness: float = 0.97,
    speed_scale: float = 1.0,
    amp_scale: float = 1.0,
    seed: int = 7,
    return_theta: bool = False,
) -> tuple[np.ndarray, ...]:
    """Simulate the regime-switching oscillator; returns x (N, len, 2), s (N, len).

    With ``return_theta=True`` a third array (N, len) of the true underlying
    phases theta_t (the exact values inside the observation readout) is
    appended — used by the state-probe experiment as the regression target.

    ``stickiness`` trades off two failure modes of the task design.  At 0.9
    (mean sojourn 10 steps) windows contain ~6 regime switches and even the
    12-step forecast span usually crosses a switch, so the Bayes predictor is
    a mixture over future regime paths — the task becomes hard for every
    model including the analytic readout.  At 0.97 (mean sojourn ~33) windows
    are mostly single-regime with a couple of switches, the forecast span
    usually stays in-regime, and "identify the regime, then apply its
    dynamics" is well defined — the cleanest test of roadmap item 1.
    """
    if not 1 <= num_regimes <= len(REGIME_AMPS):
        raise ValueError(f"num_regimes must be in 1..{len(REGIME_AMPS)}")
    generator = torch.Generator().manual_seed(seed)
    amps = torch.tensor(REGIME_AMPS[:num_regimes]) * amp_scale
    omegas = torch.tensor(REGIME_OMEGAS[:num_regimes]) * speed_scale

    transition = torch.full((num_regimes, num_regimes), (1 - stickiness) / (num_regimes - 1))
    transition.fill_diagonal_(stickiness)

    theta = (torch.rand(n_sequences, generator=generator) * 2 - 1) * math.pi
    states = torch.randint(num_regimes, (n_sequences,), generator=generator)
    obs = torch.empty(n_sequences, length, 2)
    states_history = torch.empty(n_sequences, length, dtype=torch.long)
    theta_history = torch.empty(n_sequences, length)

    for t in range(length):
        states_history[:, t] = states
        theta_history[:, t] = theta
        obs[:, t, 0] = amps[states] * torch.cos(theta) + noise_std * torch.randn(
            n_sequences, generator=generator
        )
        obs[:, t, 1] = amps[states] * torch.sin(theta) + noise_std * torch.randn(
            n_sequences, generator=generator
        )
        rows = transition[states]
        states = torch.multinomial(rows, 1, generator=generator).squeeze(-1)
        theta = wrap(theta + omegas[states])

    if return_theta:
        return (
            obs.numpy().astype(np.float32),
            states_history.numpy(),
            theta_history.numpy(),
        )
    return obs.numpy().astype(np.float32), states_history.numpy()


def phase_readout_forecast(x: torch.Tensor, horizon: int, est_len: int = 8) -> torch.Tensor:
    """Generator-matched analytic readout (no learning).

    Estimates the current phase, envelope and rotation rate from the last
    ``est_len`` window steps and propagates the oscillator.  ``est_len`` is a
    deliberate compromise: a longer span lowers estimation variance but mixes
    in pre-switch dynamics, a shorter one is noisier — 8 steps keeps the
    estimate inside the current regime for the default stickiness while
    averaging the observation noise.  The readout's own MSE is dominated by
    rotation-rate estimation noise on low-amplitude regimes, so treat it as a
    crude structure-aware reference ("the information is in the window"), not
    as a ceiling.
    """
    amplitude = x.norm(dim=-1)  # envelope readout, (B, T)
    phi = torch.atan2(x[..., 1], x[..., 0])  # (B, T)
    dphi = wrap(phi[:, 1:] - phi[:, :-1])  # per-step rotation increments
    omega = dphi[:, -est_len:].mean(dim=1)  # (B,) recent rotation rate
    amp = amplitude[:, -est_len:].mean(dim=1)  # (B,) recent envelope
    theta_last = phi[:, -1]
    steps = torch.arange(1, horizon + 1, dtype=x.dtype)
    angles = theta_last[:, None] + steps[None, :] * omega[:, None]
    return torch.stack(
        [amp[:, None] * torch.cos(angles), amp[:, None] * torch.sin(angles)], dim=-1
    )


def build_model(name: str, num_features: int, seq_len: int, horizon: int, args) -> nn.Module:
    # num_channels = data width: the 1-channel default broadcasts a single
    # shared forecast across channels under mse_loss (the Round 1-3 bug).
    toric_kwargs = dict(
        dim_angles=args.dim,
        num_levels=2,
        num_states=args.states,
        num_layers=1,
        kernel=args.kernel,
        use_attention=True,
        use_amplitude=True,  # amplitude carries the regime signature here
        use_diff_input=args.diff_input,
        quantize_positions=not args.no_quant,
        n_bits=16 if args.no_quant else 8,
    )
    if name == "mmt":
        return ToricForecaster(num_features, horizon, num_channels=num_features,
                               **toric_kwargs)
    if name == "gru":
        return RecurrentForecaster(
            num_features, horizon, num_channels=num_features,
            kind="gru", hidden_dim=args.hidden, use_diff_input=args.diff_input,
        )
    if name == "mlp":
        return MLPForecaster(seq_len, horizon, num_channels=num_features,
                             hidden_dim=args.hidden * 2)
    if name == "moe_linear":
        return MoELinearForecaster(seq_len, horizon, num_channels=num_features,
                                   num_experts=args.regimes)
    if name == "dlinear":
        return DLinearForecaster(seq_len, horizon, num_channels=num_features)
    raise ValueError(f"unknown model {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=2500)
    parser.add_argument("--n-val", type=int, default=600)
    parser.add_argument("--n-test", type=int, default=600)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--regimes", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--stickiness", type=float, default=0.97,
                        help="Markov self-transition probability of the regimes")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dim", type=int, default=48, help="toric angle dimension")
    parser.add_argument("--states", type=int, default=24, help="toric states")
    parser.add_argument("--hidden", type=int, default=64, help="neural baseline width")
    parser.add_argument("--kernel", type=str, default="stochastic", choices=("stochastic", "free"))
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--amplitude", action="store_true",
                        help="enable the toric encoder amplitude bypass")
    parser.add_argument("--diff-input", action="store_true",
                        help="append first differences to mmt/gru inputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", type=str,
                        default="dlinear,mlp,moe_linear,gru,mmt,naive,phase")
    parser.add_argument("--out", type=str, default="research/results_nonlinear.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    length = args.seq + args.horizon
    print(
        f"Simulating regime-switching oscillator "
        f"(K={args.regimes}, noise={args.noise}, length={length})..."
    )
    x_all, s_all = simulate_regime_switching_oscillator(
        args.n_train + args.n_val + args.n_test, length,
        num_regimes=args.regimes, noise_std=args.noise,
        stickiness=args.stickiness, seed=7,
    )
    n_train, n_val = args.n_train, args.n_val
    train_x = torch.from_numpy(x_all[:n_train, : args.seq])
    train_y = torch.from_numpy(x_all[:n_train, args.seq :])
    val_x = torch.from_numpy(x_all[n_train : n_train + n_val, : args.seq])
    val_y = torch.from_numpy(x_all[n_train : n_train + n_val, args.seq :])
    test_x = torch.from_numpy(
        x_all[n_train + n_val : n_train + n_val + args.n_test, : args.seq]
    )
    test_y = torch.from_numpy(
        x_all[n_train + n_val : n_train + n_val + args.n_test, args.seq :]
    )
    true_last_state = torch.from_numpy(
        s_all[n_train + n_val : n_train + n_val + args.n_test, args.seq - 1].astype(np.int64)
    )
    print(f"windows: train {tuple(train_x.shape)}, val {tuple(val_x.shape)}, test {tuple(test_x.shape)}\n")

    results = {"config": vars(args), "models": {}}
    requested = [m.strip() for m in args.models.split(",") if m.strip()]

    for name in requested:
        if name == "persistence" or name == "naive":
            mse, mae = evaluate_naive(
                lambda xb: persistence_forecast(xb, args.horizon), test_x, test_y
            )
            results["models"]["persistence"] = {
                "params": 0, "test_mse": round(mse, 5), "test_mae": round(mae, 5),
            }
            print(f"[persistence] test mse={mse:.4f} mae={mae:.4f}")
            continue
        if name == "phase":
            mse, mae = evaluate_naive(
                lambda xb: phase_readout_forecast(xb, args.horizon), test_x, test_y
            )
            results["models"]["phase_readout"] = {
                "params": 0, "test_mse": round(mse, 5), "test_mae": round(mae, 5),
            }
            print(f"[phase_readout] test mse={mse:.4f} mae={mae:.4f}")
            continue
        model = build_model(name, train_x.shape[-1], args.seq, args.horizon, args)
        params = count_parameters(model)
        print(f"[{name}] params={params:,}")
        outcome = train_model(
            model, train_x, train_y, val_x, val_y,
            epochs=args.epochs, batch_size=args.batch, lr=args.lr,
            patience=args.patience, seed=args.seed,
        )
        test_mse, test_mae = evaluate(outcome["model"], test_x, test_y)
        entry = {
            "params": params,
            "best_val_mse": round(outcome["best_val_mse"], 5),
            "test_mse": round(test_mse, 5),
            "test_mae": round(test_mae, 5),
        }
        if name == "mmt" and args.states == args.regimes:
            # Interpretability check: with exactly as many chain states as true
            # regimes, does the unsupervised chain recover them (up to labels)?
            _, probs = outcome["model"](test_x[:512], return_state=True)
            entry["state_agreement_vs_true_regime"] = round(
                markov_agreement(true_last_state[:512], probs, args.regimes), 4
            )
        results["models"][name] = entry
        print(f"[{name}] test mse={test_mse:.4f} mae={test_mae:.4f}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== Summary (MSE, lower is better) ===")
    ranked = sorted(results["models"].items(), key=lambda kv: kv[1]["test_mse"])
    for name, entry in ranked:
        extra = f"  params={entry['params']:,}" if entry.get("params") else ""
        agree = entry.get("state_agreement_vs_true_regime")
        suffix = f"  regime-agreement={agree}" if agree is not None else ""
        print(f"  {name:<14} mse={entry['test_mse']:.4f}  mae={entry['test_mae']:.4f}{extra}{suffix}")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()

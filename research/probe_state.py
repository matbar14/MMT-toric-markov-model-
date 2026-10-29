"""Round-4 pre-control 1: linear probes on the frozen MMT state (nonlinear torus).

Round 3 established that the trained MMT loses to its own analytic readout on
the regime-conditional task (0.896 vs 0.531) while its chain posterior still
agrees with the true regimes at 0.502 (chance 0.33).  Two hypotheses survive:

  (a) the state contains the information but the forecast head cannot decode
      it ("encodes, does not decode") — then making the readout/rotation
      differentiable (Round 4) is the right candidate;
  (b) the state itself is dead — then Round 4 is pointless.

This experiment separates (a) from (b) with the cheapest possible tool: LINEAR
probes on the frozen encoder state of the trained MMT.  No probe can make the
model better; it only measures what the state already contains.

Probes (all trained on the train split, reported on test):
  regime   multinomial logistic regression on {chain posterior p_T, fusion
           output (what the head sees), raw phase state, raw window} scored
           by best-permutation agreement vs the true regime; the model's own
           argmax(p_T) agreement is reported next to it (Round 3: 0.502).
  phase    closed-form ridge regression on (cos theta_T, sin theta_T) with
           theta_T the TRUE underlying phase at the last window step; scored
           by mean wrapped angle error and per-component R^2.
  forecast closed-form ridge from frozen features to the full 12-step
           forecast.  If a LINEAR readout on the frozen state beats the
           trained MMT head (test MSE 0.8963), the barrier is in the head /
           output path, not in the state — the direct green light for Round 4.
           The window probe doubles as a closed-form DLinear reference.

Usage (repository root)::

    python -m research.probe_state --states 24 --diff-input --amplitude \
        --epochs 30 --patience 5 --seed 0 \
        --out research/results_probe_states24.json
    python -m research.probe_state --states 3 --diff-input --amplitude \
        --epochs 30 --patience 5 --seed 0 \
        --out research/results_probe_states3.json
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
from .toric_core import ToricForecaster, markov_agreement
from .train_utils import count_parameters, evaluate, evaluate_naive, train_model

FEATURE_SETS = ("chain_probs", "head_input", "phase_state", "window", "state_plus_probs")


def extract_features(
    model: ToricForecaster, x: torch.Tensor, use_amplitude: bool, batch: int = 256
) -> dict[str, np.ndarray]:
    """Frozen-feature extraction from the trained encoder.

    Returns per-window: ``phase_state`` (the concatenation of real/imag phase
    state BEFORE fusion — the state proper), ``head_input`` (exactly what the
    forecast head consumes: fusion output plus the amplitude bypass), and
    ``chain_probs`` (the HMM-filter belief at the last input step).
    """
    captured: dict[str, torch.Tensor] = {}
    encoder = model.encoder

    def pre_hook(_module, args):
        captured["phase_state"] = args[0].detach()

    def post_hook(_module, _args, output):
        captured["fusion_out"] = output.detach()

    h1 = encoder.fusion.register_forward_pre_hook(pre_hook)
    h2 = encoder.fusion.register_forward_hook(post_hook)
    model.eval()
    chunks: dict[str, list] = {k: [] for k in ("phase_state", "head_input", "chain_probs")}
    with torch.no_grad():
        for start in range(0, x.shape[0], batch):
            xb = x[start : start + batch]
            _out, probs = encoder(xb, return_state=True)
            # Rebuild the exact head input from the CAPTURED fusion output:
            # encode() adds the amplitude bypass after fusion, so adding it to
            # the returned value would double it.
            head_input = captured["fusion_out"]
            if use_amplitude:
                head_input = head_input + encoder.amp_proj(xb[:, -1])
            chunks["phase_state"].append(captured["phase_state"].cpu())
            chunks["head_input"].append(head_input.cpu())
            chunks["chain_probs"].append(probs.cpu())
    h1.remove()
    h2.remove()

    features = {k: torch.cat(v).numpy() for k, v in chunks.items()}
    features["window"] = x.reshape(x.shape[0], -1).numpy()
    features["state_plus_probs"] = np.concatenate(
        [features["head_input"], features["chain_probs"]], axis=-1
    )
    # Sanity: the reconstructed head_input must equal the model's own head
    # input.  Reference forward uses the same batch size as extraction —
    # GEMM tiling makes float32 results batch-size dependent at the ~1e-3
    # level for trained weights, which is not a wiring error.
    with torch.no_grad():
        own = model.encoder(x[:batch], return_state=False)
    err = float(np.abs(features["head_input"][:batch] - own.numpy()).max())
    if err > 1e-4:
        raise RuntimeError(f"head_input reconstruction mismatch ({err:.2e})")
    return features


def ridge_fit(
    X: np.ndarray, Y: np.ndarray, lam_rel: float = 1e-3
) -> tuple[np.ndarray, float]:
    """Closed-form ridge; returns (beta, intercept) with lambda relative to the
    mean diagonal of X^T X (scale-free regularization)."""
    X64 = X.astype(np.float64)
    Y64 = Y.astype(np.float64)
    XtX = X64.T @ X64
    lam = lam_rel * float(np.mean(np.diag(XtX))) + 1e-12
    Xty = X64.T @ Y64
    beta = np.linalg.solve(XtX + lam * np.eye(XtX.shape[0]), Xty)
    resid = Y64 - X64 @ beta
    return beta, float((resid**2).mean())


def ridge_predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return X.astype(np.float64) @ beta


def linear_probe_forecast(
    split: dict[str, np.ndarray], y: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Ridge from frozen features to the full forecast, per feature set."""
    y_train = y["train"].reshape(y["train"].shape[0], -1).numpy()
    y_test = y["test"].reshape(y["test"].shape[0], -1).numpy()
    out = {}
    for name in ("phase_state", "head_input", "state_plus_probs", "window"):
        beta, train_mse = ridge_fit(split["train"][name], y_train)
        pred_test = ridge_predict(split["test"][name], beta)
        test_mse = float(((pred_test - y_test) ** 2).mean())
        out[name] = {
            "train_mse": round(train_mse, 5),
            "test_mse": round(test_mse, 5),
        }
    return out


def phase_probe(
    split: dict[str, np.ndarray], theta_last: dict[str, np.ndarray]
) -> dict[str, dict]:
    """Ridge on (cos theta_T, sin theta_T); scored by wrapped angle error + R^2."""
    target_train = np.stack(
        [np.cos(theta_last["train"]), np.sin(theta_last["train"])], axis=-1
    )
    target_test = np.stack(
        [np.cos(theta_last["test"]), np.sin(theta_last["test"])], axis=-1
    )
    out = {}
    for name in ("phase_state", "head_input", "window"):
        beta, _ = ridge_fit(split["train"][name], target_train)
        pred = ridge_predict(split["test"][name], beta)
        err_vec = pred - target_test
        mse = float((err_vec**2).mean())
        cos_sin_r2 = 1 - mse / float((target_test**2).mean())
        est_angle = np.arctan2(pred[:, 1], pred[:, 0])
        d = est_angle - theta_last["test"]
        d = (d + math.pi) % (2 * math.pi) - math.pi
        out[name] = {
            "angle_err_rad": round(float(np.abs(d).mean()), 4),
            "cos_sin_r2": round(float(cos_sin_r2), 4),
        }
    return out


def train_logistic(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, steps: int = 600
) -> np.ndarray:
    """Multinomial logistic probe (full-batch Adam); returns test probabilities."""
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    Xtr = torch.from_numpy((X_train - mean) / std).float()
    Xte = torch.from_numpy((X_test - mean) / std).float()
    ytr = torch.from_numpy(y_train).long()
    n, d = Xtr.shape
    k = int(ytr.max().item()) + 1
    W = torch.zeros(d, k, requires_grad=True)
    b = torch.zeros(k, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(Xtr @ W + b, ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        probs = torch.softmax(Xte @ W + b, dim=-1)
    return probs.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=2500)
    parser.add_argument("--n-val", type=int, default=600)
    parser.add_argument("--n-test", type=int, default=600)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--regimes", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--stickiness", type=float, default=0.97)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--states", type=int, default=24)
    parser.add_argument("--kernel", type=str, default="stochastic")
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--amplitude", action="store_true")
    parser.add_argument("--diff-input", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="research/results_probe_states24.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    length = args.seq + args.horizon
    print("Simulating regime-switching oscillator (with theta history)...")
    x_all, s_all, theta_all = simulate_regime_switching_oscillator(
        args.n_train + args.n_val + args.n_test, length,
        num_regimes=args.regimes, noise_std=args.noise,
        stickiness=args.stickiness, seed=7, return_theta=True,
    )
    n_tr, n_va = args.n_train, args.n_val
    sl = lambda a, s: slice(s, s + a)  # noqa: E731
    x = {
        "train": torch.from_numpy(x_all[sl(n_tr, 0), : args.seq]),
        "val": torch.from_numpy(x_all[sl(n_va, n_tr), : args.seq]),
        "test": torch.from_numpy(x_all[sl(args.n_test, n_tr + n_va), : args.seq]),
    }
    y = {
        "train": torch.from_numpy(x_all[sl(n_tr, 0), args.seq :]),
        "val": torch.from_numpy(x_all[sl(n_va, n_tr), args.seq :]),
        "test": torch.from_numpy(x_all[sl(args.n_test, n_tr + n_va), args.seq :]),
    }
    regime_last = {
        "train": s_all[sl(n_tr, 0), args.seq - 1].astype(np.int64),
        "test": s_all[sl(args.n_test, n_tr + n_va), args.seq - 1].astype(np.int64),
    }
    theta_last = {
        "train": theta_all[sl(n_tr, 0), args.seq - 1],
        "test": theta_all[sl(args.n_test, n_tr + n_va), args.seq - 1],
    }
    print(f"windows: {', '.join(f'{k} {tuple(v.shape)}' for k, v in x.items())}\n")

    # ---- 1. train MMT with the exact Round-3 protocol ----
    model = ToricForecaster(
        x["train"].shape[-1], args.horizon,
        num_channels=x["train"].shape[-1],  # proper multi-channel head
        dim_angles=args.dim, num_levels=2, num_states=args.states, num_layers=1,
        kernel=args.kernel, use_attention=True, use_amplitude=args.amplitude,
        use_diff_input=args.diff_input, quantize_positions=not args.no_quant,
        n_bits=16 if args.no_quant else 8,
    )
    print(f"[mmt] params={count_parameters(model):,}")
    outcome = train_model(
        model, x["train"], y["train"], x["val"], y["val"],
        epochs=args.epochs, batch_size=args.batch, lr=args.lr,
        patience=args.patience, seed=args.seed, log=True,
    )
    head_mse, head_mae = evaluate(outcome["model"], x["test"], y["test"])
    print(f"[mmt head] test mse={head_mse:.4f} mae={head_mae:.4f}")

    phase_mse, _ = evaluate_naive(
        lambda xb: phase_readout_forecast(xb, args.horizon), x["test"], y["test"]
    )
    print(f"[phase readout reference] test mse={phase_mse:.4f}")

    # ---- 2. frozen features ----
    print("\nextracting frozen features (train/val/test)...")
    feats = {
        split: extract_features(model, x[split], args.amplitude)
        for split in ("train", "val", "test")
    }

    results = {
        "config": vars(args),
        "mmt_head": {
            "test_mse": round(head_mse, 5),
            "test_mae": round(head_mae, 5),
            "best_val_mse": round(outcome["best_val_mse"], 5),
        },
        "phase_readout_reference": {"test_mse": round(phase_mse, 5)},
    }

    # ---- 3a. regime probes (only meaningful with states == regimes) ----
    if args.states == args.regimes:
        print("\n[regime probes]")
        own_probs_test = feats["test"]["chain_probs"]
        own_agree = markov_agreement(
            torch.from_numpy(regime_last["test"]),
            torch.from_numpy(own_probs_test), args.regimes,
        )
        regime_report = {
            "model_own_argmax": round(own_agree, 4),
            "chance": round(1.0 / args.regimes, 4),
        }
        for name in ("chain_probs", "head_input", "phase_state", "window"):
            probs = train_logistic(
                feats["train"][name], regime_last["train"], feats["test"][name]
            )
            agree = markov_agreement(
                torch.from_numpy(regime_last["test"]), torch.from_numpy(probs),
                args.regimes,
            )
            regime_report[f"logistic_{name}"] = round(agree, 4)
            print(f"  logistic[{name}] agreement={agree:.4f}")
        print(f"  model own argmax agreement={own_agree:.4f} (chance {1/args.regimes:.2f})")
        results["regime_probe"] = regime_report

    # ---- 3b. phase probes ----
    print("\n[phase probes] target: true theta at the last window step")
    results["phase_probe"] = phase_probe(feats, theta_last)
    for name, rep in results["phase_probe"].items():
        print(f"  ridge[{name}] angle_err={rep['angle_err_rad']:.3f} rad"
              f"  cos_sin_R2={rep['cos_sin_r2']:.4f}")

    # ---- 3c. linear readout forecasters on frozen features ----
    print("\n[linear readout forecasters] ridge on frozen features -> 12x2 forecast")
    results["linear_readout_forecast"] = linear_probe_forecast(feats, y)
    for name, rep in results["linear_readout_forecast"].items():
        print(f"  ridge[{name}] train {rep['train_mse']:.4f}  test {rep['test_mse']:.4f}")
    print(f"\nreferences: mmt head {head_mse:.4f} | phase readout {phase_mse:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()

"""Roadmap item 3: convergence race with a real budget — lr sweep x seeds x
long-horizon training on the 300-sequence overfit probe.

Background
----------
Round 1's optimization probe showed that neither the toric model nor the GRU
can overfit even 300 sequences: both plateau at "predict the unconditional
mean" (train MSE ~0.49) within 50 epochs at lr 3e-3, while DLinear fits the
same data to ~0.10.  The honest open question is whether that plateau is a
true optimization barrier or simply an insufficient budget.  This harness
answers it with the cheapest possible design that separates the two
hypotheses: keep the tiny 300-sequence task (so one epoch is fast) and scale
epochs (200), patience (50) and the learning rate (sweep) with several seeds.

Reading the results
-------------------
- If NO learning-rate/seed combination drives the train MSE materially below
  the plateau, the optimization-failure claim is established at a real budget.
- If some configuration starts fitting, the plateau is a recipe problem and
  the winning configuration tells the later experiments what to use.
- ``dlinear`` is the calibration reference (it must still fit), and ``mlp``
  is the control that separates "any nonlinear model fails on small data"
  from "recurrent/phase dynamics specifically fail".

Note on hardware: this run is CPU-only (no GPU in the environment), which
limits the absolute budget; the design keeps the task tiny so that 200-epoch
runs stay minutes-long.  The harness is budget-generic — pass --epochs/--lrs/
--seeds to scale it up on better hardware.

Usage (repository root)::

    python -m research.overfit_budget --models mmt,gru,mlp,dlinear \
        --lrs 1e-3,3e-3,1e-2 --seeds 0,1 --epochs 200 --patience 50 \
        --out research/results_overfit_budget.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .synthetic_torus import build_model as build_race_model
from .synthetic_torus import simulate_switching_rotation
from .train_utils import count_parameters, evaluate, train_model


def parse_floats(text: str) -> list[float]:
    values = [float(v) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("empty learning-rate list")
    return values


def parse_ints(text: str) -> list[int]:
    values = [int(v) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("empty seed list")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--lrs", type=str, default="1e-3,3e-3,1e-2",
                        help="comma-separated learning-rate grid")
    parser.add_argument("--seeds", type=str, default="0,1",
                        help="comma-separated seed list")
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--states", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--kernel", type=str, default="stochastic", choices=("stochastic", "free"))
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--amplitude", action="store_true")
    parser.add_argument("--diff-input", action="store_true")
    parser.add_argument("--models", type=str, default="mmt,gru,mlp,dlinear")
    parser.add_argument("--out", type=str, default="research/results_overfit_budget.json")
    args = parser.parse_args()

    lrs = parse_floats(args.lrs)
    seeds = parse_ints(args.seeds)
    torch.manual_seed(0)
    np.random.seed(0)
    length = args.seq + args.horizon

    print(
        f"Overfit probe: {args.n_train} train sequences, {args.epochs} epochs,"
        f" patience {args.patience}, lrs={lrs}, seeds={seeds}"
    )
    x_all, _ = simulate_switching_rotation(
        args.n_train + args.n_val + args.n_test, length, seed=7
    )
    train_x = torch.from_numpy(x_all[: args.n_train, : args.seq])
    train_y = torch.from_numpy(x_all[: args.n_train, args.seq :])
    val_x = torch.from_numpy(x_all[args.n_train : args.n_train + args.n_val, : args.seq])
    val_y = torch.from_numpy(x_all[args.n_train : args.n_train + args.n_val, args.seq :])
    test_x = torch.from_numpy(
        x_all[args.n_train + args.n_val :, : args.seq]
    )
    test_y = torch.from_numpy(x_all[args.n_train + args.n_val :, args.seq :])
    print(f"windows: train {tuple(train_x.shape)}, val {tuple(val_x.shape)}, test {tuple(test_x.shape)}\n")

    # min achievable val MSE reference: the mean predictor on this split set.
    mean_val_mse = ((val_y - train_y.mean()) ** 2).mean().item()
    print(f"predict-train-mean reference: val mse {mean_val_mse:.4f}\n")

    results = {
        "config": {**vars(args), "lrs": lrs, "seeds": seeds},
        "mean_predictor_val_mse": round(mean_val_mse, 5),
        "runs": {},
    }
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        results["runs"][name] = {}
        best = None
        for lr in lrs:
            for seed in seeds:
                model = build_race_model(name, train_x.shape[-1], args.seq, args.horizon, args)
                params = count_parameters(model)
                print(f"[{name}] lr={lr:g} seed={seed} params={params:,}", flush=True)
                outcome = train_model(
                    model, train_x, train_y, val_x, val_y,
                    epochs=args.epochs, batch_size=args.batch, lr=lr,
                    patience=args.patience, seed=seed, log=False,
                )
                test_mse, test_mae = evaluate(outcome["model"], test_x, test_y)
                train_mses = [h["train_mse"] for h in outcome["history"] if "train_mse" in h]
                val_mses = [h["val_mse"] for h in outcome["history"] if "val_mse" in h]
                entry = {
                    "lr": lr,
                    "seed": seed,
                    "params": params,
                    "epochs_run": len(train_mses),
                    "min_train_mse": round(min(train_mses), 5),
                    "final_train_mse": round(train_mses[-1], 5),
                    "best_val_mse": round(min(val_mses), 5),
                    "test_mse": round(test_mse, 5),
                    "test_mae": round(test_mae, 5),
                    "train_seconds": outcome["history"][-1].get("train_seconds"),
                }
                key = f"lr{lr:g}_seed{seed}"
                results["runs"][name][key] = entry
                print(
                    f"    min train {entry['min_train_mse']:.4f}  best val"
                    f" {entry['best_val_mse']:.4f}  test {entry['test_mse']:.4f}"
                    f"  ({entry['epochs_run']} epochs, {entry['train_seconds']}s)",
                    flush=True,
                )
                if best is None or entry["test_mse"] < best["test_mse"]:
                    best = {**entry, "config_key": key}
        results["runs"][name]["best"] = best
        print(
            f"[{name}] BEST: {best['config_key']}  min train {best['min_train_mse']:.4f}"
            f"  test {best['test_mse']:.4f}\n",
            flush=True,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== Long-budget overfit probe: best config per model ===")
    print(f"(mean-predictor val mse reference: {mean_val_mse:.4f};")
    print(" the plateau hypothesis predicts min train MSE stuck near ~0.49)")
    for name, runs in results["runs"].items():
        best = runs["best"]
        print(
            f"  {name:<10} {best['config_key']:<18} min train {best['min_train_mse']:.4f}"
            f"  test {best['test_mse']:.4f}"
        )
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()

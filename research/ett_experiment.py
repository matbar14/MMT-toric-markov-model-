"""Pilot on real seasonal data (ETTh1) — testing the model outside trading.

ETTh1 (electricity transformer temperature, hourly, 7 channels) is the
standard public long-horizon forecasting benchmark, with strong daily and
weekly seasonality — exactly the structure a phase/torus state space is meant
to encode.  The canonical protocol is used:

  - input 96 hourly steps -> predict the next 96;
  - chronological split 12 / 4 / 4 months (train 8640 rows, val 2880, test 2880);
  - per-channel z-score fitted on the training slice only;
  - MSE / MAE reported on the standardized scale (comparable with published
    DLinear / PatchTST / iTransformer tables).

Published reference on this exact protocol: DLinear 96->96 multivariate MSE
is about 0.375-0.41.  This pilot uses small models and few epochs, so it is
directional evidence, not a benchmark claim.

Data source (public, no keys):
  https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv

Usage (repository root)::

    python -m research.ett_experiment --models dlinear,gru,seasonal
    python -m research.ett_experiment --models mmt,hybrid --epochs 5
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines import (
    DLinearForecaster,
    RecurrentForecaster,
    seasonal_naive_forecast,
)
from .train_utils import (
    Standardizer,
    count_parameters,
    evaluate,
    evaluate_naive,
    make_windows,
    train_hybrid_residual,
    train_model,
)
from .toric_core import HybridForecaster, ToricForecaster

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
SEQ_LEN = 96
HORIZON = 96
TRAIN_END = 12 * 30 * 24  # 8640
VAL_END = TRAIN_END + 4 * 30 * 24  # 11520
TEST_END = VAL_END + 4 * 30 * 24  # 14400


def load_etth1(data_dir: Path) -> np.ndarray:
    """Download (once) and return the 7-channel float array of the first 14400 rows."""
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "ETTh1.csv"
    if not csv_path.exists():
        print(f"Downloading {ETTH1_URL} ...")
        urllib.request.urlretrieve(ETTH1_URL, csv_path)
    frame = pd.read_csv(csv_path)
    columns = [c for c in frame.columns if c != "date"]
    values = frame[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("ETTh1 contains non-finite values")
    print(f"ETTh1: {values.shape[0]} rows x {values.shape[1]} channels ({', '.join(columns)})")
    return values[:TEST_END]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data/ETT-small")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--dim", type=int, default=32, help="toric angle dimension")
    parser.add_argument("--states", type=int, default=32, help="toric states")
    parser.add_argument("--hidden", type=int, default=64, help="baseline width")
    parser.add_argument("--diff-input", action="store_true",
                        help="append first differences to the model inputs")
    parser.add_argument("--base-epochs", type=int, default=4,
                        help="linear-base epochs for hybrid_res phase 1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--models", type=str, default="dlinear,gru,seasonal",
        help="comma list from: dlinear, gru, lstm, transformer, mmt, hybrid, hybrid_res, seasonal",
    )
    parser.add_argument("--out", type=str, default="research/results_ett.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    values = load_etth1(Path(args.data_dir))
    standardizer = Standardizer.fit(values[:TRAIN_END])
    normalized = standardizer.transform(values)

    borders = [
        (0, TRAIN_END),
        (TRAIN_END - SEQ_LEN, VAL_END),
        (VAL_END - SEQ_LEN, TEST_END),
    ]
    splits = {}
    for name, (start, end) in zip(("train", "val", "test"), borders):
        splits[name] = make_windows(normalized[start:end], SEQ_LEN, HORIZON)
        print(f"{name}: {splits[name][0].shape[0]} windows")
    train_x, train_y = splits["train"]
    val_x, val_y = splits["val"]
    test_x, test_y = splits["test"]

    def build(name: str):
        if name == "dlinear":
            return DLinearForecaster(SEQ_LEN, HORIZON, num_channels=7)
        if name in ("gru", "lstm"):
            return RecurrentForecaster(
                7, HORIZON, num_channels=7,
                kind=name, hidden_dim=args.hidden, use_diff_input=args.diff_input,
            )
        if name == "transformer":
            from .baselines import TransformerForecaster

            return TransformerForecaster(
                7, HORIZON, num_channels=7, d_model=args.hidden,
                use_diff_input=args.diff_input,
            )
        if name in ("mmt", "hybrid", "hybrid_res"):
            toric_kwargs = dict(
                dim_angles=args.dim, num_levels=2, num_states=args.states,
                num_layers=1, kernel="stochastic", use_attention=True,
                use_amplitude=True, use_diff_input=args.diff_input,
            )
            if name == "mmt":
                return ToricForecaster(7, HORIZON, num_channels=7, **toric_kwargs)
            return HybridForecaster(7, SEQ_LEN, HORIZON, num_channels=7, **toric_kwargs)
        raise ValueError(f"unknown model {name!r}")

    results = {"config": vars(args), "models": {}}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        if name == "seasonal":
            mse, mae = evaluate_naive(
                lambda xb: seasonal_naive_forecast(xb, HORIZON, 24), test_x, test_y
            )
            results["models"][name] = {"test_mse": round(mse, 5), "test_mae": round(mae, 5)}
            print(f"[seasonal-24] test mse={mse:.4f} mae={mae:.4f}")
            continue
        model = build(name)
        print(f"[{name}] params={count_parameters(model):,}")
        if name == "hybrid_res":
            outcome = train_hybrid_residual(
                model, train_x, train_y, val_x, val_y,
                base_epochs=args.base_epochs, residual_epochs=args.epochs,
                batch_size=args.batch, lr=args.lr, patience=args.patience,
                seed=args.seed,
            )
        else:
            outcome = train_model(
                model, train_x, train_y, val_x, val_y,
                epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                patience=args.patience, seed=args.seed,
            )
        test_mse, test_mae = evaluate(outcome["model"], test_x, test_y)
        results["models"][name] = {
            "params": count_parameters(model),
            "best_val_mse": round(outcome["best_val_mse"], 5),
            "test_mse": round(test_mse, 5),
            "test_mae": round(test_mae, 5),
        }
        print(f"[{name}] test mse={test_mse:.4f} mae={test_mae:.4f}\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== ETTh1 96->96, standardized-scale MSE (published DLinear reference ~0.375-0.41) ===")
    ranked = sorted(results["models"].items(), key=lambda kv: kv[1]["test_mse"])
    for name, entry in ranked:
        extra = f"  params={entry['params']:,}" if entry.get("params") else ""
        print(f"  {name:<12} mse={entry['test_mse']:.4f}  mae={entry['test_mae']:.4f}{extra}")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()

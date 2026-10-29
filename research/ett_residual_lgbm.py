"""Roadmap item 2: is the post-linear ETTh1 residual machine-learnable at all?

Round 2 left a sharp question: after the DLinear base, the ETTh1 residual
still carries real structure (val MSE ~0.67 on the standardized scale) but
the toric core cannot extract it.  Before blaming the toric core, this
experiment asks whether ANY learner can extract that residual — using
gradient boosting (LightGBM) over lag / envelope / calendar features, the
same family the trading pipeline already trusts elsewhere in this repository.

Design
------
1. Train the DLinear base exactly as in ``ett_experiment`` (96 -> 96 protocol,
   chronological splits, train-only standardization) and freeze it.
2. Compute residuals  r = y - base(x)  on train / val / test windows.
3. For every channel c fit one LightGBM regressor shared across forecast
   steps, with one row per (window, step) pair and features:
     - the 96 window values of channel c (the raw lags);
     - envelope statistics of the window (mean, std, last value, last diff,
       RMS of the last 24 steps, mean absolute diff) — the same quadratic
       evidence a regime detector would need;
     - the forecast step index h;
     - calendar phase of the TARGET timestamp (hour-of-day and day-of-week
       as sin/cos pairs), taken from the ETTh1 ``date`` column.
4. Report how much of the residual variance is explained on val (per channel
   and overall) and the resulting full-model test MSE vs the DLinear base.

Interpretation
--------------
- Residual R^2 ~ 0  -> the leftover is (almost) pure noise; the linear base is
  the achievable ceiling and "the toric core cannot learn the residual" stops
  being an accusation.
- Residual R^2 clearly > 0 -> the structure exists and is extractable; the
  feature-importance breakdown then says WHAT carries it (calendar phase vs
  envelope vs raw lags), which is exactly the signal the toric model should
  be able to represent with its phase machinery.

Usage (repository root)::

    python -m research.ett_residual_lgbm --base-epochs 8 \
        --out research/results_ett_residual_lgbm.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from .baselines import DLinearForecaster
from .train_utils import (
    Standardizer,
    count_parameters,
    evaluate,
    make_windows,
    train_model,
)

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
SEQ_LEN = 96
HORIZON = 96
TRAIN_END = 12 * 30 * 24  # 8640
VAL_END = TRAIN_END + 4 * 30 * 24  # 11520
TEST_END = VAL_END + 4 * 30 * 24  # 14400
CHANNELS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]

STAT_NAMES = ["w_mean", "w_std", "w_last", "w_lastdiff", "w_rms24", "w_mad"]
FEATURE_NAMES = (
    [f"lag{i}" for i in range(SEQ_LEN)]
    + STAT_NAMES
    + ["step"]
    + [f"cal{i}" for i in range(4)]
)


def load_etth1(data_dir: Path) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Download (once) and return values (14400, 7) and the matching timestamps."""
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "ETTh1.csv"
    if not csv_path.exists():
        print(f"Downloading {ETTH1_URL} ...")
        urllib.request.urlretrieve(ETTH1_URL, csv_path)
    frame = pd.read_csv(csv_path).iloc[:TEST_END]
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
    values = frame[CHANNELS].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("ETTh1 contains non-finite values")
    print(f"ETTh1: {values.shape[0]} rows x {values.shape[1]} channels")
    return values, dates


def target_meta_matrix(dates: pd.DatetimeIndex, start: int, n_windows: int) -> np.ndarray:
    """(N + H - 1, 2) array of (hour, dayofweek) for all target timestamps.

    Windows are built from rows [start, end); window i predicts rows
    start + i + 96 ... start + i + 191, so the target rows for all windows
    jointly span [start + 96, end), which holds N + H - 1 rows.
    """
    targets = dates[start + SEQ_LEN : start + SEQ_LEN + n_windows + HORIZON - 1]
    if len(targets) != n_windows + HORIZON - 1:
        raise ValueError("calendar alignment broken: wrong number of target rows")
    return np.stack([targets.hour.to_numpy(), targets.dayofweek.to_numpy()], axis=-1)


def window_stats(x: np.ndarray) -> np.ndarray:
    """Per-window envelope statistics; x is (N, T) -> (N, 6)."""
    diffs = np.diff(x, axis=1)
    last24 = x[:, -24:]
    return np.stack(
        [
            x.mean(axis=1),
            x.std(axis=1),
            x[:, -1],
            diffs[:, -1],
            np.sqrt((last24**2).mean(axis=1)),
            np.abs(diffs).mean(axis=1),
        ],
        axis=-1,
    ).astype(np.float32)


def build_rows(
    x_channel: np.ndarray,
    target_meta: np.ndarray,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the (window, step) feature rows for ONE channel.

    x_channel: (N, T) window values of the channel; target_meta: (N+H-1, 2).
    Returns X (N' * H, 107), y (N' * H) where N' = ceil(N / stride).
    Row (i, h) = "window i predicts step h".  Window-level blocks (lags,
    stats) are repeated across steps; step-level blocks (step index, target
    calendar phase) are tiled across windows.  Sampling every ``stride``-th
    window keeps the training matrix bounded (overlapping windows are highly
    redundant anyway).
    """
    n_windows = x_channel.shape[0]
    # (N, H, 2): target (hour, dow) per window and step, via a sliding view.
    meta = np.lib.stride_tricks.sliding_window_view(target_meta, HORIZON, axis=0)
    meta = np.ascontiguousarray(meta.transpose(0, 2, 1))  # (N, H, 2)
    if meta.shape[0] != n_windows:
        raise ValueError(f"calendar misalignment: {meta.shape[0]} != {n_windows}")
    hour, dow = meta[..., 0], meta[..., 1]
    cal = np.stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
        ],
        axis=-1,
    ).astype(np.float32)  # (N, H, 4)

    xs = x_channel[::stride]
    cs = cal[::stride]
    n_sub = xs.shape[0]
    steps = np.tile(np.arange(HORIZON, dtype=np.float32), n_sub)[:, None]
    X = np.concatenate(
        [
            np.repeat(xs.astype(np.float32), HORIZON, axis=0),  # (N'*H, T)
            np.repeat(window_stats(xs), HORIZON, axis=0),  # (N'*H, 6)
            steps,  # (N'*H, 1)
            cs.reshape(n_sub * HORIZON, 4),  # (N'*H, 4)
        ],
        axis=1,
    )
    return X, n_sub


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data/ETT-small")
    parser.add_argument("--base-epochs", type=int, default=8,
                        help="DLinear base training epochs (phase 1)")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--stride-train", type=int, default=2,
                        help="window subsampling for LightGBM training rows")
    parser.add_argument("--stride-val", type=int, default=4,
                        help="window subsampling for LightGBM early-stopping rows")
    parser.add_argument("--num-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="research/results_ett_residual_lgbm.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    values, dates = load_etth1(Path(args.data_dir))
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
    split_start = {"train": 0, "val": TRAIN_END - SEQ_LEN, "test": VAL_END - SEQ_LEN}

    # ---- Phase 1: DLinear base, identical protocol to ett_experiment ----
    base = DLinearForecaster(SEQ_LEN, HORIZON, num_channels=7)
    print(f"\n[base dlinear] params={count_parameters(base):,}")
    base_outcome = train_model(
        base, train_x, train_y, val_x, val_y,
        epochs=args.base_epochs, batch_size=args.batch, lr=args.lr,
        patience=args.patience, seed=args.seed, log=True,
    )
    base_model = base_outcome["model"]

    # ---- Phase 2: residuals and per-channel LightGBM corrections ----
    with torch.no_grad():
        base_model.eval()
        residual_train = (train_y - base_model(train_x)).numpy()
        residual_val = (val_y - base_model(val_x)).numpy()
        residual_test = (test_y - base_model(test_x)).numpy()
        base_val_pred = base_model(val_x)
        base_test_pred = base_model(test_x)

    res_val_before = float((residual_val**2).mean())
    base_test_mse, base_test_mae = evaluate(base_model, test_x, test_y)
    print(
        f"\nresidual val mse after base: {res_val_before:.4f} "
        f"(base test mse {base_test_mse:.4f}, mae {base_test_mae:.4f})"
    )

    # Per-split metadata + target index matrices shared by all channels.
    meta_by_split = {}
    for split in ("train", "val", "test"):
        x_split = {"train": train_x, "val": val_x, "test": test_x}[split]
        meta_by_split[split] = target_meta_matrix(
            dates, split_start[split], x_split.shape[0]
        )

    corr_val_channels = []
    corr_test_channels = []
    channel_reports = []

    for c, channel in enumerate(CHANNELS):
        t0 = time.time()
        X_parts, y_parts, sizes = [], [], {}
        for split in ("train", "val", "test"):
            x_channel = {"train": train_x, "val": val_x, "test": test_x}[split][:, :, c]
            residual = {"train": residual_train, "val": residual_val, "test": residual_test}[split]
            stride = {"train": args.stride_train, "val": args.stride_val, "test": 1}[split]
            X, n_sub = build_rows(x_channel.numpy(), meta_by_split[split], stride)
            y = residual[::stride, :, c].reshape(n_sub * HORIZON)
            X_parts.append(X)
            y_parts.append(y)
            sizes[split] = (n_sub, X.shape[0], X.shape[1])
        X_tr, X_va, X_te = X_parts
        y_tr, y_va, _ = y_parts
        # Full-resolution val rows for evaluation (early stopping may use a
        # subsampled copy, but the reported metrics must cover every window).
        X_va_full, n_val_full = build_rows(
            val_x[:, :, c].numpy(), meta_by_split["val"], 1
        )
        print(
            f"[{channel}] rows: train {X_tr.shape[0]:,}  val {X_va.shape[0]:,}"
            f"  test {X_te.shape[0]:,}  features {X_tr.shape[1]}",
            flush=True,
        )

        booster = lgb.LGBMRegressor(
            objective="l2",
            n_estimators=args.num_estimators,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=200,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=1,
            n_jobs=2,
            seed=args.seed,
            verbosity=-1,
        )
        booster.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="l2",
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )

        n_val = n_val_full
        n_test = sizes["test"][0]
        corr_val = booster.predict(X_va_full).reshape(n_val, HORIZON)
        corr_test = booster.predict(X_te).reshape(n_test, HORIZON)
        corr_val_channels.append(corr_val)
        corr_test_channels.append(corr_test)

        before_c = float((residual_val[:, :, c] ** 2).mean())
        after_c = float(((residual_val[:, :, c] - corr_val) ** 2).mean())
        r2_c = 1 - after_c / before_c
        importance = booster.booster_.feature_importance(importance_type="gain")
        top = np.argsort(importance)[::-1][:10]
        report = {
            "residual_val_mse_before": round(before_c, 5),
            "residual_val_mse_after": round(after_c, 5),
            "residual_r2_val": round(r2_c, 4),
            "best_iteration": int(booster.best_iteration_ or args.num_estimators),
            "top_features_by_gain": [
                {"feature": FEATURE_NAMES[i], "gain": round(float(importance[i]), 1)}
                for i in top
                if importance[i] > 0
            ],
            "seconds": round(time.time() - t0, 1),
        }
        channel_reports.append({"channel": channel, **report})
        print(
            f"[{channel}] residual val mse {before_c:.4f} -> {after_c:.4f}"
            f"  (R2={r2_c:.3f}, {report['best_iteration']} trees,"
            f" {report['seconds']}s)",
            flush=True,
        )

    # ---- Full-model metrics: base + stacked per-channel corrections ----
    corr_val_full = torch.from_numpy(
        np.stack(corr_val_channels, axis=-1).astype(np.float32)
    )
    corr_test_full = torch.from_numpy(
        np.stack(corr_test_channels, axis=-1).astype(np.float32)
    )
    full_val_pred = base_val_pred + corr_val_full
    full_test_pred = base_test_pred + corr_test_full
    full_val_mse = torch.nn.functional.mse_loss(full_val_pred, val_y).item()
    full_test_mse = torch.nn.functional.mse_loss(full_test_pred, test_y).item()
    full_test_mae = (full_test_pred - test_y).abs().mean().item()
    after_val_all = float(
        ((residual_val - corr_val_full.numpy()) ** 2).mean()
    )
    overall_r2 = 1 - after_val_all / res_val_before

    results = {
        "config": vars(args),
        "base": {
            "best_val_mse": round(base_outcome["best_val_mse"], 5),
            "test_mse": round(base_test_mse, 5),
            "test_mae": round(base_test_mae, 5),
        },
        "residual": {
            "val_mse_before": round(res_val_before, 5),
            "val_mse_after_lgbm": round(after_val_all, 5),
            "overall_r2_val": round(overall_r2, 4),
        },
        "full_model": {
            "val_mse": round(full_val_mse, 5),
            "test_mse": round(full_test_mse, 5),
            "test_mae": round(full_test_mae, 5),
        },
        "channels": channel_reports,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== ETTh1 residual learnability (LightGBM over lag/envelope/calendar) ===")
    print(
        f"  residual val mse  {res_val_before:.4f} -> {after_val_all:.4f}"
        f"   (R2 = {overall_r2:.3f})"
    )
    print(f"  base        test mse {base_test_mse:.4f}  mae {base_test_mae:.4f}")
    print(f"  base+lgbm   test mse {full_test_mse:.4f}  mae {full_test_mae:.4f}")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()

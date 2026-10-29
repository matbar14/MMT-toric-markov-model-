"""Roadmap item 2 (hardened): is the post-linear ETTh1 residual machine-learnable at all?

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
4. Harden the headline number with the controls a reviewer needs
   (all fits share the identical feature matrix; only the target changes):
     - ``real_es``     residual target, early stopping on val (the original run);
     - ``real_full``   residual target, all N trees, no early stopping — removes
                       the val-selection bias of small best_iteration values;
     - ``shuffle_es``  TRAIN residual targets row-permuted, val kept real —
                       the noise floor: any val R^2 above it is signal;
     - ``direct_es``   RAW target y (not the residual), early stopping — can
                       boosting on these features beat the linear base at all
                       when trained directly?
     - ``direct_full`` RAW target, all N trees.
   Every variant is repeated over ``--seeds`` (booster seed varies: bagging,
   feature fraction, tie-breaking); the DLinear base stays at seed 0 so the
   residual definition is fixed.  Reported per variant: mean +- std over
   seeds of per-channel and pooled R^2 (val AND test — the test numbers are
   untouched by any selection), plus the stacked full-model test MSE.
5. Report how much of the residual variance is explained on val/test and the
   resulting full-model test MSE vs the DLinear base.

Interpretation
--------------
- Residual R^2 (real) ~ shuffle R^2  -> the leftover is indistinguishable
  from noise FOR THIS FEATURE FAMILY; this bounds boosting-over-lags, not the
  task: trees do not extrapolate, so any structure that requires it is
  outside what this test can see.
- Residual R^2 (real) clearly > shuffle R^2, stable across seeds  -> the
  structure exists and is extractable; the feature-importance breakdown then
  says WHAT carries it (calendar phase vs envelope vs raw lags), which is
  exactly the signal the toric model should be able to represent with its
  phase machinery.

Usage (repository root)::

    python -m research.ett_residual_lgbm --base-epochs 8 --seeds 0,1,2 \
        --stride-train 3 --stride-val 6 \
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

# (target_kind, early_stopping): the four fit variants per (channel, seed).
VARIANTS = [
    ("real_es", "residual", True),
    ("real_full", "residual", False),
    ("shuffle_es", "shuffled", True),
    ("direct_es", "raw", True),
]


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


def make_booster(seed: int, num_estimators: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="l2",
        n_estimators=num_estimators,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=200,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=1,
        n_jobs=2,
        seed=seed,
        verbosity=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data/ETT-small")
    parser.add_argument("--base-epochs", type=int, default=8,
                        help="DLinear base training epochs (phase 1)")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--stride-train", type=int, default=3,
                        help="window subsampling for LightGBM training rows")
    parser.add_argument("--stride-val", type=int, default=6,
                        help="window subsampling for LightGBM early-stopping rows")
    parser.add_argument("--num-estimators", type=int, default=250)
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="booster seeds (mean +- std over these); base stays at seed 0")
    parser.add_argument("--cache-base", type=str, default="",
                        help="optional path: load/save the frozen DLinear base state "
                             "(skips retraining on repeated invocations, e.g. per-seed runs)")
    parser.add_argument("--out", type=str, default="research/results_ett_residual_lgbm.json")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    torch.manual_seed(0)
    np.random.seed(0)

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
    cache_path = Path(args.cache_base) if args.cache_base else None
    if cache_path and cache_path.exists():
        base.load_state_dict(torch.load(cache_path, weights_only=True))
        base_model = base
        with torch.no_grad():
            base_model.eval()
            base_test_mse, base_test_mae = evaluate(base_model, test_x, test_y)
        base_outcome = {"model": base_model, "best_val_mse": None,
                        "history": [{"loaded_from_cache": True}]}
        print(f"[base dlinear] loaded from {cache_path} (test mse {base_test_mse:.4f})")
    else:
        base_outcome = train_model(
            base, train_x, train_y, val_x, val_y,
            epochs=args.base_epochs, batch_size=args.batch, lr=args.lr,
            patience=args.patience, seed=0, log=True,
        )
        base_model = base_outcome["model"]
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(base_model.state_dict(), cache_path)
            print(f"[base dlinear] cached to {cache_path}")

    # ---- Phase 2: residuals, per-channel boosters, controls ----
    with torch.no_grad():
        base_model.eval()
        residual_train = (train_y - base_model(train_x)).numpy()
        residual_val = (val_y - base_model(val_x)).numpy()
        residual_test = (test_y - base_model(test_x)).numpy()
        base_val_pred = base_model(val_x)
        base_test_pred = base_model(test_x)

    res_val_before = float((residual_val**2).mean())
    res_test_before = float((residual_test**2).mean())
    base_test_mse, base_test_mae = evaluate(base_model, test_x, test_y)
    print(
        f"\nresidual val mse after base: {res_val_before:.4f} "
        f"(test {res_test_before:.4f}; base test mse {base_test_mse:.4f},"
        f" mae {base_test_mae:.4f})"
    )

    # Per-split metadata + target index matrices shared by all channels.
    meta_by_split = {}
    for split in ("train", "val", "test"):
        x_split = {"train": train_x, "val": val_x, "test": test_x}[split]
        meta_by_split[split] = target_meta_matrix(
            dates, split_start[split], x_split.shape[0]
        )

    # Predictions collected per (variant, seed): (N_split, H, C) once stacked.
    corr_store: dict[tuple[str, int], list[np.ndarray]] = {
        (name, seed): [] for name, _, _ in VARIANTS for seed in seeds
    }
    per_channel: dict[tuple[str, int, str], dict] = {}
    t0_all = time.time()

    for c, channel in enumerate(CHANNELS):
        t0 = time.time()
        # ---- feature matrices: built once, shared by every variant/seed ----
        X_by_split, n_sub = {}, {}
        for split in ("train", "val", "test"):
            x_channel = {"train": train_x, "val": val_x, "test": test_x}[split][:, :, c]
            stride = {"train": args.stride_train, "val": args.stride_val, "test": 1}[split]
            X_by_split[split], n_sub[split] = build_rows(
                x_channel.numpy(), meta_by_split[split], stride
            )
        X_tr, X_va, X_te = (X_by_split[s] for s in ("train", "val", "test"))
        # Full-resolution val rows for evaluation (early stopping may use a
        # subsampled copy, but the reported metrics must cover every window).
        X_va_full, n_val_full = build_rows(val_x[:, :, c].numpy(), meta_by_split["val"], 1)

        # ---- targets per variant ----
        targets = {
            "residual": {
                "train": residual_train[:: args.stride_train, :, c].reshape(-1),
                "val_es": residual_val[:: args.stride_val, :, c].reshape(-1),
            },
            "raw": {
                "train": train_y[:: args.stride_train, :, c].reshape(-1).numpy(),
                "val_es": val_y[:: args.stride_val, :, c].reshape(-1).numpy(),
            },
        }
        rng = np.random.RandomState(seeds[0] * 1000 + 17)
        targets["shuffled"] = {
            "train": targets["residual"]["train"][
                rng.permutation(targets["residual"]["train"].shape[0])
            ],
            "val_es": targets["residual"]["val_es"],
        }
        before_c = float((residual_val[:, :, c] ** 2).mean())
        before_test_c = float((residual_test[:, :, c] ** 2).mean())

        print(
            f"[{channel}] rows: train {X_tr.shape[0]:,}  val {X_va.shape[0]:,}"
            f"  test {X_te.shape[0]:,}  features {X_tr.shape[1]}",
            flush=True,
        )

        for seed in seeds:
            # Re-derive the shuffle with the seed-specific RNG.
            targets["shuffled"]["train"] = targets["residual"]["train"][
                np.random.RandomState(seed * 1000 + 17).permutation(
                    targets["residual"]["train"].shape[0]
                )
            ]
            for vname, tkind, use_es in VARIANTS:
                booster = make_booster(seed, args.num_estimators)
                y_tr = targets[tkind]["train"]
                if use_es:
                    booster.fit(
                        X_tr, y_tr,
                        eval_set=[(X_va, targets[tkind]["val_es"])],
                        eval_metric="l2",
                        callbacks=[lgb.early_stopping(40, verbose=False)],
                    )
                else:
                    booster.fit(X_tr, y_tr)
                best_iter = int(booster.best_iteration_ or args.num_estimators)

                if tkind == "raw":
                    # Direct task: score the full-model test MSE of the
                    # direct predictions (no residual R^2 to report).
                    pred_test = booster.predict(X_te).reshape(n_sub["test"], HORIZON)
                    corr_store[(vname, seed)].append(pred_test.astype(np.float32))
                    per_channel[(vname, seed, channel)] = {"best_iteration": best_iter}
                    continue

                corr_val = booster.predict(X_va_full).reshape(n_val_full, HORIZON)
                corr_test = booster.predict(X_te).reshape(n_sub["test"], HORIZON)
                after_val = float(((residual_val[:, :, c] - corr_val) ** 2).mean())
                after_test = float(((residual_test[:, :, c] - corr_test) ** 2).mean())
                entry = {
                    "residual_r2_val": round(1 - after_val / before_c, 4),
                    "residual_r2_test": round(1 - after_test / before_test_c, 4),
                    "best_iteration": best_iter,
                }
                if vname == "real_es" and seed == seeds[0]:
                    importance = booster.booster_.feature_importance(importance_type="gain")
                    top = np.argsort(importance)[::-1][:10]
                    entry["top_features_by_gain"] = [
                        {"feature": FEATURE_NAMES[i], "gain": round(float(importance[i]), 1)}
                        for i in top
                        if importance[i] > 0
                    ]
                per_channel[(vname, seed, channel)] = entry
                if tkind == "residual":
                    # Only real residual fits enter the stacked full model.
                    corr_store[(vname, seed)].append(corr_test.astype(np.float32))
                    if vname == "real_es":
                        print(
                            f"  seed {seed}: val R2={entry['residual_r2_val']:+.4f}"
                            f"  test R2={entry['residual_r2_test']:+.4f}"
                            f"  ({best_iter} trees)",
                            flush=True,
                        )
                    elif vname == "shuffle_es":
                        print(
                            f"  seed {seed}: shuffle val R2={entry['residual_r2_val']:+.4f}"
                            f"  ({best_iter} trees)",
                            flush=True,
                        )
        print(f"[{channel}] done in {time.time() - t0:.0f}s", flush=True)

    # ---- aggregate over seeds ----
    # Pooled (all channels) residual R^2 per variant per seed, from stacked corrections.
    pooled = {}
    for vname in ("real_es", "real_full", "shuffle_es"):
        per_seed_val, per_seed_test = [], []
        for seed in seeds:
            per_seed_val.append(
                float(np.mean([per_channel[(vname, seed, ch)]["residual_r2_val"] for ch in CHANNELS]))
            )
            per_seed_test.append(
                float(np.mean([per_channel[(vname, seed, ch)]["residual_r2_test"] for ch in CHANNELS]))
            )
        pooled[vname] = {
            "r2_val_per_seed": [round(v, 4) for v in per_seed_val],
            "r2_test_per_seed": [round(v, 4) for v in per_seed_test],
            "r2_val_mean": round(float(np.mean(per_seed_val)), 4),
            "r2_val_std": round(float(np.std(per_seed_val, ddof=1)), 4) if len(seeds) > 1 else 0.0,
            "r2_test_mean": round(float(np.mean(per_seed_test)), 4),
            "r2_test_std": round(float(np.std(per_seed_test, ddof=1)), 4) if len(seeds) > 1 else 0.0,
        }

    # Full-model test MSE: DLinear base + stacked per-channel corrections.
    def stacked_test_mse(vname: str, seed: int) -> float:
        corr = torch.from_numpy(np.stack(corr_store[(vname, seed)], axis=-1))
        return torch.nn.functional.mse_loss(base_test_pred + corr, test_y).item()

    full_model = {}
    for vname in ("real_es", "real_full"):
        mses = [stacked_test_mse(vname, seed) for seed in seeds]
        full_model[vname] = {
            "test_mse_per_seed": [round(m, 5) for m in mses],
            "test_mse_mean": round(float(np.mean(mses)), 5),
            "test_mse_std": round(float(np.std(mses, ddof=1)), 5) if len(seeds) > 1 else 0.0,
        }

    def direct_stacked_test_mse(seed: int, vname: str) -> float:
        direct = torch.from_numpy(np.stack(corr_store[(vname, seed)], axis=-1))
        return torch.nn.functional.mse_loss(direct, test_y).item()

    direct_model = {}
    for vname in ("direct_es",):
        mses = [direct_stacked_test_mse(seed, vname) for seed in seeds]
        direct_model[vname] = {
            "test_mse_per_seed": [round(m, 5) for m in mses],
            "test_mse_mean": round(float(np.mean(mses)), 5),
            "test_mse_std": round(float(np.std(mses, ddof=1)), 5) if len(seeds) > 1 else 0.0,
        }

    # Per-channel aggregation over seeds (+ seed-0 top features for real_es).
    channel_reports = []
    for c, channel in enumerate(CHANNELS):
        entry = {"channel": channel}
        for vname in ("real_es", "real_full", "shuffle_es", "direct_es"):
            per_seed = [per_channel[(vname, seed, channel)] for seed in seeds]
            agg = {}
            for key in per_seed[0]:
                if key == "top_features_by_gain":
                    agg[key] = per_seed[0][key]
                    continue
                values_ = [p[key] for p in per_seed]
                agg[key] = {
                    "mean": round(float(np.mean(values_)), 4),
                    "std": round(float(np.std(values_, ddof=1)), 4) if len(seeds) > 1 else 0.0,
                }
            entry[vname] = agg
        channel_reports.append(entry)

    results = {
        "config": {**vars(args), "seeds": seeds},
        "base": {
            "best_val_mse": (round(base_outcome["best_val_mse"], 5)
                             if base_outcome["best_val_mse"] is not None else "cached"),
            "test_mse": round(base_test_mse, 5),
            "test_mae": round(base_test_mae, 5),
        },
        "residual": {
            "val_mse_before": round(res_val_before, 5),
            "test_mse_before": round(res_test_before, 5),
            **{f"pooled_{vname}": stats for vname, stats in pooled.items()},
        },
        "full_model": full_model,
        "direct_lgbm": direct_model,
        "channels": channel_reports,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n=== ETTh1 residual learnability (hardened: seeds + shuffle + direct) ===")
    print(f"  base dlinear test mse {base_test_mse:.4f}  mae {base_test_mae:.4f}")
    for vname in ("real_es", "real_full", "shuffle_es"):
        p = pooled[vname]
        print(
            f"  {vname:<11} pooled R2 val {p['r2_val_mean']:+.4f} +- {p['r2_val_std']:.4f}"
            f"   test {p['r2_test_mean']:+.4f} +- {p['r2_test_std']:.4f}"
        )
    for vname in ("direct_es",):
        d = direct_model[vname]
        print(
            f"  {vname:<11} direct test mse {d['test_mse_mean']:.4f} +- {d['test_mse_std']:.4f}"
        )
    for vname in ("real_es", "real_full"):
        f = full_model[vname]
        print(
            f"  base+{vname:<9} test mse {f['test_mse_mean']:.4f} +- {f['test_mse_std']:.4f}"
        )
    print(f"\n(total {time.time() - t0_all:.0f}s)\nsaved: {out_path}")


if __name__ == "__main__":
    main()

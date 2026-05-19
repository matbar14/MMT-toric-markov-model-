#!/usr/bin/env python3
"""Train a strong tabular baseline for gate task: Hold vs Any Pattern.

This script builds temporal tabular features from sequence windows and trains
an sklearn classifier (default: HistGradientBoostingClassifier).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from toric_markov_model.data.trading_dataset_v3 import TradingDatasetV3


def parse_int_list(value: str) -> list[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("List cannot be empty")
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("All values must be > 0")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to merged CSV")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--prediction-horizon", type=int, default=4)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--min-pattern-profit", type=float, default=0.003)
    parser.add_argument("--windows", type=parse_int_list, default=[3, 5, 10], help="Rolling windows, e.g. 3,5,10")
    parser.add_argument("--lags", type=parse_int_list, default=[1, 2, 3], help="Lag steps, e.g. 1,2,3")
    parser.add_argument("--model", type=str, default="hist_gb", choices=["hist_gb", "rf"])
    parser.add_argument("--threshold", type=float, default=0.50, help="Operating threshold for metrics")
    parser.add_argument("--out-dir", type=str, default="checkpoints_gate_baseline")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_gate_tabular(
    dataset: TradingDatasetV3,
    windows: list[int],
    lags: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    seq_len = dataset.seq_len
    horizon = dataset.prediction_horizon
    total = len(dataset.features) - seq_len - horizon
    num_features = dataset.features.shape[1]

    rows: list[np.ndarray] = []
    labels: list[int] = []

    for idx in range(total):
        seq = dataset.features[idx:idx + seq_len]  # [seq_len, feat]
        y = int(dataset.patterns[idx + seq_len, :-1].sum() > 0.0)

        parts = [seq[-1]]  # last timestep

        for lag in lags:
            if lag <= seq_len:
                parts.append(seq[-lag])
            else:
                parts.append(np.zeros(num_features, dtype=np.float32))

        for w in windows:
            w_eff = min(w, seq_len)
            chunk = seq[-w_eff:]
            parts.append(chunk.mean(axis=0))
            parts.append(chunk.std(axis=0))
            parts.append(chunk.min(axis=0))
            parts.append(chunk.max(axis=0))

        row = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        rows.append(row)
        labels.append(y)

    x = np.vstack(rows).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int32)
    return x, y


def threshold_sweep_f1(
    probs: np.ndarray,
    targets: np.ndarray,
    min_thr: float = 0.01,
    max_thr: float = 0.99,
    step: float = 0.01,
) -> tuple[float, float, float, float]:
    best_f1 = -1.0
    best_thr = min_thr
    best_p = 0.0
    best_r = 0.0
    thr = min_thr
    while thr <= max_thr + 1e-9:
        pred = probs >= thr
        p, r, f1, _ = precision_recall_fscore_support(
            targets,
            pred,
            average="binary",
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
            best_p = float(p)
            best_r = float(r)
        thr += step
    return best_thr, best_p, best_r, best_f1


def evaluate_at_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    pred = probs >= threshold
    p, r, f1, _ = precision_recall_fscore_support(
        targets,
        pred,
        average="binary",
        zero_division=0,
    )
    acc = accuracy_score(targets, pred)
    return {
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "accuracy": float(acc),
        "positive_rate": float(pred.mean()),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GATE BASELINE TRAINING (HOLD vs NON-HOLD)")
    print("=" * 80)

    train_ds = TradingDatasetV3(
        csv_path=args.data,
        seq_len=args.seq_len,
        prediction_horizon=args.prediction_horizon,
        train=True,
        train_split=args.train_split,
        min_pattern_profit=args.min_pattern_profit,
        return_aux_targets=False,
        verbose=False,
    )
    norm_stats = train_ds.get_normalization_stats()
    aux_stats = train_ds.get_aux_target_stats()
    val_ds = TradingDatasetV3(
        csv_path=args.data,
        seq_len=args.seq_len,
        prediction_horizon=args.prediction_horizon,
        train=False,
        train_split=args.train_split,
        min_pattern_profit=args.min_pattern_profit,
        normalization_stats=norm_stats,
        aux_target_stats=aux_stats,
        return_aux_targets=False,
        verbose=False,
    )

    x_train, y_train = make_gate_tabular(train_ds, args.windows, args.lags)
    x_val, y_val = make_gate_tabular(val_ds, args.windows, args.lags)

    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    pos_weight = neg_count / (pos_count + 1e-6)
    sample_weight = np.where(y_train > 0, pos_weight, 1.0).astype(np.float32)

    print(f"Train samples: {len(y_train):,}, positives: {pos_count:,}, negatives: {neg_count:,}, pos_weight={pos_weight:.2f}")
    print(f"Val samples:   {len(y_val):,}, positives: {int(y_val.sum()):,}, negatives: {int(len(y_val)-y_val.sum()):,}")
    print(f"Tabular features: {x_train.shape[1]} (windows={args.windows}, lags={args.lags})")

    if args.model == "hist_gb":
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=600,
            max_depth=8,
            min_samples_leaf=50,
            l2_regularization=1e-3,
            random_state=args.seed,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=600,
            max_depth=16,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.seed,
        )

    print(f"Training model: {args.model}")
    if args.model == "hist_gb":
        model.fit(x_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(x_train, y_train)

    train_probs = model.predict_proba(x_train)[:, 1]
    val_probs = model.predict_proba(x_val)[:, 1]

    train_metrics = evaluate_at_threshold(train_probs, y_train, args.threshold)
    val_metrics = evaluate_at_threshold(val_probs, y_val, args.threshold)
    best_thr, best_p, best_r, best_f1 = threshold_sweep_f1(val_probs, y_val)

    print("\nFixed threshold metrics")
    print(
        f"  Train @ {args.threshold:.2f}: F1={train_metrics['f1']*100:.2f}% "
        f"(P={train_metrics['precision']*100:.2f}%, R={train_metrics['recall']*100:.2f}%), "
        f"Acc={train_metrics['accuracy']*100:.2f}%, PosRate={train_metrics['positive_rate']*100:.2f}%"
    )
    print(
        f"  Val   @ {args.threshold:.2f}: F1={val_metrics['f1']*100:.2f}% "
        f"(P={val_metrics['precision']*100:.2f}%, R={val_metrics['recall']*100:.2f}%), "
        f"Acc={val_metrics['accuracy']*100:.2f}%, PosRate={val_metrics['positive_rate']*100:.2f}%"
    )
    print(
        f"\nBest val threshold: {best_thr:.2f} -> F1={best_f1*100:.2f}% "
        f"(P={best_p*100:.2f}%, R={best_r*100:.2f}%)"
    )

    model_path = out_dir / "gate_baseline.pkl"
    meta_path = out_dir / "gate_baseline_meta.json"
    joblib.dump(model, model_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "data": args.data,
                "model": args.model,
                "seq_len": args.seq_len,
                "prediction_horizon": args.prediction_horizon,
                "train_split": args.train_split,
                "min_pattern_profit": args.min_pattern_profit,
                "windows": args.windows,
                "lags": args.lags,
                "threshold": args.threshold,
                "val_best_threshold": best_thr,
                "val_best_f1": best_f1,
                "val_best_precision": best_p,
                "val_best_recall": best_r,
            },
            f,
            indent=2,
        )
    print(f"\nSaved model: {model_path}")
    print(f"Saved meta:  {meta_path}")


if __name__ == "__main__":
    main()
